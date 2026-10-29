//! MobileSAM's image encoder on the GPU (feature `model`): TinyViT-5M and its neck as
//! `sam.wgsl`'s passes through wgpu. candle runs the encoder's depthwise convolutions as one
//! convolution a channel (256 for the first ones) and took half a second a picture on Metal; here
//! a picture is one upload, one command buffer and one read-back, every pass and binding made once
//! (the input is always 1024 × 1024). The batch norms are folded into their convolutions. Every
//! failure is an error the segmenter answers on candle: no adapter, a device error or loss (from
//! then on candle for the session), a read-back that does not come.
//!
//! The network, as candle's `segment_anything::tiny_vit` has it: two strided 3 × 3 convolutions
//! (GELU between); two MBConv blocks (1 × 1 up four times, GELU, depthwise 3 × 3, GELU, 1 × 1
//! down, the input added, GELU); patch merging (1 × 1, GELU, depthwise 3 × 3 strided, GELU, 1 × 1)
//! between the stages; three stages of TinyViT blocks — the map padded to whole windows, layer
//! norm, window attention with a learnt bias by offset, projection, cropped and added; depthwise
//! 3 × 3; layer norm, MLP (GELU) and added —; the neck: 1 × 1 to 256, a layer norm over the
//! channels, 3 × 3, a layer norm.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use candle_core::DType;
use wgpu::util::DeviceExt;

use crate::VisionError;

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("segmenter GPU: {e}"))
}

/// The input's side, and the embedding's channels and side.
pub(crate) const INPUT: usize = 1024;
pub(crate) const EMB_C: usize = 256;
pub(crate) const EMB_SIDE: usize = 64;

const DIMS: [usize; 4] = [64, 128, 160, 320];
const DEPTHS: [usize; 4] = [2, 2, 6, 2];
const HEADS: [usize; 4] = [2, 4, 5, 10];
const WINDOWS: [usize; 4] = [7, 7, 14, 7];
const MBCONV_EXPAND: usize = 4;
const MLP_RATIO: usize = 4;
const BN_EPS: f32 = 1e-5;
const LN_EPS: f32 = 1e-5;
const NECK_EPS: f32 = 1e-6;

/// Set once the device reported an error or was lost: candle for the rest of the session.
static FAILED: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------------------------
// the weights
// ---------------------------------------------------------------------------------------------

/// A matrix product's weights: `[m][k]` (a 1 × 1 convolution, a linear layer, or a 3 × 3
/// convolution with `k` its input channels × 9) and its bias; `stride` for a 3 × 3 one.
#[derive(Clone, Debug, Default)]
pub(crate) struct Lin {
    pub m: usize,
    pub k: usize,
    pub w: Vec<f32>,
    pub b: Option<Vec<f32>>,
    pub conv3: bool,
    pub stride: usize,
}

/// A depthwise 3 × 3 convolution: `[c][9]`, its bias and stride.
#[derive(Clone, Debug, Default)]
pub(crate) struct Dw3 {
    pub c: usize,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub stride: usize,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Norm {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Block {
    pub norm: Norm,
    pub qkv: Lin,
    pub proj: Lin,
    /// The position bias `[head][query][key]`.
    pub ab: Vec<f32>,
    pub local: Dw3,
    pub mlp_norm: Norm,
    pub fc1: Lin,
    pub fc2: Lin,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Merge {
    pub c1: Lin,
    pub c2: Dw3,
    pub c3: Lin,
}

/// The encoder's weights.
#[derive(Clone, Debug, Default)]
pub(crate) struct EncoderWeights {
    pub pe1: Lin,
    pub pe2: Lin,
    pub mb: Vec<(Lin, Dw3, Lin)>,
    pub merges: Vec<Merge>,
    pub stages: Vec<Vec<Block>>,
    pub neck1: Lin,
    pub ln1: Norm,
    pub neck2: Lin,
    pub ln2: Norm,
}

fn vec(vb: &candle_nn::VarBuilder, shape: &[usize], name: &str) -> Result<Vec<f32>, VisionError> {
    vb.get(shape, name).and_then(|t| t.to_dtype(DType::F32)).and_then(|t| t.flatten_all()).and_then(|t| t.to_vec1::<f32>()).map_err(err)
}

/// A convolution and its batch norm (`c.weight`, `bn.*`), the norm folded into its weights.
fn conv_bn(vb: &candle_nn::VarBuilder, cin_per_group: usize, cout: usize, k: usize) -> Result<(Vec<f32>, Vec<f32>), VisionError> {
    let mut w = vec(&vb.pp("c"), &[cout, cin_per_group, k, k], "weight")?;
    let bn = vb.pp("bn");
    let (g, b, mean, var) = (vec(&bn, &[cout], "weight")?, vec(&bn, &[cout], "bias")?, vec(&bn, &[cout], "running_mean")?, vec(&bn, &[cout], "running_var")?);
    let per = cin_per_group * k * k;
    let mut bias = vec![0f32; cout];
    for co in 0..cout {
        let s = g[co] / (var[co] + BN_EPS).sqrt();
        for v in w.iter_mut().skip(co * per).take(per) {
            *v *= s;
        }
        bias[co] = b[co] - mean[co] * s;
    }
    Ok((w, bias))
}

fn lin_bn(vb: &candle_nn::VarBuilder, cin: usize, cout: usize, k: usize, stride: usize) -> Result<Lin, VisionError> {
    let (w, b) = conv_bn(vb, cin, cout, k)?;
    Ok(Lin { m: cout, k: cin * k * k, w, b: Some(b), conv3: k == 3, stride })
}

fn dw_bn(vb: &candle_nn::VarBuilder, c: usize, stride: usize) -> Result<Dw3, VisionError> {
    let (w, b) = conv_bn(vb, 1, c, 3)?;
    Ok(Dw3 { c, w, b, stride })
}

fn linear(vb: &candle_nn::VarBuilder, cin: usize, cout: usize) -> Result<Lin, VisionError> {
    Ok(Lin { m: cout, k: cin, w: vec(vb, &[cout, cin], "weight")?, b: Some(vec(vb, &[cout], "bias")?), conv3: false, stride: 1 })
}

fn norm(vb: &candle_nn::VarBuilder, c: usize) -> Result<Norm, VisionError> {
    Ok(Norm { w: vec(vb, &[c], "weight")?, b: vec(vb, &[c], "bias")? })
}

/// The attention's position bias for a `ws` × `ws` window: `attention_biases` (`[heads][offsets]`)
/// spread to `[head][query][key]` by the two tokens' offset, as candle builds it.
fn position_bias(vb: &candle_nn::VarBuilder, heads: usize, ws: usize) -> Result<Vec<f32>, VisionError> {
    let points: Vec<(i64, i64)> = (0..ws as i64).flat_map(|x| (0..ws as i64).map(move |y| (x, y))).collect();
    let mut offsets: std::collections::HashMap<(i64, i64), usize> = std::collections::HashMap::new();
    let mut idx = Vec::with_capacity(points.len() * points.len());
    for &(x1, y1) in &points {
        for &(x2, y2) in &points {
            let o = ((x2 - x1).abs(), (y2 - y1).abs());
            let l = offsets.len();
            idx.push(*offsets.entry(o).or_insert(l));
        }
    }
    let table = vec(vb, &[heads, offsets.len()], "attention_biases")?;
    let n = points.len();
    let mut ab = vec![0f32; heads * n * n];
    for h in 0..heads {
        for (q, i) in idx.iter().enumerate() {
            ab[h * n * n + q] = table.get(h * offsets.len() + i).copied().unwrap_or(0.0);
        }
    }
    Ok(ab)
}

impl EncoderWeights {
    /// Read from MobileSAM's checkpoint (`image_encoder.*`).
    pub(crate) fn read(vb: &candle_nn::VarBuilder) -> Result<Self, VisionError> {
        let vb = vb.pp("image_encoder");
        let pe = vb.pp("patch_embed").pp("seq");
        let pe1 = lin_bn(&pe.pp("0"), 3, DIMS[0] / 2, 3, 2)?;
        let pe2 = lin_bn(&pe.pp("2"), DIMS[0] / 2, DIMS[0], 3, 2)?;
        let layers = vb.pp("layers");
        let l0 = layers.pp("0");
        let hidden = DIMS[0] * MBCONV_EXPAND;
        let mut mb = Vec::new();
        for i in 0..DEPTHS[0] {
            let b = l0.pp("blocks").pp(i);
            mb.push((lin_bn(&b.pp("conv1"), DIMS[0], hidden, 1, 1)?, dw_bn(&b.pp("conv2"), hidden, 1)?, lin_bn(&b.pp("conv3"), hidden, DIMS[0], 1, 1)?));
        }
        let merge = |vb: &candle_nn::VarBuilder, dim: usize, out: usize| -> Result<Merge, VisionError> {
            let stride = if [320, 448, 576].contains(&out) { 1 } else { 2 };
            Ok(Merge { c1: lin_bn(&vb.pp("conv1"), dim, out, 1, 1)?, c2: dw_bn(&vb.pp("conv2"), out, stride)?, c3: lin_bn(&vb.pp("conv3"), out, out, 1, 1)? })
        };
        let mut merges = vec![merge(&l0.pp("downsample"), DIMS[0], DIMS[1])?];
        let mut stages = Vec::new();
        for li in 1..DIMS.len() {
            let (dim, heads, ws) = (DIMS[li], HEADS[li], WINDOWS[li]);
            let lv = layers.pp(li);
            let mut blocks = Vec::new();
            for bi in 0..DEPTHS[li] {
                let b = lv.pp("blocks").pp(bi);
                let a = b.pp("attn");
                let m = b.pp("mlp");
                blocks.push(Block {
                    norm: norm(&a.pp("norm"), dim)?,
                    qkv: linear(&a.pp("qkv"), dim, 3 * dim)?,
                    proj: linear(&a.pp("proj"), dim, dim)?,
                    ab: position_bias(&a, heads, ws)?,
                    local: dw_bn(&b.pp("local_conv"), dim, 1)?,
                    mlp_norm: norm(&m.pp("norm"), dim)?,
                    fc1: linear(&m.pp("fc1"), dim, dim * MLP_RATIO)?,
                    fc2: linear(&m.pp("fc2"), dim * MLP_RATIO, dim)?,
                });
            }
            stages.push(blocks);
            if li + 1 < DIMS.len() {
                merges.push(merge(&lv.pp("downsample"), dim, DIMS[li + 1])?);
            }
        }
        let neck = vb.pp("neck");
        let last = DIMS[DIMS.len() - 1];
        let neck1 = Lin { m: EMB_C, k: last, w: vec(&neck.pp("0"), &[EMB_C, last, 1, 1], "weight")?, b: None, conv3: false, stride: 1 };
        let neck2 = Lin { m: EMB_C, k: EMB_C * 9, w: vec(&neck.pp("2"), &[EMB_C, EMB_C, 3, 3], "weight")?, b: None, conv3: true, stride: 1 };
        Ok(Self { pe1, pe2, mb, merges, stages, neck1, ln1: norm(&neck.pp("1"), EMB_C)?, neck2, ln2: norm(&neck.pp("3"), EMB_C)? })
    }
}

// ---------------------------------------------------------------------------------------------
// the device
// ---------------------------------------------------------------------------------------------

struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pipes: [wgpu::ComputePipeline; 5],
}

const GEMM: usize = 0;
const DW: usize = 1;
const LN: usize = 2;
const PLACE: usize = 3;
const ATTN: usize = 4;

fn shared() -> Option<&'static Device> {
    static SHARED: OnceLock<Option<Device>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            // without a backend `Instance::default()` panics
            if wgpu::Instance::enabled_backend_features().is_empty() {
                return None;
            }
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(
                instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }),
            )
            .ok()?;
            let desc = wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() };
            let (device, queue) = pollster::block_on(adapter.request_device(&desc)).ok()?;
            // wgpu's default handler panics on a validation or out-of-memory error
            device.on_uncaptured_error(Arc::new(|_: wgpu::Error| FAILED.store(true, Ordering::SeqCst)));
            device.set_device_lost_callback(|_, _| FAILED.store(true, Ordering::SeqCst));
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("filmcraft-sam"),
                source: wgpu::ShaderSource::Wgsl(include_str!("sam.wgsl").into()),
            });
            let e = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let ro = wgpu::BufferBindingType::Storage { read_only: true };
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("sam"),
                entries: &[
                    e(0, wgpu::BufferBindingType::Uniform),
                    e(1, ro),
                    e(2, ro),
                    e(3, ro),
                    e(4, ro),
                    e(5, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
            let pl =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("sam"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
            let pipe = |name: &str| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            let pipes = [pipe("gemm"), pipe("dw"), pipe("ln"), pipe("place"), pipe("attn")];
            (!FAILED.load(Ordering::SeqCst)).then_some(Device { device, queue, bgl, pipes })
        })
        .as_ref()
        .filter(|_| !FAILED.load(Ordering::SeqCst))
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One pass's numbers (`sam.wgsl`'s `P`).
#[derive(Clone, Copy, Default)]
struct P {
    m: u32,
    k: u32,
    h: u32,
    w: u32,
    oh: u32,
    ow: u32,
    stride: u32,
    flags: u32,
    heads: u32,
    ws: u32,
    kd: u32,
    nw: u32,
    eps: f32,
    scale: f32,
}

const F_BIAS: u32 = 1;
const F_ADD: u32 = 2;
const F_GELU: u32 = 4;
const F_CONV3: u32 = 8;

impl P {
    fn bytes(&self) -> Vec<u8> {
        let u = [self.m, self.k, self.h, self.w, self.oh, self.ow, self.stride, self.flags, self.heads, self.ws, self.kd, self.nw];
        let mut v: Vec<u8> = u.iter().flat_map(|x| x.to_le_bytes()).collect();
        v.extend(self.eps.to_le_bytes());
        v.extend(self.scale.to_le_bytes());
        v.extend([0u8; 8]);
        v
    }
}

struct Pass {
    pipe: usize,
    bg: wgpu::BindGroup,
    groups: (u32, u32, u32),
    /// The work map it writes (what a test reads back, finding the first pass that goes wrong).
    #[cfg_attr(not(test), allow(dead_code))]
    dst: usize,
}

/// The encoder's passes, the maps they use and the read-back.
pub(crate) struct SamGpu {
    dev: &'static Device,
    input: wgpu::Buffer,
    maps: Vec<wgpu::Buffer>,
    out: usize,
    read: wgpu::Buffer,
    passes: Vec<Pass>,
    one: Mutex<()>,
    _keep: Vec<wgpu::Buffer>,
}

/// Builds the passes over `maps` (work maps by index) and the buffers they need.
struct Plan<'a> {
    dev: &'static Device,
    keep: Vec<wgpu::Buffer>,
    passes: Vec<Pass>,
    maps: &'a [wgpu::Buffer],
    input: &'a wgpu::Buffer,
    dummy: &'a wgpu::Buffer,
}

/// A map in a plan: the input, or work map `i`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum M {
    Input,
    W(usize),
}

impl Plan<'_> {
    fn buf(&self, m: M) -> &wgpu::Buffer {
        match m {
            M::Input => self.input,
            M::W(i) => self.maps.get(i).unwrap_or(self.dummy),
        }
    }
    fn upload(&mut self, v: &[f32]) -> usize {
        let b = self.dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sam-w"),
            contents: &bytes(v),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.keep.push(b);
        self.keep.len() - 1
    }
    #[allow(clippy::too_many_arguments)]
    fn pass(&mut self, pipe: usize, p: P, src: M, wt: Option<usize>, bias: Option<usize>, add: Option<M>, dst: M, groups: (u32, u32, u32)) {
        let d = &self.dev.device;
        let u = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("sam-p"), contents: &p.bytes(), usage: wgpu::BufferUsages::UNIFORM });
        let at = |i: Option<usize>| i.and_then(|i| self.keep.get(i)).unwrap_or(self.dummy);
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("sam"),
            layout: &self.dev.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.buf(src).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: at(wt).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: at(bias).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: add.map_or(self.dummy, |a| self.buf(a)).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: self.buf(dst).as_entire_binding() },
            ],
        });
        self.keep.push(u);
        let dst = match dst {
            M::W(i) => i,
            M::Input => 0,
        };
        self.passes.push(Pass { pipe, bg, groups: (groups.0.max(1), groups.1.max(1), groups.2.max(1)), dst });
    }
    /// A matrix product of `src` (`side` × `side` input) to `dst`.
    fn gemm(&mut self, l: &Lin, src: M, side: usize, add: Option<M>, gelu: bool, dst: M) -> usize {
        let wt = self.upload(&l.w);
        let bias = l.b.as_ref().map(|b| self.upload(b));
        let o = if l.conv3 { side.div_ceil(l.stride) } else { side };
        let flags =
            if bias.is_some() { F_BIAS } else { 0 } | if add.is_some() { F_ADD } else { 0 } | if gelu { F_GELU } else { 0 } | if l.conv3 { F_CONV3 } else { 0 };
        let p = P {
            m: l.m as u32,
            k: l.k as u32,
            h: side as u32,
            w: side as u32,
            oh: o as u32,
            ow: o as u32,
            stride: l.stride.max(1) as u32,
            flags,
            ..Default::default()
        };
        self.pass(GEMM, p, src, Some(wt), bias, add, dst, (((o * o) as u32).div_ceil(64), (l.m as u32).div_ceil(64), 1));
        o
    }
    fn dw(&mut self, d: &Dw3, src: M, side: usize, gelu: bool, dst: M) -> usize {
        let wt = self.upload(&d.w);
        let bias = self.upload(&d.b);
        let o = side.div_ceil(d.stride.max(1));
        let p = P {
            m: d.c as u32,
            h: side as u32,
            w: side as u32,
            oh: o as u32,
            ow: o as u32,
            stride: d.stride.max(1) as u32,
            flags: F_BIAS | if gelu { F_GELU } else { 0 },
            ..Default::default()
        };
        self.pass(DW, p, src, Some(wt), Some(bias), None, dst, ((o as u32).div_ceil(16), (o as u32).div_ceil(16), d.c as u32));
        o
    }
    fn ln(&mut self, n: &Norm, c: usize, src: M, side: usize, eps: f32, dst: M) {
        let (wt, bias) = (self.upload(&n.w), self.upload(&n.b));
        let p = P { m: c as u32, oh: side as u32, ow: side as u32, eps, ..Default::default() };
        self.pass(LN, p, src, Some(wt), Some(bias), None, dst, (((side * side) as u32).div_ceil(64), 1, 1));
    }
    /// `src` (`c` channels, `from` × `from`) into `to` × `to` (the top-left; zeros round it, or
    /// cropped), `add` added.
    fn place(&mut self, c: usize, src: M, from: usize, to: usize, add: Option<M>, dst: M) {
        let p =
            P { m: c as u32, h: from as u32, w: from as u32, oh: to as u32, ow: to as u32, flags: if add.is_some() { F_ADD } else { 0 }, ..Default::default() };
        self.pass(PLACE, p, src, None, None, add, dst, ((to as u32).div_ceil(16), (to as u32).div_ceil(16), c as u32));
    }
}

/// A work map not among `busy`.
fn free(busy: &[usize]) -> usize {
    (0..MAPS).find(|i| !busy.contains(i)).unwrap_or(0)
}

const MAPS: usize = 6;

impl SamGpu {
    /// The encoder's passes made, or `None` without a usable GPU (or weights it cannot run).
    pub(crate) fn new(w: &EncoderWeights) -> Option<Self> {
        let dev = shared()?;
        if w.stages.len() + 1 != DIMS.len()
            || w.merges.len() != DIMS.len() - 1
            || DIMS.iter().zip(HEADS).any(|(d, h)| d % h != 0 || (d / h) % 4 != 0 || d / h > 32)
        {
            return None;
        }
        let d = &dev.device;
        let buf = |floats: usize, usage: wgpu::BufferUsages| {
            d.create_buffer(&wgpu::BufferDescriptor { label: Some("sam-map"), size: (floats.max(1) * 4) as u64, usage, mapped_at_creation: false })
        };
        let st = wgpu::BufferUsages::STORAGE;
        // the largest map: the MBConv's widened one, a stage's padded qkv, its MLP's hidden one
        let patches = INPUT / 4;
        let mut widest = (DIMS[0] / 2) * (INPUT / 2) * (INPUT / 2);
        widest = widest.max(DIMS[0] * MBCONV_EXPAND * patches * patches);
        let mut res = patches / 2;
        for (li, dim) in DIMS.iter().enumerate().skip(1) {
            let ws = WINDOWS[li];
            let padded = res.div_ceil(ws) * ws;
            widest = widest.max(3 * dim * padded * padded).max(dim * MLP_RATIO * res * res);
            if li + 1 < DIMS.len() && ![320, 448, 576].contains(&DIMS[li + 1]) {
                res /= 2;
            }
        }
        widest = widest.max(EMB_C * EMB_SIDE * EMB_SIDE);
        let input = buf(3 * INPUT * INPUT, st | wgpu::BufferUsages::COPY_DST);
        let maps: Vec<wgpu::Buffer> = (0..MAPS).map(|_| buf(widest, st | wgpu::BufferUsages::COPY_SRC)).collect();
        let read = buf(EMB_C * EMB_SIDE * EMB_SIDE, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let dummy = buf(4, st);
        let mut plan = Plan { dev, keep: Vec::new(), passes: Vec::new(), maps: &maps, input: &input, dummy: &dummy };
        // the patch embedding
        let a = 0usize;
        let side = plan.gemm(&w.pe1, M::Input, INPUT, None, true, M::W(a));
        let b = free(&[a]);
        let mut side = plan.gemm(&w.pe2, M::W(a), side, None, false, M::W(b));
        let mut x = b;
        // the MBConv blocks
        for (c1, c2, c3) in &w.mb {
            let h1 = free(&[x]);
            plan.gemm(c1, M::W(x), side, None, true, M::W(h1));
            let h2 = free(&[x, h1]);
            plan.dw(c2, M::W(h1), side, true, M::W(h2));
            let o = free(&[x, h2]);
            plan.gemm(c3, M::W(h2), side, Some(M::W(x)), true, M::W(o));
            x = o;
        }
        let merge = |plan: &mut Plan, m: &Merge, x: usize, side: usize| -> (usize, usize) {
            let h1 = free(&[x]);
            plan.gemm(&m.c1, M::W(x), side, None, true, M::W(h1));
            let h2 = free(&[h1]);
            let s = plan.dw(&m.c2, M::W(h1), side, true, M::W(h2));
            let o = free(&[h2]);
            plan.gemm(&m.c3, M::W(h2), s, None, false, M::W(o));
            (o, s)
        };
        (x, side) = merge(&mut plan, w.merges.first()?, x, side);
        for (si, blocks) in w.stages.iter().enumerate() {
            let li = si + 1;
            let (dim, heads, ws) = (DIMS[li], HEADS[li], WINDOWS[li]);
            let kd = dim / heads;
            let padded = side.div_ceil(ws) * ws;
            let nw = padded / ws;
            for blk in blocks {
                // window attention on the map padded to whole windows, cropped and added
                let xp = free(&[x]);
                plan.place(dim, M::W(x), side, padded, None, M::W(xp));
                let nm = free(&[x, xp]);
                plan.ln(&blk.norm, dim, M::W(xp), padded, LN_EPS, M::W(nm));
                let q = free(&[x, nm]);
                plan.gemm(&blk.qkv, M::W(nm), padded, None, false, M::W(q));
                let at = free(&[x, q]);
                let ab = plan.upload(&blk.ab);
                let p = P {
                    oh: padded as u32,
                    ow: padded as u32,
                    heads: heads as u32,
                    ws: ws as u32,
                    kd: kd as u32,
                    nw: nw as u32,
                    scale: 1.0 / (kd as f32).sqrt(),
                    ..Default::default()
                };
                plan.pass(ATTN, p, M::W(q), Some(ab), None, None, M::W(at), (((ws * ws) as u32).div_ceil(64), heads as u32, (nw * nw) as u32));
                let pr = free(&[x, at]);
                plan.gemm(&blk.proj, M::W(at), padded, None, false, M::W(pr));
                let x2 = free(&[x, pr]);
                plan.place(dim, M::W(pr), padded, side, Some(M::W(x)), M::W(x2));
                // the local convolution, then the MLP added
                let x3 = free(&[x2]);
                plan.dw(&blk.local, M::W(x2), side, false, M::W(x3));
                let mn = free(&[x3]);
                plan.ln(&blk.mlp_norm, dim, M::W(x3), side, LN_EPS, M::W(mn));
                let hd = free(&[x3, mn]);
                plan.gemm(&blk.fc1, M::W(mn), side, None, true, M::W(hd));
                let o = free(&[x3, hd]);
                plan.gemm(&blk.fc2, M::W(hd), side, Some(M::W(x3)), false, M::W(o));
                x = o;
            }
            if let Some(m) = w.merges.get(li) {
                (x, side) = merge(&mut plan, m, x, side);
            }
        }
        if side != EMB_SIDE {
            return None;
        }
        // the neck
        let n1 = free(&[x]);
        plan.gemm(&w.neck1, M::W(x), side, None, false, M::W(n1));
        let l1 = free(&[n1]);
        plan.ln(&w.ln1, EMB_C, M::W(n1), side, NECK_EPS, M::W(l1));
        let n2 = free(&[l1]);
        plan.gemm(&w.neck2, M::W(l1), side, None, false, M::W(n2));
        let l2 = free(&[n2]);
        plan.ln(&w.ln2, EMB_C, M::W(n2), side, NECK_EPS, M::W(l2));
        let Plan { keep, passes, .. } = plan;
        let mut keep = keep;
        keep.push(dummy);
        (!FAILED.load(Ordering::SeqCst)).then_some(Self { dev, input, maps, out: l2, read, passes, one: Mutex::new(()), _keep: keep })
    }

    /// The embedding (`[256][64][64]`) of a picture prepared for the encoder: `[3][1024][1024]`,
    /// normalised and padded with zeros.
    pub(crate) fn run(&self, x: &[f32]) -> Result<Vec<f32>, VisionError> {
        self.run_to(x, self.passes.len(), self.out, EMB_C * EMB_SIDE * EMB_SIDE)
    }

    /// The first `passes` passes run, `n` values of work map `map` read back (for tests).
    fn run_to(&self, x: &[f32], passes: usize, map: usize, n: usize) -> Result<Vec<f32>, VisionError> {
        let _one = self.one.lock().unwrap_or_else(PoisonError::into_inner);
        if FAILED.load(Ordering::SeqCst) {
            return Err(err("the device failed"));
        }
        if x.len() != 3 * INPUT * INPUT {
            return Err(err("the input is not 3 × 1024 × 1024"));
        }
        let (d, q) = (&self.dev.device, &self.dev.queue);
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        q.write_buffer(&self.input, 0, &bytes(x));
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("sam") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("sam"), timestamp_writes: None });
            for p in self.passes.iter().take(passes) {
                pass.set_pipeline(&self.dev.pipes[p.pipe.min(ATTN)]);
                pass.set_bind_group(0, &p.bg, &[]);
                pass.dispatch_workgroups(p.groups.0, p.groups.1, p.groups.2);
            }
        }
        let n = n.min(EMB_C * EMB_SIDE * EMB_SIDE);
        let out_bytes = (n * 4) as u64;
        let src = self.maps.get(map).ok_or_else(|| err("no such map"))?;
        enc.copy_buffer_to_buffer(src, 0, &self.read, 0, out_bytes);
        q.submit([enc.finish()]);
        if let Some(Some(e)) = poll_now(scope.pop()) {
            return Err(err(e));
        }
        let slice = self.read.slice(..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        d.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) }).map_err(err)?;
        rx.recv_timeout(std::time::Duration::from_secs(30)).map_err(err)?.map_err(err)?;
        let data = slice.get_mapped_range().map_err(err)?;
        let v: Vec<f32> = data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        drop(data);
        self.read.unmap();
        if FAILED.load(Ordering::SeqCst) || v.len() != n {
            return Err(err("the device failed"));
        }
        Ok(v)
    }
}

/// A future polled once (error scopes resolve at once on native backends).
fn poll_now<F: std::future::Future>(f: F) -> Option<F::Output> {
    let mut f = std::pin::pin!(f);
    let waker = std::task::Waker::noop();
    match f.as_mut().poll(&mut std::task::Context::from_waker(waker)) {
        std::task::Poll::Ready(v) => Some(v),
        std::task::Poll::Pending => None,
    }
}

#[cfg(test)]
impl SamGpu {
    /// The first pass whose output has a value that is not finite: its index and pipeline.
    pub(crate) fn first_bad(&self, x: &[f32]) -> Option<(usize, usize)> {
        (1..=self.passes.len()).find_map(|k| {
            let p = self.passes.get(k - 1)?;
            let v = self.run_to(x, k, p.dst, 65536).ok()?;
            v.iter().any(|x| !x.is_finite()).then_some((k - 1, p.pipe))
        })
    }
}
