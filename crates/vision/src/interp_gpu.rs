//! RIFE's IFNet on the GPU (feature `model`; used on macOS): `rife.wgsl`'s passes through wgpu,
//! one command buffer a frame, the weights uploaded once and the feature maps kept for the next
//! frame of the same size.
//!
//! candle runs the network on Metal as hundreds of separate whole-frame operations — a 1080p
//! frame took 1.1 s, its feature head alone 0.1 s and each warp 20 ms — so the passes here write
//! straight into channel ranges of shared maps (the blocks' inputs are never concatenated) and
//! fuse each convolution with its bias, activation and residual. Every failure is an error the
//! interpolator answers on candle: no adapter, a device error or loss (from then on candle for
//! the session), a frame the device's buffers cannot hold, a read-back that does not come.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use wgpu::util::DeviceExt;

use crate::VisionError;

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("interpolator GPU: {e}"))
}

/// Set once the device reported an error or was lost: candle for the rest of the session.
static FAILED: AtomicBool = AtomicBool::new(false);

/// A 3 × 3 convolution's weights (`[cout][cin][3][3]`) or a 4 × 4 transposed one's
/// (`[cin][cout][4][4]`), its bias and, for a residual unit, its beta.
pub(crate) struct ConvWeights {
    pub cin: usize,
    pub cout: usize,
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub beta: Option<Vec<f32>>,
}

/// The network's weights in order: the head's three convolutions and its transposed one, then
/// each block's two halving convolutions, eight residual units and transposed convolution.
pub(crate) struct NetWeights {
    pub head: [ConvWeights; 4],
    pub blocks: Vec<(ConvWeights, ConvWeights, Vec<ConvWeights>, ConvWeights, usize)>,
}

struct Conv {
    cin: u32,
    cout: u32,
    wt: wgpu::Buffer,
    bias: wgpu::Buffer,
    beta: Option<wgpu::Buffer>,
}

struct Blk {
    c0: Conv,
    c1: Conv,
    res: Vec<Conv>,
    last: Conv,
    c: u32,
}

struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    pipes: [wgpu::ComputePipeline; 7],
    /// The convolution for few output channels: (at most 16, at most 32).
    conv16: wgpu::ComputePipeline,
    conv32: wgpu::ComputePipeline,
    dummy: wgpu::Buffer,
}

const CONV: usize = 0;
const CONVT: usize = 1;
const RESIZE: usize = 2;
const WARP: usize = 3;
const AXPY: usize = 4;
const FILL: usize = 5;
const MIX: usize = 6;
const CONV16: usize = 7;
const CONV32: usize = 8;

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
            let source = include_str!("rife.wgsl");
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("filmcraft-rife"), source: wgpu::ShaderSource::Wgsl(source.into()) });
            // the same passes with the convolution's tile reshaped (`rife.wgsl`'s tile lines)
            let tile = "const BM: u32 = 64u;\nconst BN: u32 = 64u;\nconst BK: u32 = 16u;\nconst WX: u32 = 16u;\nconst WY: u32 = 16u;\n\nvar<workgroup> a_tile: array<f32, 1024>;\nvar<workgroup> b_tile: array<f32, 1024>;";
            let shaped = |bm: u32, bn: u32, bk: u32, wx: u32, wy: u32| {
                let lines = format!(
                    "const BM: u32 = {bm}u;\nconst BN: u32 = {bn}u;\nconst BK: u32 = {bk}u;\nconst WX: u32 = {wx}u;\nconst WY: u32 = {wy}u;\n\nvar<workgroup> a_tile: array<f32, {}>;\nvar<workgroup> b_tile: array<f32, {}>;",
                    bm * bk,
                    bn * bk
                );
                source.contains(tile).then(|| device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("filmcraft-rife-tile"), source: wgpu::ShaderSource::Wgsl(source.replace(tile, &lines).into()) }))
            };
            let (m16, m32) = (shaped(256, 16, 8, 64, 4)?, shaped(128, 32, 16, 32, 8)?);
            let e = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let ro = wgpu::BufferBindingType::Storage { read_only: true };
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("rife"),
                entries: &[e(0, wgpu::BufferBindingType::Uniform), e(1, ro), e(2, ro), e(3, ro), e(4, ro), e(5, wgpu::BufferBindingType::Storage { read_only: false })],
            });
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("rife"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
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
            let pipes = [pipe("conv"), pipe("convt"), pipe("resize"), pipe("warp"), pipe("axpy"), pipe("fill"), pipe("mix")];
            let conv_in = |module: &wgpu::ShaderModule| {
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("conv-tile"),
                    layout: Some(&pl),
                    module,
                    entry_point: Some("conv"),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            let (conv16, conv32) = (conv_in(&m16), conv_in(&m32));
            let dummy = device.create_buffer(&wgpu::BufferDescriptor { label: Some("rife-dummy"), size: 16, usage: wgpu::BufferUsages::STORAGE, mapped_at_creation: false });
            (!FAILED.load(Ordering::SeqCst)).then_some(Device { device, queue, bgl, pipes, conv16, conv32, dummy })
        })
        .as_ref()
        .filter(|_| !FAILED.load(Ordering::SeqCst))
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One pass's numbers (`rife.wgsl`'s `P`).
#[derive(Clone, Copy, Default)]
struct P {
    sc: u32,
    sh: u32,
    sw: u32,
    dh: u32,
    dw: u32,
    n: u32,
    s_off: u32,
    s_total: u32,
    d_off: u32,
    d_total: u32,
    cout: u32,
    stride: u32,
    act: u32,
    shuffle: u32,
    x_off: u32,
    split: u32,
    factor: f32,
    keep: f32,
}

impl P {
    fn bytes(&self) -> Vec<u8> {
        let u = [
            self.sc,
            self.sh,
            self.sw,
            self.dh,
            self.dw,
            self.n,
            self.s_off,
            self.s_total,
            self.d_off,
            self.d_total,
            self.cout,
            self.stride,
            self.act,
            self.shuffle,
            self.x_off,
            self.split,
        ];
        let mut v: Vec<u8> = u.iter().flat_map(|x| x.to_le_bytes()).collect();
        v.extend(self.factor.to_le_bytes());
        v.extend(self.keep.to_le_bytes());
        v.extend([0u8; 8]);
        v
    }
}

/// A frame's feature maps (sizes in the padded frame's pixels `w` × `h`).
struct Maps {
    w: usize,
    h: usize,
    imgs: wgpu::Buffer,
    feats: wgpu::Buffer,
    ha: wgpu::Buffer,
    hb: wgpu::Buffer,
    flow: wgpu::Buffer,
    mf: wgpu::Buffer,
    x: wgpu::Buffer,
    xs: wgpu::Buffer,
    c0: wgpu::Buffer,
    ra: wgpu::Buffer,
    rb: wgpu::Buffer,
    tmps: wgpu::Buffer,
    tmp: wgpu::Buffer,
    wab: wgpu::Buffer,
    out: wgpu::Buffer,
    read: wgpu::Buffer,
}

/// The network's layers on the GPU.
pub(crate) struct GpuNet {
    dev: &'static Device,
    head: [Conv; 4],
    blocks: Vec<Blk>,
    maps: Mutex<Option<Maps>>,
}

impl GpuNet {
    /// The layers uploaded, or `None` without a usable GPU (or weights of the wrong sizes).
    pub(crate) fn new(w: &NetWeights) -> Option<Self> {
        let dev = shared()?;
        let up = |label: &str, v: &[f32]| {
            dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &bytes(v), usage: wgpu::BufferUsages::STORAGE })
        };
        // a 3 × 3 convolution's weights as `[cin · 9][cout]`; a transposed one's as they are
        let conv = |c: &ConvWeights, transposed: bool| -> Option<Conv> {
            let taps = if transposed { 16 } else { 9 };
            if c.weight.len() != c.cin.checked_mul(c.cout)?.checked_mul(taps)? || c.bias.len() != c.cout || c.beta.as_ref().is_some_and(|b| b.len() != c.cout) {
                return None;
            }
            let wt = if transposed {
                c.weight.clone()
            } else {
                let k = c.cin * 9;
                let mut v = vec![0f32; k * c.cout];
                for co in 0..c.cout {
                    for kk in 0..k {
                        v[kk * c.cout + co] = c.weight[co * k + kk];
                    }
                }
                v
            };
            Some(Conv {
                cin: u32::try_from(c.cin).ok()?,
                cout: u32::try_from(c.cout).ok()?,
                wt: up("rife-w", &wt),
                bias: up("rife-b", &c.bias),
                beta: c.beta.as_ref().map(|b| up("rife-beta", b)),
            })
        };
        let [h0, h1, h2, h3] = &w.head;
        let head = [conv(h0, false)?, conv(h1, false)?, conv(h2, false)?, conv(h3, true)?];
        let mut blocks = Vec::new();
        for (c0, c1, res, last, c) in &w.blocks {
            blocks.push(Blk {
                c0: conv(c0, false)?,
                c1: conv(c1, false)?,
                res: res.iter().map(|r| conv(r, false)).collect::<Option<Vec<_>>>()?,
                last: conv(last, true)?,
                c: u32::try_from(*c).ok()?,
            });
        }
        (!FAILED.load(Ordering::SeqCst)).then_some(Self { dev, head, blocks, maps: Mutex::new(None) })
    }

    fn maps(&self, w: usize, h: usize, scales: &[usize; 5]) -> Result<Maps, VisionError> {
        let d = &self.dev.device;
        let limits = d.limits();
        let most = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let full = (w * h) as u64 * 4;
        let mut need: Vec<(&str, u64)> = vec![
            ("imgs", 6 * full),
            ("feats", 8 * full),
            ("ha", 16 * full / 4 + 64),
            ("hb", 16 * full / 4 + 64),
            ("flow", 4 * full),
            ("mf", 9 * full),
            ("x", 28 * full),
            ("tmp", 13 * full),
            ("wab", 6 * full),
            ("out", 3 * full),
        ];
        // the largest of each block's maps (at 1/s, 1/2s, 1/4s of the frame)
        let (mut xs, mut c0, mut r, mut ts) = (16u64, 16u64, 16u64, 16u64);
        for (b, s) in self.blocks.iter().zip(scales) {
            let s = *s as u64;
            let at = |k: u64| full / (k * k) + 64;
            xs = xs.max(28 * at(s));
            c0 = c0.max(u64::from(b.c / 2) * at(2 * s));
            r = r.max(u64::from(b.c) * at(4 * s));
            ts = ts.max(13 * at(s));
        }
        need.extend([("xs", xs), ("c0", c0), ("ra", r), ("rb", r), ("tmps", ts)]);
        if need.iter().any(|(_, n)| *n > most) {
            return Err(err("the frame is larger than the device's buffers"));
        }
        let buf = |label: &str, size: u64, extra: wgpu::BufferUsages| {
            d.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size, usage: wgpu::BufferUsages::STORAGE | extra, mapped_at_creation: false })
        };
        let size = |n: &str| need.iter().find(|(k, _)| *k == n).map_or(16, |(_, v)| *v);
        let copy = wgpu::BufferUsages::COPY_DST;
        Ok(Maps {
            w,
            h,
            imgs: buf("rife-imgs", size("imgs"), copy),
            feats: buf("rife-feats", size("feats"), wgpu::BufferUsages::empty()),
            ha: buf("rife-ha", size("ha"), wgpu::BufferUsages::empty()),
            hb: buf("rife-hb", size("hb"), wgpu::BufferUsages::empty()),
            flow: buf("rife-flow", size("flow"), wgpu::BufferUsages::empty()),
            mf: buf("rife-mf", size("mf"), wgpu::BufferUsages::empty()),
            x: buf("rife-x", size("x"), wgpu::BufferUsages::empty()),
            xs: buf("rife-xs", size("xs"), wgpu::BufferUsages::empty()),
            c0: buf("rife-c0", size("c0"), wgpu::BufferUsages::empty()),
            ra: buf("rife-ra", size("ra"), wgpu::BufferUsages::empty()),
            rb: buf("rife-rb", size("rb"), wgpu::BufferUsages::empty()),
            tmps: buf("rife-tmps", size("tmps"), wgpu::BufferUsages::empty()),
            tmp: buf("rife-tmp", size("tmp"), wgpu::BufferUsages::empty()),
            wab: buf("rife-wab", size("wab"), wgpu::BufferUsages::empty()),
            out: buf("rife-out", size("out"), wgpu::BufferUsages::COPY_SRC),
            read: d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("rife-read"),
                size: size("out"),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        })
    }

    /// The frame at `t` between two planar RGB frames (0..1, `w` × `h`, padded to the coarsest
    /// scale's grid), looked at at `scales`: planar RGB.
    pub(crate) fn between(&self, a: &[f32], b: &[f32], w: usize, h: usize, t: f32, scales: [usize; 5]) -> Result<Vec<f32>, VisionError> {
        if FAILED.load(Ordering::SeqCst) {
            return Err(err("the device failed earlier"));
        }
        let npx = w.checked_mul(h).ok_or_else(|| err("frame too large"))?;
        if npx == 0 || a.len() != 3 * npx || b.len() != 3 * npx || scales.iter().any(|s| *s == 0 || !w.is_multiple_of(s * 8) || !h.is_multiple_of(s * 8)) {
            return Err(err("frames of the wrong size"));
        }
        let mut slot = self.maps.lock().unwrap_or_else(PoisonError::into_inner);
        if slot.as_ref().is_none_or(|m| (m.w, m.h) != (w, h)) {
            *slot = Some(self.maps(w, h, &scales)?);
        }
        let m = slot.as_ref().ok_or_else(|| err("no maps"))?;
        let d = &self.dev.device;
        let q = &self.dev.queue;
        q.write_buffer(&m.imgs, 0, &bytes(a));
        q.write_buffer(&m.imgs, (3 * npx * 4) as u64, &bytes(b));
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rife") });
        let mut keep = Vec::new();
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("rife"), timestamp_writes: None });
            let mut run = |pipe: usize,
                           p: P,
                           src: &wgpu::Buffer,
                           wt: &wgpu::Buffer,
                           bias: &wgpu::Buffer,
                           extra: &wgpu::Buffer,
                           dst: &wgpu::Buffer,
                           groups: (u32, u32, u32)| {
                let u =
                    d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("rife-p"), contents: &p.bytes(), usage: wgpu::BufferUsages::UNIFORM });
                let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("rife"),
                    layout: &self.dev.bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: wt.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: bias.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: extra.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: dst.as_entire_binding() },
                    ],
                });
                pass.set_pipeline(match pipe {
                    CONV16 => &self.dev.conv16,
                    CONV32 => &self.dev.conv32,
                    _ => &self.dev.pipes[pipe.min(MIX)],
                });
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(groups.0.max(1), groups.1.max(1), groups.2.max(1));
                keep.push(u);
            };
            let dm = &self.dev.dummy;
            let px8 = |w: usize, h: usize, n: u32| ((w as u32).div_ceil(8), (h as u32).div_ceil(8), n);
            let (w32, h32) = (w as u32, h as u32);
            // a convolution of `c` (stride 1 or 2) from `src` (channel `s_off`, `sw` × `sh`) to `dst`
            let conv = |run: &mut dyn FnMut(usize, P, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, (u32, u32, u32)),
                        c: &Conv,
                        src: &wgpu::Buffer,
                        s_off: u32,
                        (sw, sh): (u32, u32),
                        stride: u32,
                        act: u32,
                        dst: &wgpu::Buffer| {
                let (dw, dh) = (sw.div_ceil(stride), sh.div_ceil(stride));
                let p = P { sc: c.cin, sh, sw, dh, dw, s_off, cout: c.cout, stride, act, ..Default::default() };
                // the tile that fits the layer's channels
                let (pipe, bm, bn) = match c.cout {
                    0..=16 => (CONV16, 256, 16),
                    17..=32 => (CONV32, 128, 32),
                    _ => (CONV, 64, 64),
                };
                run(pipe, p, src, &c.wt, &c.bias, c.beta.as_ref().unwrap_or(dm), dst, ((dw * dh).div_ceil(bm), c.cout.div_ceil(bn), 1));
            };
            let convt = |run: &mut dyn FnMut(usize, P, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, (u32, u32, u32)),
                         c: &Conv,
                         src: &wgpu::Buffer,
                         (sw, sh): (u32, u32),
                         shuffle: u32,
                         dst: &wgpu::Buffer| {
                let p = P { sc: c.cin, sh, sw, cout: c.cout, shuffle, ..Default::default() };
                run(CONVT, p, src, &c.wt, &c.bias, dm, dst, ((2 * sw).div_ceil(8), (2 * sh).div_ceil(8), c.cout.div_ceil(4)));
            };
            let copy = |run: &mut dyn FnMut(usize, P, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, (u32, u32, u32)),
                        src: &wgpu::Buffer,
                        s_off: u32,
                        dst: &wgpu::Buffer,
                        d_off: u32,
                        n: u32,
                        keep: f32,
                        factor: f32| {
                let p = P { sh: h32, sw: w32, n, s_off, d_off, keep, factor, ..Default::default() };
                run(AXPY, p, src, dm, dm, dm, dst, px8(w, h, n));
            };
            let warp = |run: &mut dyn FnMut(usize, P, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, (u32, u32, u32)),
                        src: &wgpu::Buffer,
                        s_off: u32,
                        n: u32,
                        flow_off: u32,
                        dst: &wgpu::Buffer,
                        d_off: u32| {
                let p = P { sh: h32, sw: w32, n, s_off, d_off, x_off: flow_off, ..Default::default() };
                run(WARP, p, src, dm, dm, &m.flow, dst, px8(w, h, n));
            };
            // the features of each frame
            for (k, off) in [(0u32, 0u32), (3, 4)] {
                let half = (w32 / 2, h32 / 2);
                conv(&mut run, &self.head[0], &m.imgs, k, (w32, h32), 2, 1, &m.ha);
                conv(&mut run, &self.head[1], &m.ha, 0, half, 1, 1, &m.hb);
                conv(&mut run, &self.head[2], &m.hb, 0, half, 1, 1, &m.ha);
                let p = P { sc: 16, sh: half.1, sw: half.0, cout: 4, d_off: off, ..Default::default() };
                run(CONVT, p, &m.ha, &self.head[3].wt, &self.head[3].bias, dm, &m.feats, (w32.div_ceil(8), h32.div_ceil(8), 1));
            }
            for (i, blk) in self.blocks.iter().enumerate() {
                let s = scales[i] as u32;
                let cin = if i == 0 { 15 } else { 28 };
                if i == 0 {
                    copy(&mut run, &m.imgs, 0, &m.x, 0, 6, 0.0, 1.0);
                    copy(&mut run, &m.feats, 0, &m.x, 6, 8, 0.0, 1.0);
                } else {
                    warp(&mut run, &m.imgs, 0, 3, 0, &m.x, 0);
                    warp(&mut run, &m.imgs, 3, 3, 2, &m.x, 3);
                    warp(&mut run, &m.feats, 0, 4, 0, &m.x, 6);
                    warp(&mut run, &m.feats, 4, 4, 2, &m.x, 10);
                    copy(&mut run, &m.mf, 0, &m.x, 15, 9, 0.0, 1.0);
                    copy(&mut run, &m.flow, 0, &m.x, 24, 4, 0.0, 1.0);
                }
                run(FILL, P { sh: h32, sw: w32, d_off: 14, factor: t, ..Default::default() }, dm, dm, dm, dm, &m.x, px8(w, h, 1));
                let (bw, bh) = (w32 / s, h32 / s);
                let input = if s > 1 {
                    let p = P { sh: h32, sw: w32, dh: bh, dw: bw, n: cin, split: if i == 0 { cin } else { 24 }, factor: 1.0 / s as f32, ..Default::default() };
                    run(RESIZE, p, &m.x, dm, dm, dm, &m.xs, (bw.div_ceil(8), bh.div_ceil(8), cin));
                    &m.xs
                } else {
                    &m.x
                };
                conv(&mut run, &blk.c0, input, 0, (bw, bh), 2, 1, &m.c0);
                let q2 = (bw / 2, bh / 2);
                conv(&mut run, &blk.c1, &m.c0, 0, q2, 2, 1, &m.ra);
                let q4 = (bw / 4, bh / 4);
                for (k, r) in blk.res.iter().enumerate() {
                    let (from, to) = if k % 2 == 0 { (&m.ra, &m.rb) } else { (&m.rb, &m.ra) };
                    conv(&mut run, r, from, 0, q4, 1, 2, to);
                }
                let last = if blk.res.len() % 2 == 0 { &m.ra } else { &m.rb };
                convt(&mut run, &blk.last, last, q4, 1, &m.tmps);
                let tmp = if s > 1 {
                    let p = P { sh: bh, sw: bw, dh: h32, dw: w32, n: 13, split: 13, factor: 1.0, ..Default::default() };
                    run(RESIZE, p, &m.tmps, dm, dm, dm, &m.tmp, px8(w, h, 13));
                    &m.tmp
                } else {
                    &m.tmps
                };
                copy(&mut run, tmp, 0, &m.flow, 0, 4, if i == 0 { 0.0 } else { 1.0 }, s as f32);
                copy(&mut run, tmp, 4, &m.mf, 0, 9, 0.0, 1.0);
            }
            warp(&mut run, &m.imgs, 0, 3, 0, &m.wab, 0);
            warp(&mut run, &m.imgs, 3, 3, 2, &m.wab, 3);
            run(MIX, P { sh: h32, sw: w32, x_off: 0, ..Default::default() }, &m.wab, dm, dm, &m.mf, &m.out, px8(w, h, 3));
        }
        let out_bytes = (3 * npx * 4) as u64;
        enc.copy_buffer_to_buffer(&m.out, 0, &m.read, 0, out_bytes);
        q.submit([enc.finish()]);
        if let Some(Some(e)) = poll_now(scope.pop()) {
            return Err(err(e));
        }
        let slice = m.read.slice(..out_bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        d.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) }).map_err(err)?;
        rx.recv_timeout(std::time::Duration::from_secs(30)).map_err(err)?.map_err(err)?;
        let data = slice.get_mapped_range().map_err(err)?;
        let v: Vec<f32> = data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        drop(data);
        m.read.unmap();
        if FAILED.load(Ordering::SeqCst) || v.len() != 3 * npx {
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
