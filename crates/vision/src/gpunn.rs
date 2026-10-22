//! The profile pass's convolutional layers on the GPU (feature `model`; used on macOS): the
//! person detector's backbone and hybrid encoder and the pose network's backbone as WGSL compute
//! passes (`nn.wgsl`) through wgpu, on the face detector's shared device ([`gpu::Ctx`]). The
//! transformer parts (one encoder layer, the decoder, the pose head) stay on the CPU: small
//! matrices, read back once.
//!
//! The networks are written once over [`crate::layers::Backend`]; [`Run`] records each
//! operation into a command buffer, submitted when a result is read back. Every failure is an
//! error the caller answers by running the CPU passes: no adapter, a device error or loss (from
//! then on the CPU for the session), a tensor larger than the device takes, a read-back that does
//! not come.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use wgpu::util::DeviceExt;

use crate::VisionError;
use crate::gpu;
use crate::layers::{Act, Backend, Conv, Feat};

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("GPU: {e}"))
}

/// Most bytes of idle buffers kept for the next picture.
const POOL_BYTES: u64 = 1 << 30;

/// The pipelines of `nn.wgsl` on the shared device, and the buffer pool.
pub(crate) struct Nn {
    ctx: Arc<gpu::Ctx>,
    layout: wgpu::BindGroupLayout,
    conv: wgpu::ComputePipeline,
    pointwise: wgpu::ComputePipeline,
    maxpool: wgpu::ComputePipeline,
    upsample: wgpu::ComputePipeline,
    copy: wgpu::ComputePipeline,
    pad: wgpu::ComputePipeline,
    mean: wgpu::ComputePipeline,
    attn: wgpu::ComputePipeline,
    scale: wgpu::ComputePipeline,
    dummy: wgpu::Buffer,
    pool: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
}

impl Nn {
    /// The process's GPU for the profile pass, or `None` ([`gpu::Ctx::shared`]).
    pub(crate) fn shared() -> Option<Arc<Nn>> {
        static SHARED: OnceLock<Option<Arc<Nn>>> = OnceLock::new();
        let ctx = gpu::Ctx::shared()?;
        SHARED.get_or_init(|| Self::new(ctx).map(Arc::new)).clone().filter(|_| !gpu::failed())
    }

    fn new(ctx: Arc<gpu::Ctx>) -> Option<Nn> {
        let device = ctx.device();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filmcraft-vision-nn"),
            source: wgpu::ShaderSource::Wgsl(include_str!("nn.wgsl").into()),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vision-nn"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                storage(1, true),
                storage(2, true),
                storage(3, true),
                storage(4, false),
                storage(5, true),
            ],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vision-nn"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vision-nn-dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let nn = Nn {
            conv: pipeline("conv"),
            pointwise: pipeline("pointwise"),
            maxpool: pipeline("maxpool"),
            upsample: pipeline("upsample"),
            copy: pipeline("copy"),
            pad: pipeline("pad"),
            mean: pipeline("mean"),
            attn: pipeline("attn"),
            scale: pipeline("scale"),
            layout,
            dummy,
            pool: Mutex::new(HashMap::new()),
            ctx,
        };
        (!gpu::failed()).then_some(nn)
    }

    /// Values in a storage buffer.
    pub(crate) fn upload(&self, data: &[f32]) -> wgpu::Buffer {
        self.ctx.upload(data)
    }
}

/// A network's GPU and how its runs there went.
pub(crate) struct GpuUse {
    nn: Arc<Nn>,
    failures: AtomicU32,
}

impl GpuUse {
    pub(crate) fn new(nn: Arc<Nn>) -> Self {
        Self { nn, failures: AtomicU32::new(0) }
    }

    /// `f` on the GPU, or `None` for the caller to run the CPU path: no GPU, it failed
    /// [`crate::MAX_GPU_FAILURES`] times in a row, or this run failed or panicked.
    pub(crate) fn run<T>(g: &Option<GpuUse>, f: impl FnOnce(&Run<'_>) -> Result<T, VisionError>) -> Option<T> {
        let g = g.as_ref()?;
        if g.failures.load(Ordering::Relaxed) >= crate::MAX_GPU_FAILURES || gpu::failed() {
            return None;
        }
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(&Run::new(&g.nn)))) {
            Ok(Ok(v)) => {
                g.failures.store(0, Ordering::Relaxed);
                Some(v)
            }
            Ok(Err(_)) | Err(_) => {
                g.failures.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    pub(crate) fn active(g: &Option<GpuUse>) -> bool {
        !gpu::failed() && g.as_ref().is_some_and(|g| g.failures.load(Ordering::Relaxed) < crate::MAX_GPU_FAILURES)
    }
}

/// A convolution's weights on the GPU.
pub(crate) struct ConvW {
    pub(crate) w: wgpu::Buffer,
    pub(crate) b: wgpu::Buffer,
}

/// A tensor on the GPU: `n` pictures of `c` planes of `h` × `w`.
pub(crate) struct GT {
    buf: wgpu::Buffer,
    n: usize,
    c: usize,
    h: usize,
    w: usize,
}

/// The parameters of one pass (`P` in `nn.wgsl`).
#[derive(Clone, Copy, Default)]
struct Params {
    n: usize,
    cin: usize,
    h: usize,
    w: usize,
    cout: usize,
    oh: usize,
    ow: usize,
    k: usize,
    stride: usize,
    pad_t: usize,
    pad_l: usize,
    flags: u32,
    act: u32,
    act_a: f32,
    act_b: f32,
    cin_off: usize,
    cin_total: usize,
    cout_off: usize,
    cout_total: usize,
}

impl Params {
    fn bytes(&self) -> Result<Vec<u8>, VisionError> {
        let u = |v: usize| u32::try_from(v).map_err(err);
        let words: [u32; 20] = [
            u(self.n)?,
            u(self.cin)?,
            u(self.h)?,
            u(self.w)?,
            u(self.cout)?,
            u(self.oh)?,
            u(self.ow)?,
            u(self.k)?,
            u(self.stride)?,
            u(self.pad_t)?,
            u(self.pad_l)?,
            self.flags,
            self.act,
            self.act_a.to_bits(),
            self.act_b.to_bits(),
            u(self.cin_off)?,
            u(self.cin_total)?,
            u(self.cout_off)?,
            u(self.cout_total)?,
            0,
        ];
        Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect())
    }
}

/// One picture's (or batch's) passes, recorded into a command buffer that [`Backend::read`]
/// submits; recording may go on after a read.
pub(crate) struct Run<'a> {
    nn: &'a Nn,
    enc: RefCell<Option<wgpu::CommandEncoder>>,
    held: RefCell<Vec<(u64, wgpu::Buffer)>>,
}

impl Drop for Run<'_> {
    fn drop(&mut self) {
        let held = std::mem::take(&mut *self.held.borrow_mut());
        let mut pool = self.nn.pool.lock().unwrap_or_else(PoisonError::into_inner);
        let mut total: u64 = pool.iter().map(|(k, v)| k.saturating_mul(v.len() as u64)).sum();
        for (size, b) in held {
            if total.saturating_add(size) <= POOL_BYTES {
                total = total.saturating_add(size);
                pool.entry(size).or_default().push(b);
            }
        }
    }
}

fn act_code(a: Act) -> (u32, f32, f32) {
    match a {
        Act::None => (0, 0.0, 0.0),
        Act::Relu => (1, 0.0, 0.0),
        Act::ReluAffine(s, b) => (2, s, b),
        Act::Silu => (3, 0.0, 0.0),
    }
}

impl<'a> Run<'a> {
    pub(crate) fn new(nn: &'a Nn) -> Self {
        Self { nn, enc: RefCell::new(Some(Self::encoder(nn))), held: RefCell::new(Vec::new()) }
    }

    fn encoder(nn: &Nn) -> wgpu::CommandEncoder {
        nn.ctx.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vision-nn") })
    }

    fn alloc(&self, n: usize, c: usize, h: usize, w: usize) -> Result<GT, VisionError> {
        let floats = n.checked_mul(c).and_then(|x| x.checked_mul(h)).and_then(|x| x.checked_mul(w)).ok_or_else(|| err("tensor too large"))?;
        let size = (floats.max(1) as u64).checked_mul(4).ok_or_else(|| err("tensor too large"))?;
        let limits = self.nn.ctx.device().limits();
        if size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size {
            return Err(err(format!("a {size}-byte tensor is more than the device takes")));
        }
        let pooled = self.nn.pool.lock().unwrap_or_else(PoisonError::into_inner).get_mut(&size).and_then(Vec::pop);
        let buf = pooled.unwrap_or_else(|| {
            self.nn.ctx.device().create_buffer(&wgpu::BufferDescriptor {
                label: Some("vision-nn-map"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        self.held.borrow_mut().push((size, buf.clone()));
        Ok(GT { buf, n, c, h, w })
    }

    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        pipeline: &wgpu::ComputePipeline,
        p: &Params,
        src: &wgpu::Buffer,
        wt: Option<&wgpu::Buffer>,
        bias: Option<&wgpu::Buffer>,
        dst: &wgpu::Buffer,
        res: Option<&wgpu::Buffer>,
        groups: (usize, usize, usize),
    ) -> Result<(), VisionError> {
        let max = self.nn.ctx.device().limits().max_compute_workgroups_per_dimension;
        let g = |v: usize| u32::try_from(v.max(1)).map_err(err).and_then(|v| if v > max { Err(err("a tensor larger than one dispatch")) } else { Ok(v) });
        let (gx, gy, gz) = (g(groups.0)?, g(groups.1)?, g(groups.2)?);
        let d = self.nn.ctx.device();
        let uniform = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vision-nn-params"),
            contents: &p.bytes()?,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dummy = &self.nn.dummy;
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vision-nn"),
            layout: &self.nn.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wt.unwrap_or(dummy).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bias.unwrap_or(dummy).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: res.unwrap_or(dummy).as_entire_binding() },
            ],
        });
        let mut enc = self.enc.borrow_mut();
        let enc = enc.as_mut().ok_or_else(|| err("the run has ended"))?;
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("vision-nn"), timestamp_writes: None });
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(gx, gy, gz);
        Ok(())
    }

    /// Channels `from..from + c` of `x` into channels `at..` of `dst` (same pictures and size).
    fn copy_into(&self, x: &GT, from: usize, c: usize, dst: &GT, at: usize) -> Result<(), VisionError> {
        if x.n != dst.n || x.h != dst.h || x.w != dst.w || from + c > x.c || at + c > dst.c {
            return Err(err("copy shapes"));
        }
        let p = Params {
            n: x.n,
            cin: c,
            h: x.h,
            w: x.w,
            cout: c,
            oh: x.h,
            ow: x.w,

            cin_off: from,
            cin_total: x.c,
            cout_off: at,
            cout_total: dst.c,
            ..Params::default()
        };
        self.pass(&self.nn.copy, &p, &x.buf, None, None, &dst.buf, None, ((x.h * x.w).div_ceil(64), c, x.n))
    }
}

impl Backend for Run<'_> {
    type T = GT;

    fn input(&self, x: &Feat) -> Result<GT, VisionError> {
        let t = self.alloc(x.n, x.c, x.h, x.w)?;
        let bytes: Vec<u8> = x.v.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.nn.ctx.queue().write_buffer(&t.buf, 0, &bytes);
        Ok(t)
    }

    fn conv(&self, c: &Conv, x: &GT, res: Option<&GT>) -> Result<GT, VisionError> {
        let dev = c.dev.as_ref().ok_or_else(|| err("a convolution's weights are not on the GPU"))?;
        if x.c != c.cin {
            return Err(err(format!("a convolution takes {} channels, got {}", c.cin, x.c)));
        }
        let (oh, ow) = c.out_size(x.h, x.w)?;
        if res.is_some_and(|r| (r.n, r.c, r.h, r.w) != (x.n, c.cout, oh, ow)) {
            return Err(err("a residual of another shape"));
        }
        let y = self.alloc(x.n, c.cout, oh, ow)?;
        let (act, act_a, act_b) = act_code(c.act);
        let p = Params {
            n: x.n,
            cin: c.cin,
            h: x.h,
            w: x.w,
            cout: c.cout,
            oh,
            ow,
            k: c.k,
            stride: c.stride,
            pad_t: c.pad,
            pad_l: c.pad,
            flags: u32::from(c.is_depthwise()) | if res.is_some() { 2 } else { 0 },
            act,
            act_a,
            act_b,
            cin_off: 0,
            cin_total: x.c,
            cout_off: 0,
            cout_total: c.cout,
        };
        let res = res.map(|r| &r.buf);
        if c.k == 1 && c.stride == 1 && c.pad == 0 && !c.is_depthwise() {
            self.pass(&self.nn.pointwise, &p, &x.buf, Some(&dev.w), Some(&dev.b), &y.buf, res, ((oh * ow).div_ceil(16), c.cout.div_ceil(16), x.n))?;
        } else {
            self.pass(&self.nn.conv, &p, &x.buf, Some(&dev.w), Some(&dev.b), &y.buf, res, ((oh * ow).div_ceil(64), c.cout.div_ceil(4), x.n))?;
        }
        Ok(y)
    }

    fn concat(&self, parts: &[&GT]) -> Result<GT, VisionError> {
        let Some(first) = parts.first() else { return Err(err("nothing to join")) };
        let c: usize = parts.iter().map(|p| p.c).sum();
        let out = self.alloc(first.n, c, first.h, first.w)?;
        let mut at = 0;
        for p in parts {
            self.copy_into(p, 0, p.c, &out, at)?;
            at += p.c;
        }
        Ok(out)
    }

    fn channels(&self, x: &GT, from: usize, to: usize) -> Result<GT, VisionError> {
        if from > to || to > x.c {
            return Err(err("channel range"));
        }
        let out = self.alloc(x.n, to - from, x.h, x.w)?;
        self.copy_into(x, from, to - from, &out, 0)?;
        Ok(out)
    }

    fn pad_rb(&self, x: &GT) -> Result<GT, VisionError> {
        let (oh, ow) = (x.h + 1, x.w + 1);
        let out = self.alloc(x.n, x.c, oh, ow)?;
        let p = Params { n: x.n, cin: x.c, h: x.h, w: x.w, cout: x.c, oh, ow, cin_total: x.c, cout_total: x.c, ..Params::default() };
        self.pass(&self.nn.pad, &p, &x.buf, None, None, &out.buf, None, ((oh * ow).div_ceil(64), x.c, x.n))?;
        Ok(out)
    }

    fn max_pool(&self, x: &GT, k: usize, s: usize, pad: usize) -> Result<GT, VisionError> {
        let s = s.max(1);
        let (Some(a), Some(b)) = ((x.h + 2 * pad).checked_sub(k), (x.w + 2 * pad).checked_sub(k)) else {
            return Err(err("pooling window larger than the map"));
        };
        let (oh, ow) = (a / s + 1, b / s + 1);
        let out = self.alloc(x.n, x.c, oh, ow)?;
        let p = Params {
            n: x.n,
            cin: x.c,
            h: x.h,
            w: x.w,
            cout: x.c,
            oh,
            ow,
            k,
            stride: s,
            pad_t: pad,
            pad_l: pad,
            cin_total: x.c,
            cout_total: x.c,
            ..Params::default()
        };
        self.pass(&self.nn.maxpool, &p, &x.buf, None, None, &out.buf, None, ((oh * ow).div_ceil(64), x.c, x.n))?;
        Ok(out)
    }

    fn upsample2(&self, x: &GT) -> Result<GT, VisionError> {
        let (oh, ow) = (x.h * 2, x.w * 2);
        let out = self.alloc(x.n, x.c, oh, ow)?;
        let p = Params { n: x.n, cin: x.c, h: x.h, w: x.w, cout: x.c, oh, ow, cin_total: x.c, cout_total: x.c, ..Params::default() };
        self.pass(&self.nn.upsample, &p, &x.buf, None, None, &out.buf, None, ((oh * ow).div_ceil(64), x.c, x.n))?;
        Ok(out)
    }

    fn chan_attn(&self, x: &GT, fc: &Conv) -> Result<GT, VisionError> {
        let dev = fc.dev.as_ref().ok_or_else(|| err("the attention's weights are not on the GPU"))?;
        if fc.cin != x.c || fc.cout != x.c || fc.k != 1 {
            return Err(err("channel attention shapes"));
        }
        let means = self.alloc(x.n, x.c, 1, 1)?;
        let scales = self.alloc(x.n, x.c, 1, 1)?;
        let p = Params { n: x.n, cin: x.c, h: x.h, w: x.w, cout: x.c, oh: x.h, ow: x.w, cin_total: x.c, cout_total: x.c, ..Params::default() };
        self.pass(&self.nn.mean, &p, &x.buf, None, None, &means.buf, None, (x.c.div_ceil(64), 1, x.n))?;
        self.pass(&self.nn.attn, &p, &means.buf, Some(&dev.w), Some(&dev.b), &scales.buf, None, (x.c.div_ceil(64), 1, x.n))?;
        let out = self.alloc(x.n, x.c, x.h, x.w)?;
        self.pass(&self.nn.scale, &p, &x.buf, None, None, &out.buf, Some(&scales.buf), ((x.h * x.w).div_ceil(64), x.c, x.n))?;
        Ok(out)
    }

    fn shape(t: &GT) -> (usize, usize, usize, usize) {
        (t.n, t.c, t.h, t.w)
    }

    fn read(&self, xs: Vec<GT>) -> Result<Vec<Feat>, VisionError> {
        let enc = self.enc.borrow_mut().replace(Self::encoder(self.nn)).ok_or_else(|| err("the run has ended"))?;
        let mut enc = enc;
        let d = self.nn.ctx.device();
        let mut staging = Vec::with_capacity(xs.len());
        for m in &xs {
            let size = (m.n * m.c * m.h * m.w).max(1) as u64 * 4;
            let s = d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vision-nn-read"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            enc.copy_buffer_to_buffer(&m.buf, 0, &s, 0, size);
            staging.push(s);
        }
        let ok = self.nn.ctx.submit_and_map(enc, &staging)?;
        if !ok {
            return Err(err("the device failed or the read-back did not come"));
        }
        let mut out = Vec::with_capacity(xs.len());
        for (m, s) in xs.iter().zip(&staging) {
            let data = s.slice(..).get_mapped_range().map_err(err)?;
            let v: Vec<f32> = data.as_chunks::<4>().0.iter().take(m.n * m.c * m.h * m.w).map(|b| f32::from_le_bytes(*b)).collect();
            drop(data);
            s.unmap();
            out.push(Feat::new(v, m.n, m.c, m.h, m.w)?);
        }
        Ok(out)
    }
}

impl Conv {
    /// Put the weights on the GPU.
    pub(crate) fn upload(&mut self, nn: &Nn) {
        let (w, b) = self.weights();
        self.dev = Some(ConvW { w: nn.upload(&w), b: nn.upload(&b) });
    }
}
