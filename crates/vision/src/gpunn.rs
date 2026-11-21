//! The profile pass on the GPU (feature `model`; used on macOS), on the face detector's shared
//! device ([`gpu::Ctx`]) through wgpu: the person detector's backbone, hybrid encoder (its
//! transformer layer too) and decoder, and the pose network's backbone and head, as WGSL compute
//! passes — `nn.wgsl` (convolutions, tiled pointwise, pools, upsampling, copies, padding, channel
//! attention), `tok.wgsl` (matrix products, norms, attention, D-FINE's deformable sampling, gate,
//! query selection and box distributions, RTMPose's gated attention unit and SimCC maxima) and
//! `img.wgsl` (the frame resized for D-FINE, the people's crops for RTMPose). A frame goes up once;
//! two read-backs a frame (the people, then their keypoints).
//!
//! The convolutional parts are written once over [`crate::layers::Backend`]; the token parts as
//! [`Run`]'s operations, mirroring the CPU code (the reference). [`Run`] records each operation
//! into a command buffer, submitted when a result is read back. Every failure is an error the
//! caller answers by running the CPU passes: no adapter, a device error or loss (from then on the
//! CPU for the session), a tensor larger than the device takes, a read-back that does not come.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use wgpu::util::DeviceExt;

use crate::VisionError;
use crate::gpu;
use crate::layers::{Act, Attention, Backend, Conv, Feat, Linear, Mlp, Norm};

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
    /// `tok.wgsl` and `img.wgsl` (one layout: the parameters, six inputs, the output).
    tlayout: wgpu::BindGroupLayout,
    tok: HashMap<&'static str, wgpu::ComputePipeline>,
    dummy: wgpu::Buffer,
    pool: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
}

/// The entry points of `tok.wgsl`.
const TOK_ENTRIES: &[&str] = &[
    "linear",
    "layernorm",
    "scalenorm",
    "softmax",
    "attention",
    "add",
    "concat2",
    "gate",
    "mask_rows",
    "planes2rows",
    "rows2planes",
    "rowmax",
    "rank",
    "gather",
    "refs_init",
    "init_box",
    "to_box",
    "lqe_stats",
    "add_rowscalar",
    "deform",
    "gau_attn",
    "gau_mix",
    "scale_add",
    "argmax",
];

/// The entry points of `img.wgsl`.
const IMG_ENTRIES: &[&str] = &["resize_area", "crop"];

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
        // the token and picture passes: the parameters, six inputs, the output
        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        }];
        entries.extend((1..=6).map(|b| storage(b, true)));
        entries.push(storage(7, false));
        let tlayout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("vision-tok"), entries: &entries });
        let tpl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("vision-tok"),
            bind_group_layouts: &[Some(&tlayout)],
            immediate_size: 0,
        });
        let mut tok = HashMap::new();
        for (src, names) in [(include_str!("tok.wgsl"), TOK_ENTRIES), (include_str!("img.wgsl"), IMG_ENTRIES)] {
            let module =
                device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("filmcraft-vision-tok"), source: wgpu::ShaderSource::Wgsl(src.into()) });
            for name in names {
                let pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&tpl),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                });
                tok.insert(*name, pipe);
            }
        }
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
            tlayout,
            tok,
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
        Act::PRelu => (4, 0.0, 0.0),
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
}

impl Run<'_> {
    /// Submit the recorded passes and read tensors back (the GPU's recorded work is submitted
    /// and waited for here; recording may go on after). The face network's 7 × 7 maps come back
    /// this way.
    pub(crate) fn read(&self, xs: &[&GT]) -> Result<Vec<Feat>, VisionError> {
        let bufs: Vec<(&wgpu::Buffer, usize)> = xs.iter().map(|m| (&m.buf, m.n * m.c * m.h * m.w)).collect();
        let vals = self.read_raw(&bufs)?;
        xs.iter().zip(vals).map(|(m, v)| Feat::new(v, m.n, m.c, m.h, m.w)).collect()
    }
}

impl Conv {
    /// Put the weights on the GPU.
    pub(crate) fn upload(&mut self, nn: &Nn) {
        let (w, b) = self.weights();
        self.dev = Some(ConvW { w: nn.upload(&w), b: nn.upload(&b) });
    }
}

impl Linear {
    /// Put the weights on the GPU.
    pub(crate) fn upload(&mut self, nn: &Nn) -> Result<(), VisionError> {
        let (w, b) = self.weights()?;
        let b = b.map(|b| nn.upload(b));
        self.dev = Some(LinW { w: nn.upload(&w), b });
        Ok(())
    }
}

impl Norm {
    /// Put the gain and bias on the GPU.
    pub(crate) fn upload(&mut self, nn: &Nn) {
        self.dev = Some(NormW { g: nn.upload(&self.g), b: nn.upload(&self.b) });
    }
}

impl Mlp {
    pub(crate) fn upload(&mut self, nn: &Nn) -> Result<(), VisionError> {
        self.layers.iter_mut().try_for_each(|l| l.upload(nn))
    }
}

impl Attention {
    pub(crate) fn upload(&mut self, nn: &Nn) -> Result<(), VisionError> {
        for l in [&mut self.q, &mut self.k, &mut self.v, &mut self.o] {
            l.upload(nn)?;
        }
        Ok(())
    }
}

/// A linear layer's weights on the GPU: `[in, out]` and the bias.
pub(crate) struct LinW {
    w: wgpu::Buffer,
    b: Option<wgpu::Buffer>,
}

/// A layer norm's gain and bias on the GPU.
pub(crate) struct NormW {
    g: wgpu::Buffer,
    b: wgpu::Buffer,
}

/// A matrix on the GPU: `rows` × `cols`, row-major. Cloning shares the buffer (every operation
/// writes a new one).
#[derive(Clone)]
pub(crate) struct GM {
    buf: wgpu::Buffer,
    pub(crate) rows: usize,
    pub(crate) cols: usize,
}

/// A frame on the GPU: RGBA8, one u32 a pixel.
pub(crate) struct GImg {
    buf: wgpu::Buffer,
    pub(crate) w: usize,
    pub(crate) h: usize,
}

/// The activation after a linear layer (`act` in `tok.wgsl`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TAct {
    None = 0,
    Relu = 1,
    Gelu = 2,
    Silu = 3,
}

/// The parameters of a token or picture pass (`T` in `tok.wgsl` and `img.wgsl`).
#[derive(Clone, Copy, Default)]
struct TP {
    rows: usize,
    cols: usize,
    cin: usize,
    cout: usize,
    n: usize,
    heads: usize,
    hd: usize,
    k: usize,
    flags: u32,
    act: u32,
    a_off: usize,
    total: usize,
    levels: usize,
    f0: f32,
}

impl TP {
    fn bytes(&self) -> Result<Vec<u8>, VisionError> {
        let u = |v: usize| u32::try_from(v).map_err(err);
        let words: [u32; 20] = [
            u(self.rows)?,
            u(self.cols)?,
            u(self.cin)?,
            u(self.cout)?,
            u(self.n)?,
            u(self.heads)?,
            u(self.hd)?,
            u(self.k)?,
            self.flags,
            self.act,
            u(self.a_off)?,
            0,
            u(self.total)?,
            u(self.levels)?,
            0,
            0,
            self.f0.to_bits(),
            0,
            0,
            0,
        ];
        Ok(words.iter().flat_map(|w| w.to_le_bytes()).collect())
    }
}

/// Values a matrix of `rows` × `cols` holds, checked.
fn count(rows: usize, cols: usize) -> Result<usize, VisionError> {
    rows.checked_mul(cols).ok_or_else(|| err("matrix too large"))
}

impl Run<'_> {
    /// A pooled storage buffer for `floats` values.
    fn buffer(&self, floats: usize) -> Result<wgpu::Buffer, VisionError> {
        Ok(self.alloc(1, 1, 1, floats)?.buf)
    }

    /// A matrix whose values are not set: every one must be written.
    pub(crate) fn mat(&self, rows: usize, cols: usize) -> Result<GM, VisionError> {
        Ok(GM { buf: self.buffer(count(rows, cols)?)?, rows, cols })
    }

    /// The same values as rows of `cols` (a view: the same buffer).
    pub(crate) fn reshape(&self, m: &GM, cols: usize) -> Result<GM, VisionError> {
        let n = count(m.rows, m.cols)?;
        if cols == 0 || !n.is_multiple_of(cols) {
            return Err(err("not whole rows"));
        }
        Ok(GM { buf: m.buf.clone(), rows: n / cols, cols })
    }

    /// A matrix from the CPU.
    pub(crate) fn upload_mat(&self, v: &[f32], rows: usize, cols: usize) -> Result<GM, VisionError> {
        if v.len() != count(rows, cols)? {
            return Err(err("matrix size"));
        }
        let m = self.mat(rows, cols)?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        if !bytes.is_empty() {
            self.nn.ctx.queue().write_buffer(&m.buf, 0, &bytes);
        }
        Ok(m)
    }

    /// One token or picture pass.
    fn tpass(&self, entry: &str, p: &TP, ins: [Option<&wgpu::Buffer>; 6], dst: &wgpu::Buffer, groups: (usize, usize, usize)) -> Result<(), VisionError> {
        let pipeline = self.nn.tok.get(entry).ok_or_else(|| err(format!("no pass `{entry}`")))?;
        let max = self.nn.ctx.device().limits().max_compute_workgroups_per_dimension;
        let g = |v: usize| u32::try_from(v.max(1)).map_err(err).and_then(|v| if v > max { Err(err("a tensor larger than one dispatch")) } else { Ok(v) });
        let (gx, gy, gz) = (g(groups.0)?, g(groups.1)?, g(groups.2)?);
        let d = self.nn.ctx.device();
        let uniform = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vision-tok-params"),
            contents: &p.bytes()?,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let dummy = &self.nn.dummy;
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() }];
        for (i, b) in (1u32..).zip(ins.iter()) {
            entries.push(wgpu::BindGroupEntry { binding: i, resource: b.unwrap_or(dummy).as_entire_binding() });
        }
        entries.push(wgpu::BindGroupEntry { binding: 7, resource: dst.as_entire_binding() });
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("vision-tok"), layout: &self.nn.tlayout, entries: &entries });
        let mut enc = self.enc.borrow_mut();
        let enc = enc.as_mut().ok_or_else(|| err("the run has ended"))?;
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some(entry), timestamp_writes: None });
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(gx, gy, gz);
        Ok(())
    }

    /// `x · W + bias`, then `act`, then `res` added, then clamped to ±`clamp`.
    pub(crate) fn linear(&self, x: &GM, l: &Linear, act: TAct, res: Option<&GM>, clamp: Option<f32>) -> Result<GM, VisionError> {
        let dev = l.dev.as_ref().ok_or_else(|| err("a linear layer's weights are not on the GPU"))?;
        if x.cols != l.cin {
            return Err(err(format!("a linear layer takes rows of {}, got {}", l.cin, x.cols)));
        }
        if res.is_some_and(|r| (r.rows, r.cols) != (x.rows, l.cout)) {
            return Err(err("a residual of another shape"));
        }
        let y = self.mat(x.rows, l.cout)?;
        let flags = u32::from(dev.b.is_some()) | if res.is_some() { 2 } else { 0 } | if clamp.is_some() { 4 } else { 0 };
        let p = TP { rows: x.rows, cin: l.cin, cout: l.cout, flags, act: act as u32, f0: clamp.unwrap_or(0.0), ..TP::default() };
        self.tpass(
            "linear",
            &p,
            [Some(&x.buf), Some(&dev.w), dev.b.as_ref(), res.map(|r| &r.buf), None, None],
            &y.buf,
            (l.cout.div_ceil(16), x.rows.div_ceil(16), 1),
        )?;
        Ok(y)
    }

    /// An MLP: ReLU between its layers.
    pub(crate) fn mlp(&self, x: &GM, m: &Mlp) -> Result<GM, VisionError> {
        let last = m.layers.len().checked_sub(1).ok_or_else(|| err("an empty MLP"))?;
        let mut y: Option<GM> = None;
        for (i, l) in m.layers.iter().enumerate() {
            let act = if i < last { TAct::Relu } else { TAct::None };
            y = Some(self.linear(y.as_ref().unwrap_or(x), l, act, None, None)?);
        }
        y.ok_or_else(|| err("an empty MLP"))
    }

    pub(crate) fn layernorm(&self, x: &GM, n: &Norm) -> Result<GM, VisionError> {
        let dev = n.dev.as_ref().ok_or_else(|| err("a norm's weights are not on the GPU"))?;
        if x.cols != n.g.len() {
            return Err(err("a norm of another width"));
        }
        let y = self.mat(x.rows, x.cols)?;
        let p = TP { rows: x.rows, cols: x.cols, f0: n.eps, ..TP::default() };
        self.tpass("layernorm", &p, [Some(&x.buf), Some(&dev.g), Some(&dev.b), None, None, None], &y.buf, (x.rows.div_ceil(64), 1, 1))?;
        Ok(y)
    }

    /// RTMPose's scale norm with gain `g` over each row.
    pub(crate) fn scalenorm(&self, x: &GM, g: f32) -> Result<GM, VisionError> {
        let y = self.mat(x.rows, x.cols)?;
        let p = TP { rows: x.rows, cols: x.cols, f0: g, ..TP::default() };
        self.tpass("scalenorm", &p, [Some(&x.buf), None, None, None, None, None], &y.buf, (x.rows.div_ceil(64), 1, 1))?;
        Ok(y)
    }

    pub(crate) fn softmax(&self, x: &GM) -> Result<GM, VisionError> {
        let y = self.mat(x.rows, x.cols)?;
        let p = TP { rows: x.rows, cols: x.cols, ..TP::default() };
        self.tpass("softmax", &p, [Some(&x.buf), None, None, None, None, None], &y.buf, (x.rows.div_ceil(64), 1, 1))?;
        Ok(y)
    }

    fn same(a: &GM, b: &GM) -> Result<(), VisionError> {
        if (a.rows, a.cols) != (b.rows, b.cols) { Err(err("matrices of different shapes")) } else { Ok(()) }
    }

    pub(crate) fn add(&self, a: &GM, b: &GM) -> Result<GM, VisionError> {
        Self::same(a, b)?;
        let y = self.mat(a.rows, a.cols)?;
        let p = TP { rows: a.rows, cols: a.cols, ..TP::default() };
        self.tpass("add", &p, [Some(&a.buf), Some(&b.buf), None, None, None, None], &y.buf, (count(a.rows, a.cols)?.div_ceil(64), 1, 1))?;
        Ok(y)
    }

    /// Multi-head attention ([`Attention::forward`]): queries and keys from `qk`, values from
    /// `v`, then the output projection with `res` added.
    pub(crate) fn mha(&self, at: &Attention, qk: &GM, v: &GM, res: Option<&GM>) -> Result<GM, VisionError> {
        let d = at.q.cout;
        let heads = at.heads.max(1);
        if !d.is_multiple_of(heads) || (qk.rows, qk.cols) != (v.rows, v.cols) || d / heads > 64 {
            return Err(err("attention shapes"));
        }
        let hd = d / heads;
        let q = self.linear(qk, &at.q, TAct::None, None, None)?;
        let k = self.linear(qk, &at.k, TAct::None, None, None)?;
        let vv = self.linear(v, &at.v, TAct::None, None, None)?;
        let o = self.mat(qk.rows, d)?;
        let p = TP { rows: qk.rows, cols: d, heads, hd, f0: 1.0 / (hd as f32).sqrt(), ..TP::default() };
        self.tpass("attention", &p, [Some(&q.buf), Some(&k.buf), Some(&vv.buf), None, None, None], &o.buf, (qk.rows.div_ceil(64), heads, 1))?;
        self.linear(&o, &at.o, TAct::None, res, None)
    }

    /// `[a | b]` side by side.
    pub(crate) fn concat2(&self, a: &GM, b: &GM) -> Result<GM, VisionError> {
        if a.rows != b.rows {
            return Err(err("matrices of different heights"));
        }
        let y = self.mat(a.rows, a.cols + b.cols)?;
        let p = TP { rows: a.rows, cin: a.cols, cout: b.cols, ..TP::default() };
        self.tpass("concat2", &p, [Some(&a.buf), Some(&b.buf), None, None, None, None], &y.buf, ((a.cols + b.cols).div_ceil(64), a.rows, 1))?;
        Ok(y)
    }

    /// D-FINE's gate: `σ(g₁) · h1 + σ(g₂) · cross`.
    pub(crate) fn gate(&self, g: &GM, h1: &GM, cross: &GM) -> Result<GM, VisionError> {
        Self::same(h1, cross)?;
        if g.rows != h1.rows || g.cols != 2 * h1.cols {
            return Err(err("gate shapes"));
        }
        let y = self.mat(h1.rows, h1.cols)?;
        let p = TP { rows: h1.rows, cols: h1.cols, ..TP::default() };
        self.tpass("gate", &p, [Some(&g.buf), Some(&h1.buf), Some(&cross.buf), None, None, None], &y.buf, (h1.cols.div_ceil(64), h1.rows, 1))?;
        Ok(y)
    }

    /// Rows whose `keep` is 0 set to zeros.
    pub(crate) fn mask_rows(&self, x: &GM, keep: &GM) -> Result<GM, VisionError> {
        if keep.rows * keep.cols != x.rows {
            return Err(err("a mask of another length"));
        }
        let y = self.mat(x.rows, x.cols)?;
        let p = TP { rows: x.rows, cols: x.cols, ..TP::default() };
        self.tpass("mask_rows", &p, [Some(&x.buf), Some(&keep.buf), None, None, None, None], &y.buf, (x.cols.div_ceil(64), x.rows, 1))?;
        Ok(y)
    }

    /// Picture 0 of `x` as token rows `[h·w, c]`, written from row `at` of `dst`.
    pub(crate) fn planes_into_rows(&self, x: &GT, dst: &GM, at: usize) -> Result<(), VisionError> {
        let hw = count(x.h, x.w)?;
        if x.c != dst.cols || at.checked_add(hw).is_none_or(|end| end > dst.rows) {
            return Err(err("token rows out of range"));
        }
        let p = TP { rows: hw, cols: x.c, a_off: at, ..TP::default() };
        self.tpass("planes2rows", &p, [Some(&x.buf), None, None, None, None, None], &dst.buf, (hw.div_ceil(64), x.c, 1))
    }

    /// Token rows `from..from + h·w` of `x` as one picture's planes `[1, cols, h, w]`.
    pub(crate) fn rows_to_planes(&self, x: &GM, from: usize, h: usize, w: usize) -> Result<GT, VisionError> {
        let hw = count(h, w)?;
        if from.checked_add(hw).is_none_or(|end| end > x.rows) {
            return Err(err("token rows out of range"));
        }
        let y = self.alloc(1, x.cols, h, w)?;
        let p = TP { rows: hw, cols: x.cols, a_off: from, ..TP::default() };
        self.tpass("rows2planes", &p, [Some(&x.buf), None, None, None, None, None], &y.buf, (hw.div_ceil(64), x.cols, 1))?;
        Ok(y)
    }

    /// Each row's largest value, `[rows, 1]`.
    pub(crate) fn rowmax(&self, x: &GM) -> Result<GM, VisionError> {
        let y = self.mat(x.rows, 1)?;
        let p = TP { rows: x.rows, cols: x.cols, ..TP::default() };
        self.tpass("rowmax", &p, [Some(&x.buf), None, None, None, None, None], &y.buf, (x.rows.div_ceil(64), 1, 1))?;
        Ok(y)
    }

    /// The indices of the `k` largest values of `x` (all its values), largest first, ties in
    /// order (a stable sort's): `[min(k, len), 1]`.
    pub(crate) fn top(&self, x: &GM, k: usize) -> Result<GM, VisionError> {
        let len = count(x.rows, x.cols)?;
        if len >= 1 << 24 {
            return Err(err("too many values to rank"));
        }
        let n = k.min(len);
        let y = self.mat(n, 1)?;
        let p = TP { rows: len, k: n, ..TP::default() };
        self.tpass("rank", &p, [Some(&x.buf), None, None, None, None, None], &y.buf, (len.div_ceil(64), 1, 1))?;
        Ok(y)
    }

    /// Rows `idx` of `x`.
    pub(crate) fn gather(&self, x: &GM, idx: &GM) -> Result<GM, VisionError> {
        let n = idx.rows * idx.cols;
        let y = self.mat(n, x.cols)?;
        let p = TP { rows: n, cols: x.cols, ..TP::default() };
        self.tpass("gather", &p, [Some(&x.buf), Some(&idx.buf), None, None, None, None], &y.buf, (x.cols.div_ceil(64), n, 1))?;
        Ok(y)
    }

    /// D-FINE's first boxes: `σ(reg + anchors[idx])`.
    pub(crate) fn refs_init(&self, reg: &GM, anchors: &GM, idx: &GM) -> Result<GM, VisionError> {
        if reg.cols != 4 || anchors.cols != 4 || idx.rows * idx.cols != reg.rows {
            return Err(err("box shapes"));
        }
        let y = self.mat(reg.rows, 4)?;
        let p = TP { rows: reg.rows, ..TP::default() };
        self.tpass("refs_init", &p, [Some(&reg.buf), Some(&anchors.buf), Some(&idx.buf), None, None, None], &y.buf, ((reg.rows * 4).div_ceil(64), 1, 1))?;
        Ok(y)
    }

    /// `σ(pre + σ⁻¹(refs))`.
    pub(crate) fn init_box(&self, pre: &GM, refs: &GM) -> Result<GM, VisionError> {
        Self::same(pre, refs)?;
        let y = self.mat(pre.rows, 4)?;
        let p = TP { rows: pre.rows, ..TP::default() };
        self.tpass("init_box", &p, [Some(&pre.buf), Some(&refs.buf), None, None, None, None], &y.buf, ((pre.rows * 4).div_ceil(64), 1, 1))?;
        Ok(y)
    }

    /// FDR: the boxes from the first layer's boxes and the distance distributions (`bins` each,
    /// four a query), the bin positions `project` (uploaded) and `reg_scale`.
    pub(crate) fn to_box(&self, init: &GM, corners: &GM, project: &wgpu::Buffer, bins: usize, reg_scale: f32) -> Result<GM, VisionError> {
        if init.cols != 4 || corners.rows != init.rows || corners.cols != 4 * bins {
            return Err(err("distribution shapes"));
        }
        let y = self.mat(init.rows, 4)?;
        let p = TP { rows: init.rows, k: bins, f0: reg_scale, ..TP::default() };
        self.tpass("to_box", &p, [Some(&init.buf), Some(&corners.buf), Some(project), None, None, None], &y.buf, (init.rows.div_ceil(64), 1, 1))?;
        Ok(y)
    }

    /// The localisation-quality statistics: per query and side, the `top` largest probabilities
    /// and their mean (`[rows, 4 · (top + 1)]`).
    pub(crate) fn lqe_stats(&self, corners: &GM, bins: usize, top: usize) -> Result<GM, VisionError> {
        if corners.cols != 4 * bins || top == 0 || top > bins {
            return Err(err("distribution shapes"));
        }
        let y = self.mat(corners.rows, 4 * (top + 1))?;
        let p = TP { rows: corners.rows, k: bins, n: top, ..TP::default() };
        self.tpass("lqe_stats", &p, [Some(&corners.buf), None, None, None, None, None], &y.buf, ((corners.rows * 4).div_ceil(64), 1, 1))?;
        Ok(y)
    }

    /// Each row of `x` plus that row's `s`.
    pub(crate) fn add_rowscalar(&self, x: &GM, s: &GM) -> Result<GM, VisionError> {
        if s.rows * s.cols != x.rows {
            return Err(err("one value a row"));
        }
        let y = self.mat(x.rows, x.cols)?;
        let p = TP { rows: x.rows, cols: x.cols, ..TP::default() };
        self.tpass("add_rowscalar", &p, [Some(&x.buf), Some(&s.buf), None, None, None, None], &y.buf, (x.cols.div_ceil(64), x.rows, 1))?;
        Ok(y)
    }

    /// D-FINE's deformable sampling of `tokens` (levels in `levels`: first token, rows,
    /// columns, as f32) for each query's box `refs`: `offsets` `[q, heads·total·2]`, softmaxed
    /// `weights` `[q, heads·total]`, `points` (each point's scale, then its level).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn deform(
        &self,
        tokens: &GM,
        offsets: &GM,
        weights: &GM,
        refs: &GM,
        points: &wgpu::Buffer,
        levels: &GM,
        heads: usize,
        total: usize,
        offset_scale: f32,
    ) -> Result<GM, VisionError> {
        let (nq, d) = (refs.rows, tokens.cols);
        if heads == 0
            || !d.is_multiple_of(heads)
            || offsets.rows != nq
            || offsets.cols != heads * total * 2
            || weights.rows != nq
            || weights.cols != heads * total
            || refs.cols != 4
            || levels.cols != 3
        {
            return Err(err("sampling shapes"));
        }
        let y = self.mat(nq, d)?;
        let p = TP { rows: nq, cols: d, heads, hd: d / heads, total, levels: levels.rows, f0: offset_scale, ..TP::default() };
        self.tpass(
            "deform",
            &p,
            [Some(&tokens.buf), Some(&offsets.buf), Some(&weights.buf), Some(&refs.buf), Some(points), Some(&levels.buf)],
            &y.buf,
            (d.div_ceil(64), nq, 1),
        )?;
        Ok(y)
    }

    /// RTMPose's gated attention unit over `uv` (`[crops·k, 2e + s]`): `u ⊙ (A·v)`, `[crops·k, e]`.
    pub(crate) fn gau(&self, uv: &GM, gamma: &wgpu::Buffer, beta: &wgpu::Buffer, e: usize, s: usize, k: usize) -> Result<GM, VisionError> {
        if uv.cols != 2 * e + s || k == 0 || !uv.rows.is_multiple_of(k) {
            return Err(err("gated attention shapes"));
        }
        let w = self.mat(uv.rows, k)?;
        let p = TP { rows: uv.rows, cin: e, cout: s, k, ..TP::default() };
        self.tpass("gau_attn", &p, [Some(&uv.buf), Some(gamma), Some(beta), None, None, None], &w.buf, (k.div_ceil(64), uv.rows, 1))?;
        let y = self.mat(uv.rows, e)?;
        self.tpass("gau_mix", &p, [Some(&uv.buf), Some(&w.buf), None, None, None, None], &y.buf, (e.div_ceil(64), uv.rows, 1))?;
        Ok(y)
    }

    /// `scale[col] · a + b`.
    pub(crate) fn scale_add(&self, a: &GM, b: &GM, scale: &wgpu::Buffer) -> Result<GM, VisionError> {
        Self::same(a, b)?;
        let y = self.mat(a.rows, a.cols)?;
        let p = TP { rows: a.rows, cols: a.cols, ..TP::default() };
        self.tpass("scale_add", &p, [Some(&a.buf), Some(&b.buf), Some(scale), None, None, None], &y.buf, (a.cols.div_ceil(64), a.rows, 1))?;
        Ok(y)
    }

    /// Each row's first maximum, `[rows, 2]`: its index (as f32) and value.
    pub(crate) fn argmax(&self, x: &GM) -> Result<GM, VisionError> {
        let y = self.mat(x.rows, 2)?;
        let p = TP { rows: x.rows, cols: x.cols, ..TP::default() };
        self.tpass("argmax", &p, [Some(&x.buf), None, None, None, None, None], &y.buf, (x.rows.div_ceil(64), 1, 1))?;
        Ok(y)
    }

    /// A tensor's values as a matrix of `cols` columns (a view: the same buffer).
    pub(crate) fn as_rows(&self, t: &GT, cols: usize) -> Result<GM, VisionError> {
        let n = t.n.checked_mul(t.c).and_then(|x| x.checked_mul(t.h)).and_then(|x| x.checked_mul(t.w)).ok_or_else(|| err("tensor too large"))?;
        if cols == 0 || !n.is_multiple_of(cols) {
            return Err(err("not whole rows"));
        }
        Ok(GM { buf: t.buf.clone(), rows: n / cols, cols })
    }

    /// An RGBA frame (`w` × `h`, 4 bytes a pixel) on the GPU.
    pub(crate) fn upload_rgba(&self, rgba: &[u8], w: usize, h: usize) -> Result<GImg, VisionError> {
        if w == 0 || h == 0 || w > 1 << 15 || h > 1 << 15 {
            return Err(VisionError::Invalid(format!("a {w} × {h} picture cannot be analysed")));
        }
        let len = w.checked_mul(h).and_then(|n| n.checked_mul(4)).ok_or_else(|| err("picture too large"))?;
        let src = rgba.get(..len).ok_or_else(|| VisionError::Invalid(format!("{} bytes for a {w} × {h} RGBA picture", rgba.len())))?;
        let buf = self.buffer(w * h)?;
        self.nn.ctx.queue().write_buffer(&buf, 0, src);
        Ok(GImg { buf, w, h })
    }

    /// The frame area-averaged to `ow` × `oh`, RGB in 0..1, `[1, 3, oh, ow]`
    /// ([`crate::dfine::prepare_exact`]).
    pub(crate) fn resize_area(&self, img: &GImg, ow: usize, oh: usize) -> Result<GT, VisionError> {
        if ow == 0 || oh == 0 || ow > 8192 || oh > 8192 {
            return Err(VisionError::Invalid(format!("cannot resample to {ow} × {oh}")));
        }
        // the sums of a window must fit in 32 bits
        let window = (img.w / ow + 1).saturating_mul(img.h / oh + 1);
        if window > (u32::MAX / 255) as usize {
            return Err(err("too large a reduction for the GPU"));
        }
        let y = self.alloc(1, 3, oh, ow)?;
        let p = TP { rows: img.h, cin: img.w, cols: ow, cout: oh, ..TP::default() };
        self.tpass("resize_area", &p, [Some(&img.buf), None, None, None, None, None], &y.buf, (ow.div_ceil(64), oh, 1))?;
        Ok(y)
    }

    /// The pose network's crops of the frame (each `[cx, cy, k]`: centre and frame pixels per
    /// crop pixel), `[n, 3, 256, 192]` (`rtmpose`'s crops).
    pub(crate) fn crops(&self, img: &GImg, at: &[[f32; 3]]) -> Result<GT, VisionError> {
        let (cw, ch) = (crate::rtmpose::INPUT_W, crate::rtmpose::INPUT_H);
        if at.is_empty() || at.iter().flatten().any(|v| !v.is_finite()) {
            return Err(err("crop places"));
        }
        let params: Vec<f32> = at.iter().flatten().copied().collect();
        let pm = self.upload_mat(&params, at.len(), 3)?;
        let y = self.alloc(at.len(), 3, ch, cw)?;
        let p = TP { rows: img.h, cin: img.w, n: at.len(), ..TP::default() };
        self.tpass("crop", &p, [Some(&img.buf), Some(&pm.buf), None, None, None, None], &y.buf, ((cw * ch).div_ceil(64), at.len(), 1))?;
        Ok(y)
    }

    /// Submit the recorded passes and read matrices back (recording may go on after).
    pub(crate) fn read_mats(&self, xs: &[&GM]) -> Result<Vec<Vec<f32>>, VisionError> {
        let bufs: Vec<(&wgpu::Buffer, usize)> = xs.iter().map(|m| (&m.buf, m.rows * m.cols)).collect();
        self.read_raw(&bufs)
    }

    /// Submit the recorded passes and read `len` values of each buffer back; recording may go
    /// on after.
    fn read_raw(&self, xs: &[(&wgpu::Buffer, usize)]) -> Result<Vec<Vec<f32>>, VisionError> {
        let mut enc = self.enc.borrow_mut().replace(Self::encoder(self.nn)).ok_or_else(|| err("the run has ended"))?;
        let d = self.nn.ctx.device();
        let mut staging = Vec::with_capacity(xs.len());
        for (b, len) in xs {
            let size = (*len as u64).max(1).checked_mul(4).ok_or_else(|| err("tensor too large"))?;
            if size > b.size() {
                return Err(err("a read-back past a buffer's end"));
            }
            let s = d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vision-nn-read"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            enc.copy_buffer_to_buffer(b, 0, &s, 0, size);
            staging.push(s);
        }
        let ok = self.nn.ctx.submit_and_map(enc, &staging)?;
        if !ok {
            return Err(err("the device failed or the read-back did not come"));
        }
        let mut out = Vec::with_capacity(xs.len());
        for ((_, len), s) in xs.iter().zip(&staging) {
            let data = s.slice(..).get_mapped_range().map_err(err)?;
            let v: Vec<f32> = data.as_chunks::<4>().0.iter().take(*len).map(|b| f32::from_le_bytes(*b)).collect();
            drop(data);
            s.unmap();
            out.push(v);
        }
        Ok(out)
    }
}
