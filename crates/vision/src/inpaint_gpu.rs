//! MI-GAN on the GPU (feature `model`): `migan.wgsl`'s two passes through wgpu. The network's
//! size is fixed (512 × 512), so every map, every pass and its bindings are made once when it
//! loads; a picture is one upload, one command buffer of 80-odd passes and one read-back. Every
//! failure is an error the inpainter answers on the CPU: no adapter, a device error or loss (from
//! then on the CPU for the session), a read-back that does not come.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use wgpu::util::DeviceExt;

use crate::VisionError;
use crate::inpaint::{Dw, Pw, SIDE, Weights};

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("inpainter GPU: {e}"))
}

/// Set once the device reported an error or was lost: the CPU for the rest of the session.
static FAILED: AtomicBool = AtomicBool::new(false);

struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    bgl: wgpu::BindGroupLayout,
    dw: wgpu::ComputePipeline,
    pw: wgpu::ComputePipeline,
}

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
                label: Some("filmcraft-migan"),
                source: wgpu::ShaderSource::Wgsl(include_str!("migan.wgsl").into()),
            });
            let e = |binding: u32, ty: wgpu::BufferBindingType| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            };
            let ro = wgpu::BufferBindingType::Storage { read_only: true };
            let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("migan"),
                entries: &[
                    e(0, wgpu::BufferBindingType::Uniform),
                    e(1, ro),
                    e(2, ro),
                    e(3, ro),
                    e(4, ro),
                    e(5, ro),
                    e(6, wgpu::BufferBindingType::Storage { read_only: false }),
                ],
            });
            let pl =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("migan"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
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
            let (dw, pw) = (pipe("dw"), pipe("pw"));
            (!FAILED.load(Ordering::SeqCst)).then_some(Device { device, queue, bgl, dw, pw })
        })
        .as_ref()
        .filter(|_| !FAILED.load(Ordering::SeqCst))
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One pass's numbers (`migan.wgsl`'s `P`).
#[derive(Clone, Copy, Default)]
struct P {
    cin: u32,
    cout: u32,
    h: u32,
    w: u32,
    oh: u32,
    ow: u32,
    k: u32,
    stride: u32,
    pad: u32,
    mode: u32,
    flags: u32,
    stuff: f32,
}

const F_BIAS: u32 = 1;
const F_NOISE: u32 = 2;
const F_ACT: u32 = 4;
const F_ADD: u32 = 8;

impl P {
    fn bytes(&self) -> Vec<u8> {
        let u = [self.cin, self.cout, self.h, self.w, self.oh, self.ow, self.k, self.stride, self.pad, self.mode, self.flags, 0];
        let mut v: Vec<u8> = u.iter().flat_map(|x| x.to_le_bytes()).collect();
        v.extend(self.stuff.to_le_bytes());
        v.extend([0u8; 12]);
        v
    }
}

/// A pass ready to run: its pipeline (depthwise or not), bindings and workgroups.
struct Pass {
    dw: bool,
    bg: wgpu::BindGroup,
    groups: (u32, u32, u32),
}

/// The network, its maps and passes on the GPU.
pub(crate) struct GpuNet {
    dev: &'static Device,
    input: wgpu::Buffer,
    out: wgpu::Buffer,
    read: wgpu::Buffer,
    passes: Vec<Pass>,
    /// The passes run one picture at a time (nothing in them runs on rayon).
    one: Mutex<()>,
    _keep: Vec<wgpu::Buffer>,
}

/// Builds the passes: the buffers they read and write, kept alive with them.
struct Plan<'a> {
    dev: &'static Device,
    keep: Vec<wgpu::Buffer>,
    passes: Vec<Pass>,
    dummy: &'a wgpu::Buffer,
}

impl Plan<'_> {
    fn upload(&mut self, v: &[f32]) -> usize {
        let b = self.dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("migan-w"),
            contents: &bytes(v),
            usage: wgpu::BufferUsages::STORAGE,
        });
        self.keep.push(b);
        self.keep.len() - 1
    }

    #[allow(clippy::too_many_arguments)]
    fn pass(
        &mut self,
        dw: bool,
        p: P,
        src: &wgpu::Buffer,
        wt: usize,
        bias: Option<usize>,
        noise: Option<usize>,
        add: Option<&wgpu::Buffer>,
        dst: &wgpu::Buffer,
    ) {
        let d = &self.dev.device;
        let u = d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("migan-p"), contents: &p.bytes(), usage: wgpu::BufferUsages::UNIFORM });
        let at = |i: Option<usize>| i.and_then(|i| self.keep.get(i)).unwrap_or(self.dummy);
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("migan"),
            layout: &self.dev.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: at(Some(wt)).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: at(bias).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: at(noise).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: add.unwrap_or(self.dummy).as_entire_binding() },
                wgpu::BindGroupEntry { binding: 6, resource: dst.as_entire_binding() },
            ],
        });
        let groups = if dw { (p.ow.div_ceil(16), p.oh.div_ceil(16), p.cout) } else { ((p.oh * p.ow).div_ceil(64), p.cout.div_ceil(64), 1) };
        self.keep.push(u);
        self.passes.push(Pass { dw, bg, groups });
    }

    /// A depthwise convolution of `src` (`r` × `r`) to `dst`: plain (`stride`), or up by 2.
    #[allow(clippy::too_many_arguments)]
    fn dw(
        &mut self,
        d: &Dw,
        src: &wgpu::Buffer,
        r: usize,
        stride: usize,
        up: Option<f32>,
        noise: Option<usize>,
        act: bool,
        add: Option<&wgpu::Buffer>,
        dst: &wgpu::Buffer,
    ) {
        let wt = self.upload(&d.w);
        let bias = d.b.as_ref().map(|b| self.upload(b));
        let o = match up {
            Some(_) => r * 2,
            None => r / stride,
        } as u32;
        let flags = if bias.is_some() { F_BIAS } else { 0 }
            | if noise.is_some() { F_NOISE } else { 0 }
            | if act { F_ACT } else { 0 }
            | if add.is_some() { F_ADD } else { 0 };
        let p = P {
            cout: d.c as u32,
            h: r as u32,
            w: r as u32,
            oh: o,
            ow: o,
            k: d.k as u32,
            stride: stride as u32,
            pad: 1,
            mode: u32::from(up.is_some()),
            flags,
            stuff: up.unwrap_or(1.0),
            ..Default::default()
        };
        self.pass(true, p, src, wt, bias, noise, add, dst);
    }

    /// A pointwise convolution of `src` (`r` × `r`) to `dst`.
    #[allow(clippy::too_many_arguments)]
    fn pw(&mut self, c: &Pw, src: &wgpu::Buffer, r: usize, noise: Option<usize>, act: bool, add: Option<&wgpu::Buffer>, dst: &wgpu::Buffer) {
        let wt = self.upload(&c.w);
        let bias = c.b.as_ref().map(|b| self.upload(b));
        let flags = if bias.is_some() { F_BIAS } else { 0 }
            | if noise.is_some() { F_NOISE } else { 0 }
            | if act { F_ACT } else { 0 }
            | if add.is_some() { F_ADD } else { 0 };
        let p = P { cin: c.cin as u32, cout: c.cout as u32, h: r as u32, w: r as u32, oh: r as u32, ow: r as u32, flags, ..Default::default() };
        self.pass(false, p, src, wt, bias, noise, add, dst);
    }
}

impl GpuNet {
    /// The network's passes made, or `None` without a usable GPU.
    pub(crate) fn new(w: &Weights) -> Option<Self> {
        let dev = shared()?;
        let d = &dev.device;
        let buf = |floats: usize, usage: wgpu::BufferUsages| {
            d.create_buffer(&wgpu::BufferDescriptor { label: Some("migan-map"), size: (floats * 4) as u64, usage, mapped_at_creation: false })
        };
        let st = wgpu::BufferUsages::STORAGE;
        let hw = SIDE * SIDE;
        // the widest map: the widest block's channels at its resolution
        let widest = w
            .enc
            .iter()
            .zip([512usize, 256, 128, 64, 32, 16, 8, 4])
            .map(|(e, r)| e.pw1.cout.max(e.dw1.c) * r * r)
            .max()?
            .max(w.syn.iter().map(|s| s.pw2.cout).max()? * hw);
        let input = buf(4 * hw, st | wgpu::BufferUsages::COPY_DST);
        let work: Vec<wgpu::Buffer> = (0..3).map(|_| buf(widest, st)).collect();
        let feats: Vec<wgpu::Buffer> = w.enc.iter().zip([512usize, 256, 128, 64, 32, 16, 8, 4]).map(|(e, r)| buf(e.pw1.cout * r * r, st)).collect();
        let imgs: Vec<wgpu::Buffer> = (0..3).map(|_| buf(3 * hw, st | wgpu::BufferUsages::COPY_SRC)).collect();
        let read = buf(3 * hw, wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
        let dummy = buf(4, st);
        let mut plan = Plan { dev, keep: Vec::new(), passes: Vec::new(), dummy: &dummy };
        // the work map other than `a` (and `b`)
        let other = |a: usize, b: usize| (0..3).find(|i| *i != a && *i != b).unwrap_or(0);
        let mut cur = usize::MAX;
        let mut r = SIDE;
        for (e, feat) in w.enc.iter().zip(&feats) {
            let src = |cur: usize| if cur == usize::MAX { &input } else { &work[cur] };
            if let Some(f) = &e.fromrgb {
                let n = other(cur, cur);
                plan.pw(f, src(cur), r, None, true, None, &work[n]);
                cur = n;
            }
            let n = other(cur, cur);
            plan.dw(&e.dw1, src(cur), r, 1, None, None, true, None, &work[n]);
            plan.pw(&e.pw1, &work[n], r, None, true, None, feat);
            let a = other(n, n);
            plan.dw(&e.dw2, feat, r, 1, None, None, true, None, &work[a]);
            match &e.down {
                Some(dn) => {
                    let b = other(a, a);
                    plan.dw(dn, &work[a], r, 2, None, None, false, None, &work[b]);
                    r /= 2;
                    let c = other(b, b);
                    plan.pw(&e.pw2, &work[b], r, None, true, None, &work[c]);
                    cur = c;
                }
                None => {
                    let b = other(a, a);
                    plan.pw(&e.pw2, &work[a], r, None, true, None, &work[b]);
                    cur = b;
                }
            }
        }
        let mut img: Option<usize> = None;
        for (s, feat) in w.syn.iter().zip(feats.iter().rev()) {
            let n = other(cur, cur);
            plan.dw(&s.dw1, &work[cur], r, 1, None, None, true, None, &work[n]);
            let m = other(n, n);
            match (&s.up, &s.noise1) {
                (Some(up), Some(nz)) => {
                    plan.pw(&s.pw1, &work[n], r, None, false, None, &work[m]);
                    let nz = plan.upload(nz);
                    let o = other(m, m);
                    plan.dw(up, &work[m], r, 1, Some(w.stuff), Some(nz), true, Some(feat), &work[o]);
                    r *= 2;
                    cur = o;
                }
                _ => {
                    plan.pw(&s.pw1, &work[n], r, None, true, Some(feat), &work[m]);
                    cur = m;
                }
            }
            let n = other(cur, cur);
            plan.dw(&s.dw2, &work[cur], r, 1, None, None, true, None, &work[n]);
            let m = other(n, n);
            let nz = s.noise2.as_ref().map(|v| plan.upload(v));
            plan.pw(&s.pw2, &work[n], r, nz, true, None, &work[m]);
            cur = m;
            // the picture: the last one up by 2, plus this block's
            let next = (0..3).find(|i| Some(*i) != img).unwrap_or(0);
            match (&s.up_rgb, img) {
                (Some(u), Some(prev)) => {
                    let tmp = (0..3).find(|i| *i != prev && *i != next).unwrap_or(0);
                    plan.dw(u, &imgs[prev], r / 2, 1, Some(w.stuff), None, false, None, &imgs[tmp]);
                    plan.pw(&s.torgb, &work[cur], r, None, false, Some(&imgs[tmp]), &imgs[next]);
                }
                _ => plan.pw(&s.torgb, &work[cur], r, None, false, None, &imgs[next]),
            }
            img = Some(next);
        }
        let Plan { keep, passes, .. } = plan;
        let mut keep = keep;
        keep.extend(work);
        keep.extend(feats);
        keep.push(dummy);
        let out_i = img?;
        let mut imgs = imgs;
        let out = imgs.swap_remove(out_i);
        keep.extend(imgs);
        (!FAILED.load(Ordering::SeqCst)).then_some(Self { dev, input, out, read, passes, one: Mutex::new(()), _keep: keep })
    }

    /// The network's picture (`[3][512][512]`) for its four input channels.
    pub(crate) fn run(&self, x: &[f32]) -> Result<Vec<f32>, VisionError> {
        let _one = self.one.lock().unwrap_or_else(PoisonError::into_inner);
        if FAILED.load(Ordering::SeqCst) {
            return Err(err("the device failed"));
        }
        let hw = SIDE * SIDE;
        if x.len() != 4 * hw {
            return Err(err("the input is not 4 × 512 × 512"));
        }
        let (d, q) = (&self.dev.device, &self.dev.queue);
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        q.write_buffer(&self.input, 0, &bytes(x));
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("migan") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("migan"), timestamp_writes: None });
            for p in &self.passes {
                pass.set_pipeline(if p.dw { &self.dev.dw } else { &self.dev.pw });
                pass.set_bind_group(0, &p.bg, &[]);
                pass.dispatch_workgroups(p.groups.0.max(1), p.groups.1.max(1), p.groups.2.max(1));
            }
        }
        let out_bytes = (3 * hw * 4) as u64;
        enc.copy_buffer_to_buffer(&self.out, 0, &self.read, 0, out_bytes);
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
        if FAILED.load(Ordering::SeqCst) || v.len() != 3 * hw {
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
