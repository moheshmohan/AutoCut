//! Real-ESRGAN's compact network on the GPU (feature `model`; used on macOS): `esrgan.wgsl`'s
//! passes through wgpu, one command buffer a tile, the weights uploaded once.
//!
//! candle's Metal convolution (an im2col buffer, then a matrix product) runs a 64-channel 3 × 3
//! layer at about 450 GFLOPS whatever the precision or tile, which made a 640 × 480 picture take
//! three seconds; the fused pass reads each pixel's neighbourhood in place. Every failure is an
//! error the upscaler answers on candle: no adapter, a device error or loss (from then on candle
//! for the session), a tile larger than the device's buffers, a read-back that does not come.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use wgpu::util::DeviceExt;

use crate::VisionError;

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("upscaler GPU: {e}"))
}

/// Set once the device reported an error or was lost: candle for the rest of the session.
static FAILED: AtomicBool = AtomicBool::new(false);

/// One convolution: its weights as `[cin · 9][cout]`, bias and PReLU slopes (`None`: no
/// activation).
pub(crate) struct LayerWeights {
    pub cin: usize,
    pub cout: usize,
    /// `[cout][cin][3][3]`, as the checkpoint holds them.
    pub weight: Vec<f32>,
    pub bias: Vec<f32>,
    pub slope: Option<Vec<f32>>,
}

struct Layer {
    cin: u32,
    cout: u32,
    act: u32,
    wt: wgpu::Buffer,
    bias: wgpu::Buffer,
    slope: wgpu::Buffer,
}

struct Device {
    device: wgpu::Device,
    queue: wgpu::Queue,
    conv: wgpu::ComputePipeline,
    conv_bgl: wgpu::BindGroupLayout,
    shuffle: wgpu::ComputePipeline,
    shuffle_bgl: wgpu::BindGroupLayout,
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
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
                label: Some("filmcraft-esrgan"),
                source: wgpu::ShaderSource::Wgsl(include_str!("esrgan.wgsl").into()),
            });
            let (ro, rw, un) =
                (wgpu::BufferBindingType::Storage { read_only: true }, wgpu::BufferBindingType::Storage { read_only: false }, wgpu::BufferBindingType::Uniform);
            let conv_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("esrgan-conv"),
                entries: &[entry(0, un), entry(1, ro), entry(2, ro), entry(3, ro), entry(4, ro), entry(5, rw)],
            });
            let shuffle_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("esrgan-shuffle"),
                entries: &[entry(0, un), entry(1, ro), entry(2, ro), entry(3, rw)],
            });
            let pipeline = |bgl: &wgpu::BindGroupLayout, name: &str| {
                let pl =
                    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(name), bind_group_layouts: &[Some(bgl)], immediate_size: 0 });
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(name),
                    layout: Some(&pl),
                    module: &module,
                    entry_point: Some(name),
                    compilation_options: Default::default(),
                    cache: None,
                })
            };
            let (conv, shuffle) = (pipeline(&conv_bgl, "conv3"), pipeline(&shuffle_bgl, "shuffle"));
            (!FAILED.load(Ordering::SeqCst)).then_some(Device { device, queue, conv, conv_bgl, shuffle, shuffle_bgl })
        })
        .as_ref()
        .filter(|_| !FAILED.load(Ordering::SeqCst))
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// The network's layers on the GPU.
pub(crate) struct GpuNet {
    dev: &'static Device,
    layers: Vec<Layer>,
    /// One tile at a time (each holds a few hundred MB of maps). Held only round the GPU's work,
    /// which runs nothing on rayon (a thread waiting there could take work that comes back here).
    one: std::sync::Mutex<()>,
}

impl GpuNet {
    /// The layers uploaded, or `None` without a usable GPU (or weights of the wrong sizes).
    pub(crate) fn new(weights: &[LayerWeights]) -> Option<Self> {
        let dev = shared()?;
        let mut layers = Vec::with_capacity(weights.len());
        for l in weights {
            let k = l.cin.checked_mul(9)?;
            if l.weight.len() != l.cout.checked_mul(k)? || l.bias.len() != l.cout || l.slope.as_ref().is_some_and(|s| s.len() != l.cout) {
                return None;
            }
            // `[cout][cin][3][3]` → `[cin · 9 + tap][cout]`
            let mut wt = vec![0f32; k * l.cout];
            for co in 0..l.cout {
                for kk in 0..k {
                    wt[kk * l.cout + co] = l.weight[co * k + kk];
                }
            }
            let up = |label: &str, v: &[f32]| {
                dev.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: &bytes(v), usage: wgpu::BufferUsages::STORAGE })
            };
            let slope = l.slope.clone().unwrap_or_else(|| vec![1.0; l.cout]);
            layers.push(Layer {
                cin: u32::try_from(l.cin).ok()?,
                cout: u32::try_from(l.cout).ok()?,
                act: u32::from(l.slope.is_some()),
                wt: up("esrgan-w", &wt),
                bias: up("esrgan-b", &l.bias),
                slope: up("esrgan-a", &slope),
            });
        }
        (!FAILED.load(Ordering::SeqCst)).then_some(Self { dev, layers, one: std::sync::Mutex::new(()) })
    }

    /// A planar RGB tile (0..1, `w` × `h`) through the network: `3 × 4h × 4w` planar RGB.
    pub(crate) fn run(&self, px: &[f32], w: usize, h: usize) -> Result<Vec<f32>, VisionError> {
        if FAILED.load(Ordering::SeqCst) {
            return Err(err("the device failed earlier"));
        }
        let d = &self.dev.device;
        let npx = w.checked_mul(h).ok_or_else(|| err("tile too large"))?;
        if npx == 0 || px.len() != 3 * npx {
            return Err(err(format!("{} values for a {w} × {h} tile", px.len())));
        }
        let widest = self.layers.iter().map(|l| l.cout as usize).max().unwrap_or(0).max(3);
        let limits = d.limits();
        let most = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let feat_bytes = (npx as u64).saturating_mul(widest as u64).saturating_mul(4);
        let out_n = npx.checked_mul(3 * 16).ok_or_else(|| err("tile too large"))?;
        let out_bytes = out_n as u64 * 4;
        if feat_bytes > most || out_bytes > most {
            return Err(err("the tile is larger than the device's buffers"));
        }
        let gx = u32::try_from(npx.div_ceil(64)).map_err(err)?;
        let (sx, sy) = (u32::try_from((w * 4).div_ceil(8)).map_err(err)?, u32::try_from((h * 4).div_ceil(8)).map_err(err)?);
        if gx > limits.max_compute_workgroups_per_dimension
            || sx > limits.max_compute_workgroups_per_dimension
            || sy > limits.max_compute_workgroups_per_dimension
        {
            return Err(err("the tile is larger than the device's dispatches"));
        }
        let _one = self.one.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let input =
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("esrgan-in"), contents: &bytes(px), usage: wgpu::BufferUsages::STORAGE });
        let feat = |label: &str| {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: feat_bytes.max(16),
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let (fa, fb) = (feat("esrgan-a"), feat("esrgan-b"));
        let out = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esrgan-out"),
            size: out_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rb = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("esrgan-read"),
            size: out_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (w32, h32) = (u32::try_from(w).map_err(err)?, u32::try_from(h).map_err(err)?);
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("esrgan") });
        let mut uniforms = Vec::new();
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("esrgan"), timestamp_writes: None });
            pass.set_pipeline(&self.dev.conv);
            for (i, l) in self.layers.iter().enumerate() {
                // input → a → b → a …
                let (from, to) = match i {
                    0 => (&input, &fa),
                    _ if i % 2 == 1 => (&fa, &fb),
                    _ => (&fb, &fa),
                };
                let u = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("esrgan-p"),
                    contents: &words(&[l.cin, l.cout, h32, w32, l.act, 0, 0, 0]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("esrgan-conv"),
                    layout: &self.dev.conv_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: from.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 2, resource: l.wt.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 3, resource: l.bias.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 4, resource: l.slope.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 5, resource: to.as_entire_binding() },
                    ],
                });
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(gx, l.cout.div_ceil(64), 1);
                uniforms.push(u);
            }
            let last = if self.layers.len() % 2 == 1 { &fa } else { &fb };
            let u = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("esrgan-s"),
                contents: &words(&[h32, w32, 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("esrgan-shuffle"),
                layout: &self.dev.shuffle_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: u.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: last.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 2, resource: input.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 3, resource: out.as_entire_binding() },
                ],
            });
            pass.set_pipeline(&self.dev.shuffle);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(sx, sy, 3);
            uniforms.push(u);
        }
        enc.copy_buffer_to_buffer(&out, 0, &rb, 0, out_bytes);
        self.dev.queue.submit([enc.finish()]);
        if let Some(Some(e)) = poll_now(scope.pop()) {
            return Err(err(e));
        }
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        d.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(30)) }).map_err(err)?;
        rx.recv_timeout(std::time::Duration::from_secs(30)).map_err(err)?.map_err(err)?;
        let data = slice.get_mapped_range().map_err(err)?;
        let v: Vec<f32> = data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        drop(data);
        rb.unmap();
        if FAILED.load(Ordering::SeqCst) || v.len() != out_n {
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
