//! The detector networks on the GPU (feature `model`; used on macOS): the layers of `net.rs` as
//! WGSL compute passes (`net.wgsl`) through wgpu, one command buffer a picture.
//!
//! candle's Metal backend runs the depthwise-separable units as a chain of generic tensor
//! operations (nine shifted multiply-adds a unit, broadcasts that are slow on Metal) and was five
//! times slower than the fused CPU passes; these kernels fuse each layer instead (the 1×1
//! convolution with its bias; the depthwise convolution with its bias, ReLU and pool).
//!
//! One shared device for the process ([`Ctx::shared`]), the weights uploaded once per detector,
//! feature maps in pooled storage buffers. Every failure is an error the detector answers by
//! running the CPU passes: no adapter, a device error or loss (from then on the CPU for the
//! session), a map larger than the device takes, a read-back that does not come.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use wgpu::util::DeviceExt;

use crate::VisionError;

fn err(e: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("GPU: {e}"))
}

/// Set once the shared device reported an error or was lost: the CPU for the rest of the session.
static FAILED: AtomicBool = AtomicBool::new(false);

/// Whether the shared device failed (the CPU for the rest of the session).
pub(crate) fn failed() -> bool {
    FAILED.load(Ordering::SeqCst)
}

/// Most bytes of idle feature-map buffers kept for the next picture.
const POOL_BYTES: u64 = 1 << 30;

/// The shared device, its pipelines and the buffer pool.
pub(crate) struct Ctx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    layout: wgpu::BindGroupLayout,
    stem: wgpu::ComputePipeline,
    pointwise: wgpu::ComputePipeline,
    depthwise: wgpu::ComputePipeline,
    maxpool: wgpu::ComputePipeline,
    upadd: wgpu::ComputePipeline,
    /// Bound where a pass reads nothing.
    dummy: wgpu::Buffer,
    pool: Mutex<HashMap<u64, Vec<wgpu::Buffer>>>,
}

impl Ctx {
    /// The process's GPU for the detectors, or `None`: not macOS (the other platforms keep the
    /// CPU), `FILMCRAFT_VISION_CPU` set, no adapter or no wgpu backend in this build, or the device
    /// failed earlier.
    pub(crate) fn shared() -> Option<Arc<Ctx>> {
        static SHARED: OnceLock<Option<Arc<Ctx>>> = OnceLock::new();
        if !cfg!(target_os = "macos") || std::env::var_os("FILMCRAFT_VISION_CPU").is_some() || FAILED.load(Ordering::SeqCst) {
            return None;
        }
        SHARED.get_or_init(|| Self::new().map(Arc::new)).clone().filter(|_| !FAILED.load(Ordering::SeqCst))
    }

    fn new() -> Option<Ctx> {
        // without a backend `Instance::default()` panics
        if wgpu::Instance::enabled_backend_features().is_empty() {
            return None;
        }
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(
            instance.request_adapter(&wgpu::RequestAdapterOptions { power_preference: wgpu::PowerPreference::HighPerformance, ..Default::default() }),
        )
        .ok()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
        // wgpu's default handler panics on a validation or out-of-memory error
        device.on_uncaptured_error(Arc::new(|_: wgpu::Error| FAILED.store(true, Ordering::SeqCst)));
        device.set_device_lost_callback(|_, _| FAILED.store(true, Ordering::SeqCst));
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filmcraft-vision"),
            source: wgpu::ShaderSource::Wgsl(include_str!("net.wgsl").into()),
        });
        let storage = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("vision"),
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
        let pl =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("vision"), bind_group_layouts: &[Some(&layout)], immediate_size: 0 });
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
        let (stem, pointwise, depthwise, maxpool, upadd) =
            (pipeline("stem"), pipeline("pointwise"), pipeline("depthwise"), pipeline("maxpool"), pipeline("upadd"));
        let dummy = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vision-dummy"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        if FAILED.load(Ordering::SeqCst) {
            return None;
        }
        Some(Ctx { device, queue, layout, stem, pointwise, depthwise, maxpool, upadd, dummy, pool: Mutex::new(HashMap::new()) })
    }

    /// Weights in a storage buffer.
    pub(crate) fn upload(&self, data: &[f32]) -> wgpu::Buffer {
        let bytes: Vec<u8> = if data.is_empty() { vec![0; 4] } else { data.iter().flat_map(|v| v.to_le_bytes()).collect() };
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vision-weights"),
            contents: &bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    }
}

/// A 3×3/2 stem's weights on the GPU.
pub(crate) struct StemW {
    pub w: wgpu::Buffer,
    pub b: wgpu::Buffer,
}

/// A depthwise-separable unit's weights on the GPU.
pub(crate) struct UnitW {
    pub pw: wgpu::Buffer,
    pub pw_b: wgpu::Buffer,
    pub dw: wgpu::Buffer,
    pub dw_b: wgpu::Buffer,
}

/// A feature map on the GPU: `c` planes of `h` × `w`.
pub(crate) struct GMap {
    buf: wgpu::Buffer,
    c: usize,
    h: usize,
    w: usize,
}

/// One picture's passes, recorded into one command buffer and submitted by [`Run::read`].
pub(crate) struct Run<'a> {
    ctx: &'a Ctx,
    enc: RefCell<Option<wgpu::CommandEncoder>>,
    /// Pooled buffers this run holds (returned when it ends).
    held: RefCell<Vec<(u64, wgpu::Buffer)>>,
}

impl Drop for Run<'_> {
    fn drop(&mut self) {
        let held = std::mem::take(&mut *self.held.borrow_mut());
        let mut pool = self.ctx.pool.lock().unwrap_or_else(PoisonError::into_inner);
        let mut total: u64 = pool.iter().map(|(k, v)| k.saturating_mul(v.len() as u64)).sum();
        for (size, b) in held {
            if total.saturating_add(size) <= POOL_BYTES {
                total = total.saturating_add(size);
                pool.entry(size).or_default().push(b);
            }
        }
    }
}

impl<'a> Run<'a> {
    pub(crate) fn new(ctx: &'a Ctx) -> Self {
        let enc = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("vision") });
        Self { ctx, enc: RefCell::new(Some(enc)), held: RefCell::new(Vec::new()) }
    }

    /// A feature map's buffer of `floats` values (pooled).
    fn alloc(&self, floats: usize) -> Result<wgpu::Buffer, VisionError> {
        let size = (floats.max(1) as u64).checked_mul(4).ok_or_else(|| err("map too large"))?;
        let limits = self.ctx.device.limits();
        if size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size {
            return Err(err(format!("a {size}-byte map is more than the device takes")));
        }
        let pooled = self.ctx.pool.lock().unwrap_or_else(PoisonError::into_inner).get_mut(&size).and_then(Vec::pop);
        let b = pooled.unwrap_or_else(|| {
            self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vision-map"),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        self.held.borrow_mut().push((size, b.clone()));
        Ok(b)
    }

    /// Record one pass: `params` (cin, cout, h, w, flags, oh, ow), invocations `n` × `channels`.
    #[allow(clippy::too_many_arguments)]
    fn pass(
        &self,
        pipeline: &wgpu::ComputePipeline,
        params: [u32; 7],
        src: &wgpu::Buffer,
        wt: &wgpu::Buffer,
        bias: &wgpu::Buffer,
        dst: &wgpu::Buffer,
        src2: Option<&wgpu::Buffer>,
        (n, channels): (usize, usize),
    ) -> Result<(), VisionError> {
        let groups_x = u32::try_from(n.div_ceil(64)).map_err(err)?;
        let groups_y = u32::try_from(channels).map_err(err)?;
        let max = self.ctx.device.limits().max_compute_workgroups_per_dimension;
        if groups_x > max || groups_y > max {
            return Err(err("a map larger than one dispatch"));
        }
        let mut bytes = Vec::with_capacity(32);
        for v in params.iter().chain(&[0]) {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let d = &self.ctx.device;
        let uniform =
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("vision-params"), contents: &bytes, usage: wgpu::BufferUsages::UNIFORM });
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vision"),
            layout: &self.ctx.layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: src.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wt.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: bias.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: dst.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: src2.unwrap_or(&self.ctx.dummy).as_entire_binding() },
            ],
        });
        let mut enc = self.enc.borrow_mut();
        let enc = enc.as_mut().ok_or_else(|| err("the run has ended"))?;
        let mut cp = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("vision"), timestamp_writes: None });
        cp.set_pipeline(pipeline);
        cp.set_bind_group(0, &bg, &[]);
        cp.dispatch_workgroups(groups_x, groups_y, 1);
        Ok(())
    }

    fn dims(v: [usize; 7]) -> Result<[u32; 7], VisionError> {
        let mut out = [0u32; 7];
        for (o, x) in out.iter_mut().zip(v) {
            *o = u32::try_from(x).map_err(err)?;
        }
        Ok(out)
    }

    /// A map from the CPU (tests feed layers with it).
    #[cfg(test)]
    pub(crate) fn upload_map(&self, v: &[f32], c: usize, h: usize, w: usize) -> Result<GMap, VisionError> {
        if v.len() != c * h * w {
            return Err(err("map size"));
        }
        let buf = self.alloc(v.len())?;
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.ctx.queue.write_buffer(&buf, 0, &bytes);
        Ok(GMap { buf, c, h, w })
    }

    /// The stem over a picture of 3 planes of `h` × `w` (`cout` channels out).
    pub(crate) fn stem(&self, s: &StemW, cout: usize, pic: &[f32], h: usize, w: usize) -> Result<GMap, VisionError> {
        if pic.len() != 3 * h * w || h == 0 || w == 0 {
            return Err(err("picture size"));
        }
        let input = self.alloc(pic.len())?;
        let bytes: Vec<u8> = pic.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.ctx.queue.write_buffer(&input, 0, &bytes);
        let (oh, ow) = ((h - 1) / 2 + 1, (w - 1) / 2 + 1);
        let out = self.alloc(cout * oh * ow)?;
        self.pass(&self.ctx.stem, Self::dims([3, cout, h, w, 0, oh, ow])?, &input, &s.w, &s.b, &out, None, (oh * ow, cout))?;
        Ok(GMap { buf: out, c: cout, h: oh, w: ow })
    }

    /// A depthwise-separable unit (`cin` → `cout`), optionally pooled.
    pub(crate) fn unit(&self, u: &UnitW, cin: usize, cout: usize, relu: bool, x: &GMap, pool: bool) -> Result<GMap, VisionError> {
        if x.c != cin {
            return Err(err(format!("a unit takes {cin} channels, got {}", x.c)));
        }
        let (h, w) = (x.h, x.w);
        let y = self.alloc(cout * h * w)?;
        self.pass(&self.ctx.pointwise, Self::dims([cin, cout, h, w, 0, h, w])?, &x.buf, &u.pw, &u.pw_b, &y, None, (h * w, cout.div_ceil(4)))?;
        let (oh, ow) = if pool { (h / 2, w / 2) } else { (h, w) };
        if oh == 0 || ow == 0 {
            return Err(err("a plane too small to pool"));
        }
        let out = self.alloc(cout * oh * ow)?;
        let flags = usize::from(relu) | (usize::from(pool) << 1);
        self.pass(&self.ctx.depthwise, Self::dims([cout, cout, h, w, flags, oh, ow])?, &y, &u.dw, &u.dw_b, &out, None, (oh * ow, cout))?;
        Ok(GMap { buf: out, c: cout, h: oh, w: ow })
    }

    pub(crate) fn max_pool(&self, x: &GMap) -> Result<GMap, VisionError> {
        let (oh, ow) = (x.h / 2, x.w / 2);
        if oh == 0 || ow == 0 {
            return Err(err("a plane too small to pool"));
        }
        let out = self.alloc(x.c * oh * ow)?;
        let d = &self.ctx.dummy;
        self.pass(&self.ctx.maxpool, Self::dims([x.c, x.c, x.h, x.w, 0, oh, ow])?, &x.buf, d, d, &out, None, (oh * ow, x.c))?;
        Ok(GMap { buf: out, c: x.c, h: oh, w: ow })
    }

    /// `a + nearest-neighbour 2× upsampling of b` (`b` half `a`'s size).
    pub(crate) fn add_upsampled(&self, a: &GMap, b: &GMap) -> Result<GMap, VisionError> {
        if a.c != b.c || b.h * 2 != a.h || b.w * 2 != a.w {
            return Err(err(format!("feature sizes {}×{} and {}×{} do not pair", a.w, a.h, b.w, b.h)));
        }
        let out = self.alloc(a.c * a.h * a.w)?;
        let d = &self.ctx.dummy;
        self.pass(&self.ctx.upadd, Self::dims([a.c, a.c, a.h, a.w, 0, a.h, a.w])?, &a.buf, d, d, &out, Some(&b.buf), (a.h * a.w, a.c))?;
        Ok(GMap { buf: out, c: a.c, h: a.h, w: a.w })
    }

    /// Submit the recorded passes and read `maps` back.
    pub(crate) fn read(&self, maps: Vec<GMap>) -> Result<Vec<super::net::Map>, VisionError> {
        let mut enc = self.enc.borrow_mut().take().ok_or_else(|| err("the run has ended"))?;
        let d = &self.ctx.device;
        let mut staging = Vec::with_capacity(maps.len());
        for m in &maps {
            let size = (m.c * m.h * m.w).max(1) as u64 * 4;
            let s = d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("vision-read"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            enc.copy_buffer_to_buffer(&m.buf, 0, &s, 0, size);
            staging.push(s);
        }
        self.ctx.queue.submit([enc.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        for (i, s) in staging.iter().enumerate() {
            let tx = tx.clone();
            s.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send((i, r.is_ok()));
            });
        }
        drop(tx);
        // The mapping callbacks run in whichever thread polls the device: with several pictures
        // looked at at once, another thread's poll may fire ours just after our own wait ends, so
        // they are waited for (polling again meanwhile), not only collected.
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        let waited = d.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(10)) });
        let mut ok = vec![false; staging.len()];
        let mut received = 0;
        while received < staging.len() {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            if left.is_zero() {
                break;
            }
            match rx.recv_timeout(left.min(std::time::Duration::from_millis(50))) {
                Ok((i, r)) => {
                    if let Some(o) = ok.get_mut(i) {
                        *o = r;
                    }
                    received += 1;
                }
                Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                    let _ = d.poll(wgpu::PollType::Poll);
                }
                Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
        if waited.is_err() || FAILED.load(Ordering::SeqCst) || ok.iter().any(|o| !o) {
            // the passes may still be running on buffers the pool would hand out again: the CPU
            // for the rest of the session
            FAILED.store(true, Ordering::SeqCst);
            return Err(err("the device failed or the read-back did not come"));
        }
        let mut out = Vec::with_capacity(maps.len());
        for (m, s) in maps.iter().zip(&staging) {
            let data = s.slice(..).get_mapped_range().map_err(err)?;
            let v: Vec<f32> = data.as_chunks::<4>().0.iter().take(m.c * m.h * m.w).map(|b| f32::from_le_bytes(*b)).collect();
            drop(data);
            s.unmap();
            out.push(super::net::Map { v, c: m.c, h: m.h, w: m.w });
        }
        Ok(out)
    }
}
