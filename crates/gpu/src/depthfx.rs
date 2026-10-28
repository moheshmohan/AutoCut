//! The GPU half of [`filmcraft_render::depth`]: the 3D photo and Depth Blur per pixel (WGSL
//! compute, `depthfx.wgsl`), tested against the CPU's [`parallax_cpu`] and [`blur_cpu`].
//! [`install`] makes it what the depth effects use; its device is made the first time one runs,
//! with the adapter's own limits (a 4K picture is larger than the default buffer size).
//!
//! [`parallax_cpu`]: filmcraft_render::depth::parallax_cpu
//! [`blur_cpu`]: filmcraft_render::depth::blur_cpu

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use filmcraft_render::Image;
use filmcraft_render::depth::{DepthField, DepthGpu, MAX_STEPS, MIN_STEPS, Parallax, Tap};

pub struct GpuDepthFx {
    device: wgpu::Device,
    queue: wgpu::Queue,
    parallax: wgpu::ComputePipeline,
    parallax_bgl: wgpu::BindGroupLayout,
    blur: wgpu::ComputePipeline,
    blur_bgl: wgpu::BindGroupLayout,
}

/// Set once the device reported an error or was lost: the depth effects stay on the CPU.
static FAILED: AtomicBool = AtomicBool::new(false);

fn mark_failed(why: &str) {
    if !FAILED.swap(true, Ordering::SeqCst) {
        log::error!("GPU depth effects disabled for this session ({why}); they run on the CPU");
    }
}

fn entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

fn layout(device: &wgpu::Device, label: &str, inputs: u32) -> wgpu::BindGroupLayout {
    let mut entries = vec![entry(0, wgpu::BufferBindingType::Uniform)];
    entries.extend((1..=inputs).map(|b| entry(b, wgpu::BufferBindingType::Storage { read_only: true })));
    entries.push(entry(inputs + 1, wgpu::BufferBindingType::Storage { read_only: false }));
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries: &entries })
}

fn bytes_f32(v: impl IntoIterator<Item = f32>) -> Vec<u8> {
    v.into_iter().flat_map(f32::to_le_bytes).collect()
}

impl GpuDepthFx {
    /// The pipelines on `device`, or `None` when the device cannot build them.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filmcraft-depthfx"),
            source: wgpu::ShaderSource::Wgsl(include_str!("depthfx.wgsl").into()),
        });
        let pipeline = |bgl: &wgpu::BindGroupLayout, entry: &str| {
            let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(entry), bind_group_layouts: &[Some(bgl)], immediate_size: 0 });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let parallax_bgl = layout(device, "depth-parallax", 2);
        let blur_bgl = layout(device, "depth-blur", 3);
        let parallax = pipeline(&parallax_bgl, "parallax");
        let blur = pipeline(&blur_bgl, "blur");
        match crate::poll_now(scope.pop()) {
            Some(None) => Some(Self { device: device.clone(), queue: queue.clone(), parallax, parallax_bgl, blur, blur_bgl }),
            Some(Some(e)) => {
                log::warn!("GPU depth effects unavailable on this device, they run on the CPU: {e}");
                None
            }
            None => None,
        }
    }

    /// Whether a buffer of `bytes` fits the device.
    fn fits(&self, bytes: u64) -> bool {
        let l = self.device.limits();
        bytes > 0 && bytes <= l.max_storage_buffer_binding_size.min(l.max_buffer_size)
    }

    /// Run `pipeline` over a `w` × `h` picture with `uniform` and `inputs`, reading back its
    /// `w · h` RGBA pixels.
    fn run(&self, pipeline: &wgpu::ComputePipeline, bgl: &wgpu::BindGroupLayout, uniform: &[u8], inputs: &[&[u8]], w: usize, h: usize) -> Option<Image> {
        use wgpu::util::DeviceExt;
        let n = w.checked_mul(h)?;
        let out_size = (n as u64).checked_mul(16)?;
        let limits = self.device.limits();
        let groups = |v: usize| u32::try_from(v.div_ceil(8)).ok().filter(|g| *g <= limits.max_compute_workgroups_per_dimension);
        let (gx, gy) = (groups(w)?, groups(h)?);
        if !self.fits(out_size) || inputs.iter().any(|b| !self.fits(b.len() as u64)) {
            return None;
        }
        let d = &self.device;
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let ubuf =
            d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("depthfx-params"), contents: uniform, usage: wgpu::BufferUsages::UNIFORM });
        let ins: Vec<wgpu::Buffer> = inputs
            .iter()
            .map(|b| d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("depthfx-in"), contents: b, usage: wgpu::BufferUsages::STORAGE }))
            .collect();
        let out = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depthfx-out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rb = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depthfx-read"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: ubuf.as_entire_binding() }];
        entries.extend(ins.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32 + 1, resource: b.as_entire_binding() }));
        entries.push(wgpu::BindGroupEntry { binding: ins.len() as u32 + 1, resource: out.as_entire_binding() });
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("depthfx"), layout: bgl, entries: &entries });
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("depthfx") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("depthfx"), timestamp_writes: None });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(gx, gy, 1);
        }
        enc.copy_buffer_to_buffer(&out, 0, &rb, 0, out_size);
        self.queue.submit([enc.finish()]);
        if let Some(Some(e)) = crate::poll_now(scope.pop()) {
            log::warn!("GPU depth effect refused, it runs on the CPU: {e}");
            return None;
        }
        let slice = rb.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        d.poll(wgpu::PollType::Wait { submission_index: None, timeout: Some(std::time::Duration::from_secs(8)) }).ok()?;
        rx.recv().ok()?.ok()?;
        let data = slice.get_mapped_range().ok()?;
        let px: Vec<f32> = data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        drop(data);
        rb.unmap();
        (px.len() == n * 4 && !FAILED.load(Ordering::SeqCst)).then_some(Image { w, h, px })
    }

    /// [`parallax_cpu`](filmcraft_render::depth::parallax_cpu) on the GPU.
    pub fn parallax(&self, img: &Image, held: &DepthField, p: &Parallax) -> Option<Image> {
        let (w, h) = (img.w, img.h);
        if w == 0 || h == 0 || img.px.len() != w * h * 4 || held.near.len() != held.w * held.h || held.w == 0 || held.h == 0 {
            return None;
        }
        let ok = |v: f32| v.is_finite();
        if !(p.travel.iter().all(|v| ok(*v)) && ok(p.push) && ok(p.pivot) && ok(p.zoom) && p.zoom > 0.0) {
            return None;
        }
        let u32s = |v: usize| u32::try_from(v).ok();
        let mut u = Vec::with_capacity(48);
        for v in [u32s(w)?, u32s(h)?, u32s(held.w)?, u32s(held.h)?] {
            u.extend(v.to_le_bytes());
        }
        u.extend(bytes_f32([p.travel[0], p.travel[1], p.push, p.pivot, p.zoom]));
        for v in [MIN_STEPS, MAX_STEPS, 0] {
            u.extend(v.to_le_bytes());
        }
        let pic = bytes_f32(img.px.iter().copied());
        let depth = bytes_f32(held.near.iter().copied());
        self.run(&self.parallax, &self.parallax_bgl, &u, &[&pic, &depth], w, h)
    }

    /// [`blur_cpu`](filmcraft_render::depth::blur_cpu) on the GPU.
    pub fn blur(&self, img: &Image, nr: &[[f32; 2]], taps: &[Tap]) -> Option<Image> {
        let (w, h) = (img.w, img.h);
        if w == 0 || h == 0 || img.px.len() != w * h * 4 || nr.len() != w * h {
            return None;
        }
        let mut u = Vec::with_capacity(16);
        for v in [u32::try_from(w).ok()?, u32::try_from(h).ok()?, u32::try_from(taps.len()).ok()?, 0] {
            u.extend(v.to_le_bytes());
        }
        let pic = bytes_f32(img.px.iter().copied());
        let nrb = bytes_f32(nr.iter().flatten().copied());
        // (an empty buffer is not allowed: one unused tap)
        let mut tb = bytes_f32(taps.iter().flat_map(|t| [t.dx as f32, t.dy as f32, t.dist, t.area]));
        if tb.is_empty() {
            tb = vec![0; 16];
        }
        self.run(&self.blur, &self.blur_bgl, &u, &[&pic, &nrb, &tb], w, h)
    }
}

impl DepthGpu for GpuDepthFx {
    fn parallax(&self, img: &Image, held: &DepthField, p: &Parallax) -> Option<Image> {
        if FAILED.load(Ordering::SeqCst) {
            return None;
        }
        GpuDepthFx::parallax(self, img, held, p)
    }
    fn blur(&self, img: &Image, nr: &[[f32; 2]], taps: &[Tap]) -> Option<Image> {
        if FAILED.load(Ordering::SeqCst) {
            return None;
        }
        GpuDepthFx::blur(self, img, nr, taps)
    }
}

/// The effects on a device of their own, made the first time one runs (`None` kept: a machine
/// without a GPU does not look again).
struct Lazy;

fn shared() -> Option<&'static GpuDepthFx> {
    static SHARED: OnceLock<Option<GpuDepthFx>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            // (a build without a wgpu backend: `Instance::default()` panics there)
            if wgpu::Instance::enabled_backend_features().is_empty() {
                return None;
            }
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
            let desc = wgpu::DeviceDescriptor { required_limits: adapter.limits(), ..Default::default() };
            let (device, queue) = pollster::block_on(adapter.request_device(&desc)).ok()?;
            device.on_uncaptured_error(Arc::new(|e: wgpu::Error| mark_failed(&format!("GPU error: {e}"))));
            device.set_device_lost_callback(|reason, msg| mark_failed(&format!("GPU device lost ({reason:?}): {msg}")));
            GpuDepthFx::new(&device, &queue)
        })
        .as_ref()
}

impl DepthGpu for Lazy {
    fn parallax(&self, img: &Image, held: &DepthField, p: &Parallax) -> Option<Image> {
        DepthGpu::parallax(shared()?, img, held, p)
    }
    fn blur(&self, img: &Image, nr: &[[f32; 2]], taps: &[Tap]) -> Option<Image> {
        DepthGpu::blur(shared()?, img, nr, taps)
    }
}

/// Make the GPU run the depth effects (`filmcraft_render::depth::set_gpu`); the device is made on
/// first use, and without one the CPU runs them.
pub fn install() {
    filmcraft_render::depth::set_gpu(Arc::new(Lazy));
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_render::depth::{BlurOp, blur_cpu, near_and_radius, parallax_cpu, taps};

    /// A picture of colour ramps and stripes, and a depth of a near disc over a sloping ground.
    fn scene(w: usize, h: usize) -> (Image, DepthField) {
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let c = [x as f32 / w as f32, ((x / 5 + y / 3) % 4) as f32 / 4.0, y as f32 / h as f32, 1.0];
                img.px[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&c);
            }
        }
        let (dw, dh) = (w / 2, h / 2);
        let near = (0..dw * dh)
            .map(|i| {
                let (x, y) = ((i % dw) as f32 / dw as f32, (i / dw) as f32 / dh as f32);
                if (x - 0.45).hypot(y - 0.55) < 0.2 { 0.9 } else { 0.6 * y }
            })
            .collect();
        (img, DepthField::new(dw, dh, near).unwrap())
    }

    fn gpu() -> Option<&'static GpuDepthFx> {
        let g = shared();
        if g.is_none() {
            eprintln!("no GPU; skipping");
        }
        g
    }

    /// The share of channels further than `tol` apart, and the largest difference.
    fn compare(a: &Image, b: &Image, tol: f32) -> (f32, f32) {
        assert_eq!((a.w, a.h), (b.w, b.h));
        let d: Vec<f32> = a.px.iter().zip(&b.px).map(|(x, y)| (x - y).abs()).collect();
        (d.iter().filter(|v| **v > tol).count() as f32 / d.len() as f32, d.iter().copied().fold(0.0, f32::max))
    }

    #[test]
    fn the_3d_photo_on_the_gpu_matches_the_cpu() {
        let Some(g) = gpu() else { return };
        let (img, depth) = scene(320, 180);
        let held = depth.held(2);
        for p in [
            Parallax { travel: [9.6, 0.0], push: 0.0, pivot: 0.5, zoom: 1.0 / 0.97 },
            Parallax { travel: [0.0, -6.0], push: 0.12, pivot: 0.0, zoom: 1.0 },
            Parallax { travel: [5.0, 3.0], push: -0.1, pivot: 0.3, zoom: 1.2 },
            Parallax { travel: [0.0, 0.0], push: 0.0, pivot: 0.0, zoom: 1.1 },
        ] {
            let cpu = parallax_cpu(&img, &held, &p);
            let t = std::time::Instant::now();
            let on = g.parallax(&img, &held, &p).expect("runs on the GPU");
            let (share, worst) = compare(&cpu, &on, 1e-3);
            eprintln!("{p:?}: {:.3} % of channels past 1e-3, at most {worst:.4} ({:.1} ms)", share * 100.0, t.elapsed().as_secs_f64() * 1e3);
            // (a step's crossing can fall either side of a sample by rounding: a few pixels)
            assert!(share < 0.005, "{share}");
        }
    }

    #[test]
    fn depth_blur_on_the_gpu_matches_the_cpu() {
        let Some(g) = gpu() else { return };
        let (img, depth) = scene(320, 180);
        for (focus, radius, near) in [(0.9, 6.0, true), (0.2, 14.0, true), (0.5, 10.0, false), (0.9, 0.0, true)] {
            let op = BlurOp { focus, range: 0.1, radius, near };
            let nr = near_and_radius(&depth, &op, img.w, img.h);
            let t = taps(radius);
            let cpu = blur_cpu(&img, &nr, &t);
            let on = g.blur(&img, &nr, &t).expect("runs on the GPU");
            let (share, worst) = compare(&cpu, &on, 1e-4);
            eprintln!("focus {focus}, radius {radius}: {share} past 1e-4, at most {worst:.6}");
            assert!(share < 1e-4 && worst < 1e-3, "{share} {worst}");
        }
        // what it does not take: the CPU
        assert!(g.blur(&img, &[[0.0; 2]; 3], &[]).is_none());
        assert!(g.parallax(&img, &depth, &Parallax { travel: [f32::NAN, 0.0], push: 0.0, pivot: 0.0, zoom: 1.0 }).is_none());
    }

    /// Both at 1080p and 4K on the GPU against the CPU (printed).
    #[test]
    fn speed() {
        let Some(g) = gpu() else { return };
        for (w, h) in [(1920, 1080), (3840, 2160)] {
            let (img, depth) = scene(w, h);
            let held = depth.held(2);
            let p = Parallax { travel: [w as f32 * 0.03, 0.0], push: 0.05, pivot: 0.5, zoom: 1.05 };
            let op = BlurOp { focus: 0.9, range: 0.1, radius: w as f32 * 0.015, near: true };
            let nr = near_and_radius(&depth, &op, w, h);
            let t = taps(op.radius);
            let _ = g.parallax(&img, &held, &p);
            let time = |f: &dyn Fn() -> Option<Image>| {
                let t0 = std::time::Instant::now();
                let ok = f().is_some();
                (t0.elapsed().as_secs_f64() * 1e3, ok)
            };
            let (gp, a) = time(&|| g.parallax(&img, &held, &p));
            let (gb, b) = time(&|| g.blur(&img, &nr, &t));
            let (cp, _) = time(&|| Some(parallax_cpu(&img, &held, &p)));
            let (cb, _) = time(&|| Some(blur_cpu(&img, &nr, &t)));
            eprintln!("{w} × {h}: 3D photo GPU {gp:.0} ms (CPU {cp:.0}), depth blur GPU {gb:.0} ms (CPU {cb:.0}, {} taps)", t.len());
            assert!(a && b);
        }
    }
}
