//! The GPU half of [`filmcraft_render::denoise`]: the per-pixel merge of a plane with its
//! motion-aligned neighbours (WGSL compute), tested against the CPU's
//! [`merge_plane`](filmcraft_render::denoise::merge_plane). [`install`] makes it the merger every
//! denoised source uses; its device is made the first time a frame is merged.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use filmcraft_render::denoise::{Motion, Plane, PlaneMerger};

pub struct GpuDenoise {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

/// Set once the merger's own device reported an error or was lost: merging stays on the CPU.
static FAILED: AtomicBool = AtomicBool::new(false);

fn mark_failed(why: &str) {
    if !FAILED.swap(true, Ordering::SeqCst) {
        log::error!("GPU noise reduction disabled for this session ({why}); it runs on the CPU");
    }
}

fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

impl GpuDenoise {
    /// The merge's pipeline on `device`, or `None` when the device cannot build it.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("filmcraft-denoise"),
            source: wgpu::ShaderSource::Wgsl(include_str!("denoise.wgsl").into()),
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("denoise"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                storage(2, true),
                storage(3, true),
                storage(4, false),
            ],
        });
        let pl =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some("denoise"), bind_group_layouts: &[Some(&bgl)], immediate_size: 0 });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("denoise"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        match crate::poll_now(scope.pop()) {
            Some(None) => Some(GpuDenoise { device: device.clone(), queue: queue.clone(), pipeline, bgl }),
            Some(Some(e)) => {
                log::warn!("GPU noise reduction unavailable on this device, it runs on the CPU: {e}");
                None
            }
            None => None,
        }
    }

    /// [`merge_plane`](filmcraft_render::denoise::merge_plane) on the GPU, or `None` for what it
    /// does not take (planes of other sizes or sample formats, too big for the device) or when
    /// the device fails.
    pub fn merge(&self, center: &Plane<'_>, nbs: &[(Plane<'_>, &Motion)], scale: [f32; 2], sigma: f32) -> Option<Vec<f32>> {
        use wgpu::util::DeviceExt;
        let (w, h) = center.size();
        let limits = self.device.limits();
        let n = w.checked_mul(h)?;
        let out_size = (n as u64).checked_mul(4)?;
        let layers = nbs.len().checked_add(1)?;
        if n == 0
            || nbs.is_empty()
            || w > limits.max_texture_dimension_2d as usize
            || h > limits.max_texture_dimension_2d as usize
            || layers > limits.max_texture_array_layers as usize
            || out_size > limits.max_storage_buffer_binding_size.min(limits.max_buffer_size)
            || !(sigma.is_finite() && scale.iter().all(|s| s.is_finite() && *s > 0.0))
        {
            return None;
        }
        // every plane the same size and sample format: the 8 or 16 bits uploaded as they are
        let (format, max, bytes_per) = match center {
            Plane::U8 { .. } => (wgpu::TextureFormat::R8Uint, 255.0, 1),
            Plane::U16 { max, .. } => (wgpu::TextureFormat::R16Uint, *max, 2),
        };
        let same = |p: &Plane<'_>| {
            p.size() == (w, h)
                && match (center, p) {
                    (Plane::U8 { .. }, Plane::U8 { .. }) => true,
                    (Plane::U16 { max: a, .. }, Plane::U16 { max: b, .. }) => a == b,
                    _ => false,
                }
        };
        if !nbs.iter().all(|(p, _)| same(p)) || max.is_nan() || max <= 0.0 {
            return None;
        }
        let d = &self.device;
        let scope = d.push_error_scope(wgpu::ErrorFilter::Validation);
        let tex = d.create_texture(&wgpu::TextureDescriptor {
            label: Some("denoise-planes"),
            size: wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: layers as u32 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let upload = |layer: usize, p: &Plane<'_>| -> Option<()> {
            let bytes: std::borrow::Cow<'_, [u8]> = match p {
                Plane::U8 { px, .. } => std::borrow::Cow::Borrowed(px.get(..n)?),
                Plane::U16 { px, .. } => std::borrow::Cow::Owned(px.get(..n)?.iter().flat_map(|v| v.to_le_bytes()).collect()),
            };
            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                    aspect: wgpu::TextureAspect::All,
                },
                &bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some((w * bytes_per) as u32), rows_per_image: Some(h as u32) },
                wgpu::Extent3d { width: w as u32, height: h as u32, depth_or_array_layers: 1 },
            );
            Some(())
        };
        upload(0, center)?;
        for (k, (p, _)) in nbs.iter().enumerate() {
            upload(k + 1, p)?;
        }
        let mut params = Vec::with_capacity(32);
        for v in [w as u32, h as u32, nbs.len() as u32, 0] {
            params.extend_from_slice(&v.to_le_bytes());
        }
        for v in [scale[0], scale[1], 1.0 / (sigma * sigma).max(1e-12), max] {
            params.extend_from_slice(&v.to_le_bytes());
        }
        let mut infos = Vec::new();
        let mut vecs: Vec<u8> = Vec::new();
        for (_, m) in nbs {
            let start = (vecs.len() / 8) as u32;
            // (a motion short of its blocks reads as still there, as on the CPU)
            let (bw, bh) = if m.vec.len() >= m.bw.saturating_mul(m.bh) { (m.bw, m.bh) } else { (0, 0) };
            for v in [bw as u32, bh as u32] {
                infos.extend_from_slice(&v.to_le_bytes());
            }
            infos.extend_from_slice(&(m.block as f32).to_le_bytes());
            infos.extend_from_slice(&start.to_le_bytes());
            for v in m.vec.iter().take(bw * bh) {
                vecs.extend_from_slice(&v[0].to_le_bytes());
                vecs.extend_from_slice(&v[1].to_le_bytes());
            }
        }
        if vecs.is_empty() {
            vecs.extend_from_slice(&[0; 8]);
        }
        let init = |label: &str, contents: &[u8], usage| d.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage });
        let ub = init("denoise-params", &params, wgpu::BufferUsages::UNIFORM);
        let ib = init("denoise-infos", &infos, wgpu::BufferUsages::STORAGE);
        let vb = init("denoise-vecs", &vecs, wgpu::BufferUsages::STORAGE);
        let ob = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("denoise-out"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let rb = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("denoise-read"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let view = tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("denoise"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ub.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: ib.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: vb.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: ob.as_entire_binding() },
            ],
        });
        let mut enc = d.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("denoise") });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("denoise"), timestamp_writes: None });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups((w as u32).div_ceil(8), (h as u32).div_ceil(8), 1);
        }
        enc.copy_buffer_to_buffer(&ob, 0, &rb, 0, out_size);
        self.queue.submit([enc.finish()]);
        if let Some(Some(e)) = crate::poll_now(scope.pop()) {
            log::warn!("GPU noise reduction failed on a frame, merging it on the CPU: {e}");
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
        let v: Vec<f32> = data.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        drop(data);
        rb.unmap();
        (v.len() == n && !FAILED.load(Ordering::SeqCst)).then_some(v)
    }
}

impl PlaneMerger for GpuDenoise {
    fn merge(&self, center: &Plane<'_>, nbs: &[(Plane<'_>, &Motion)], scale: [f32; 2], sigma: f32) -> Option<Vec<f32>> {
        if FAILED.load(Ordering::SeqCst) {
            return None;
        }
        GpuDenoise::merge(self, center, nbs, scale, sigma)
    }
}

/// The merger on a device of its own, made the first time a frame is merged (`None` kept: a
/// machine without a GPU does not look again).
struct Lazy;

fn shared() -> Option<&'static GpuDenoise> {
    static SHARED: OnceLock<Option<GpuDenoise>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            // (a build without a wgpu backend: `Instance::default()` panics there)
            if wgpu::Instance::enabled_backend_features().is_empty() {
                return None;
            }
            let instance = wgpu::Instance::default();
            let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
            let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
            device.on_uncaptured_error(Arc::new(|e: wgpu::Error| mark_failed(&format!("GPU error: {e}"))));
            device.set_device_lost_callback(|reason, msg| mark_failed(&format!("GPU device lost ({reason:?}): {msg}")));
            GpuDenoise::new(&device, &queue)
        })
        .as_ref()
}

impl PlaneMerger for Lazy {
    fn merge(&self, center: &Plane<'_>, nbs: &[(Plane<'_>, &Motion)], scale: [f32; 2], sigma: f32) -> Option<Vec<f32>> {
        PlaneMerger::merge(shared()?, center, nbs, scale, sigma)
    }
}

/// Make the GPU merge every denoised source's frames (`filmcraft_render::denoise::set_gpu`); the
/// device is made on first use, and without one the CPU merges.
pub fn install() {
    filmcraft_render::denoise::set_gpu(Arc::new(Lazy));
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_render::denoise::{merge_plane, motion};

    /// A noisy plane of a pattern moved `shift` pixels right.
    fn plane(w: usize, h: usize, shift: f32, seed: u64) -> Vec<u8> {
        let mut rng = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..w * h)
            .map(|i| {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                let n = (rng >> 40) as f32 / (1u64 << 24) as f32 - 0.5;
                let (x, y) = ((i % w) as f32 - shift, (i / w) as f32);
                ((0.5 + 0.3 * (x * 0.05).sin() * (y * 0.07).cos() + n * 0.1) * 255.0).clamp(0.0, 255.0) as u8
            })
            .collect()
    }

    #[test]
    fn the_gpu_merges_as_the_cpu_does() {
        let Some((device, queue)) = crate::blend_tests::device() else {
            eprintln!("no GPU adapter: skipped");
            return;
        };
        let Some(g) = GpuDenoise::new(&device, &queue) else {
            eprintln!("no denoise pipeline on this device: skipped");
            return;
        };
        let (w, h) = (300, 170);
        let c = plane(w, h, 0.0, 1);
        let nb: Vec<Vec<u8>> = [(-3.0, 2), (3.0, 3), (6.5, 4)].iter().map(|(s, seed)| plane(w, h, *s, *seed)).collect();
        let cp = Plane::U8 { px: &c, w, h };
        let np: Vec<Plane<'_>> = nb.iter().map(|p| Plane::U8 { px: p, w, h }).collect();
        let ms: Vec<Motion> = np.iter().map(|p| motion(&cp, p)).collect();
        let with: Vec<(Plane<'_>, &Motion)> = np.iter().copied().zip(&ms).collect();
        for (scale, sigma) in [([1.0, 1.0], 0.05), ([1.0, 1.0], 0.5), ([0.5, 0.5], 0.08)] {
            let cpu = merge_plane(&cp, &with, scale, sigma);
            let gpu = g.merge(&cp, &with, scale, sigma).unwrap();
            let worst = cpu.iter().zip(&gpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
            assert!(worst < 2e-4, "{scale:?} σ {sigma}: {worst}");
        }
        // 10-bit samples
        let c16: Vec<u16> = c.iter().map(|v| u16::from(*v) * 4).collect();
        let n16: Vec<Vec<u16>> = nb.iter().map(|p| p.iter().map(|v| u16::from(*v) * 4).collect()).collect();
        let cp16 = Plane::U16 { px: &c16, w, h, max: 1023.0 };
        let with16: Vec<(Plane<'_>, &Motion)> = n16.iter().map(|p| Plane::U16 { px: p, w, h, max: 1023.0 }).zip(&ms).collect();
        let cpu = merge_plane(&cp16, &with16, [1.0, 1.0], 0.06);
        let gpu = g.merge(&cp16, &with16, [1.0, 1.0], 0.06).unwrap();
        assert!(cpu.iter().zip(&gpu).all(|(a, b)| (a - b).abs() < 2e-4));
        // what it does not take goes to the CPU
        let small = Plane::U8 { px: &c[..100], w: 10, h: 10 };
        assert!(g.merge(&cp, &[(small, &ms[0])], [1.0, 1.0], 0.1).is_none(), "another size");
        assert!(g.merge(&cp, &[(with16[0].0, &ms[0])], [1.0, 1.0], 0.1).is_none(), "another format");
        assert!(g.merge(&cp, &[], [1.0, 1.0], 0.1).is_none());
        assert!(g.merge(&cp, &with, [1.0, 1.0], f32::NAN).is_none());
        let short = Motion { bw: 50, bh: 50, block: 64, vec: vec![] };
        assert!(g.merge(&cp, &[(np[0], &short)], [1.0, 1.0], 0.1).is_some(), "a motion short of its blocks reads as still");
    }
}
