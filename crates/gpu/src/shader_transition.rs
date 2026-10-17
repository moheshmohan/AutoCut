//! The shader-transition stage: GL Transitions ported to WGSL
//! (`filmcraft_render::transitions::shader`) drawn on the GPU.
//!
//! Each transition is one full-frame fragment pass reading the two isolated clip images (the
//! transition's inputs, like Push's) and compositing the result "over" the canvas. The WGSL comes
//! from the render crate's validation registry, so only sources naga accepted reach the device;
//! pipelines are built on first use inside a validation error scope, and a shader the device
//! rejects anyway is remembered and rendered on the CPU from then on. The uniforms (eased
//! progress included) are computed by `shader::uniforms`, the same function the CPU reference
//! uses.

use std::collections::HashMap;

use filmcraft_project::EffectInstance;
use filmcraft_render::transitions::shader;
use wgpu::util::DeviceExt;

pub(crate) struct ShaderStage {
    bgl: wgpu::BindGroupLayout,
    layout: wgpu::PipelineLayout,
    /// Built pipelines by effect id; None for a shader that failed to build here.
    pipelines: HashMap<&'static str, Option<wgpu::RenderPipeline>>,
}

impl ShaderStage {
    pub fn new(device: &wgpu::Device) -> Self {
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shader-transition"),
            entries: &[
                super::tex_entry(0),
                super::tex_entry(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
            ],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shader-transition"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        Self { bgl, layout, pipelines: HashMap::new() }
    }

    /// Whether the transition `id` can be drawn here, building its pipeline on first use. False
    /// for other transitions, for a shader the registry rejected, and for one the device rejected.
    pub fn ready(&mut self, device: &wgpu::Device, id: &str) -> bool {
        let Some(def) = shader::find(id) else { return false };
        if let Some(p) = self.pipelines.get(def.id) {
            return p.is_some();
        }
        let built = match shader::registry().source(def.id) {
            Ok(src) => self.build(device, def.id, src),
            Err(e) => {
                log::warn!("{e}; rendering it on the CPU");
                None
            }
        };
        let ok = built.is_some();
        self.pipelines.insert(def.id, built);
        ok
    }

    fn build(&self, device: &wgpu::Device, id: &'static str, src: &str) -> Option<wgpu::RenderPipeline> {
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(id), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(id),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState { module: &module, entry_point: Some("vs"), compilation_options: Default::default(), buffers: &[] },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: super::ACCUM_FORMAT,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });
        match super::poll_now(scope.pop()) {
            Some(None) => Some(pipeline),
            Some(Some(e)) => {
                log::warn!("shader transition `{id}` failed to build on this device, rendering it on the CPU: {e}");
                None
            }
            None => None,
        }
    }

    /// Draw transition `effect` at timeline progress `progress` from `inputs` (outgoing,
    /// incoming) over `canvas`. False (nothing drawn) when it is not [`ready`](Self::ready).
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        canvas: &wgpu::TextureView,
        inputs: [&wgpu::TextureView; 2],
        size: (u32, u32),
        effect: &EffectInstance,
        progress: f32,
        scale: f32,
    ) -> bool {
        let Some(Some(pipeline)) = shader::find(&effect.effect).and_then(|d| self.pipelines.get(d.id)) else { return false };
        let Some(values) = shader::uniforms(effect, size.0, size.1, progress, scale) else { return false };
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("shader-transition"),
            contents: &bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shader-transition"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(inputs[0]) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(inputs[1]) },
                wgpu::BindGroupEntry { binding: 2, resource: uniform.as_entire_binding() },
            ],
        });
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("shader-transition") });
        {
            let mut pass = super::accum_pass(&mut encoder, canvas, wgpu::LoadOp::Load);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
        true
    }
}
