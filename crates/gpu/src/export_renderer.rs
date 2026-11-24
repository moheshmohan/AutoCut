//! Off-screen GPU export renderer: composite a frame with the same
//! [`filmcraft_render::plan::plan_frame`] + [`GpuCompositor`] path the monitors use, then read
//! the **linear float accumulator** back (not the sRGB display texture), so the result matches
//! `filmcraft_render::render_sequence`'s premultiplied linear-light RGBA and the export pipeline's
//! placement, overlays and video limiter see exactly what the CPU renderer would have produced.
//!
//! Renderers share one off-screen wgpu device (no window), created by the first and kept for the
//! process, so an export can run several (one per export worker; each has its own compositor and
//! textures) without opening a device each. With no adapter (headless
//! machine, unsupported GPU) [`ExportRenderer::new`] returns `None` and the export falls back to
//! the CPU renderer; a `FramePlan::Image` plan (HDR / wide gamut / anything the plan sends to the
//! CPU) is returned unchanged — it already is the CPU image.
//!
//! Transitions planned as GPU steps (Push and the GL Transitions,
//! `filmcraft_render::transitions::shader`) are drawn by the same stages the Program monitor
//! uses, so an exported frame is the monitor's picture; a transition this device cannot draw
//! (a shader it rejected) sends the frame to the CPU reference renderer.

use crate::{GpuCompositor, prepare};
use filmcraft_project::{ItemId, Project};
use filmcraft_render::plan::{FramePlan, plan_frame};
use filmcraft_render::{Image, RenderOptions, SourceProvider};
use filmcraft_time::Tick;

/// Where export rendering spends its time (process-wide totals, for the bench).
pub mod timing {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Duration;

    static SUBMIT_NS: AtomicU64 = AtomicU64::new(0);
    static MAP_WAIT_NS: AtomicU64 = AtomicU64::new(0);
    static CONVERT_NS: AtomicU64 = AtomicU64::new(0);
    static FRAMES: AtomicU64 = AtomicU64::new(0);

    /// Totals since the last [`reset`].
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct GpuTimings {
        /// Recording and submitting the frame's commands.
        pub submit: Duration,
        /// Waiting for the GPU and the staging buffer's mapping.
        pub map_wait: Duration,
        /// Half float → f32 conversion of the read-back pixels.
        pub convert: Duration,
        /// Frames composited by the GPU.
        pub frames: u64,
    }

    fn add(counter: &AtomicU64, d: Duration) {
        counter.fetch_add(u64::try_from(d.as_nanos()).unwrap_or(u64::MAX), Ordering::Relaxed);
    }

    pub(crate) fn add_submit(d: Duration) {
        add(&SUBMIT_NS, d);
    }

    pub(crate) fn add_map_wait(d: Duration) {
        add(&MAP_WAIT_NS, d);
    }

    pub(crate) fn add_convert(d: Duration) {
        add(&CONVERT_NS, d);
    }

    pub(crate) fn add_frame() {
        FRAMES.fetch_add(1, Ordering::Relaxed);
    }

    pub fn get() -> GpuTimings {
        let d = |c: &AtomicU64| Duration::from_nanos(c.load(Ordering::Relaxed));
        GpuTimings { submit: d(&SUBMIT_NS), map_wait: d(&MAP_WAIT_NS), convert: d(&CONVERT_NS), frames: FRAMES.load(Ordering::Relaxed) }
    }

    pub fn reset() {
        for c in [&SUBMIT_NS, &MAP_WAIT_NS, &CONVERT_NS, &FRAMES] {
            c.store(0, Ordering::Relaxed);
        }
    }
}

/// Set once the shared export device reported an uncaptured error (validation, out of memory)
/// or was lost: every export renders on the CPU from then on. The device is shared by every
/// renderer and lives in a `OnceLock`, so it is not recreated in this process.
static DEVICE_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Retire the shared export device, logging the first reason only.
fn mark_device_failed(why: &str) {
    if !DEVICE_FAILED.swap(true, std::sync::atomic::Ordering::SeqCst) {
        log::error!("GPU export disabled for this session ({why}); exports render on the CPU");
    }
}

fn device_failed() -> bool {
    DEVICE_FAILED.load(std::sync::atomic::Ordering::SeqCst)
}

pub struct ExportRenderer {
    _instance: wgpu::Instance,
    #[expect(dead_code, reason = "kept for future frame-accuracy flows (readback fences)")]
    device: wgpu::Device,
    #[expect(dead_code, reason = "kept for future frame-accuracy flows (readback fences)")]
    queue: wgpu::Queue,
    compositor: GpuCompositor,
}

impl ExportRenderer {
    /// The off-screen device, or `None` when this host has no usable adapter or this build no
    /// wgpu backend.
    pub fn new() -> Option<Self> {
        // `None` is kept too: a machine without an adapter does not search again on every export
        static SHARED: std::sync::OnceLock<Option<(wgpu::Instance, wgpu::Device, wgpu::Queue)>> = std::sync::OnceLock::new();
        let (instance, device, queue) = SHARED
            .get_or_init(|| {
                // A build without a wgpu backend for this platform (`cargo build -p filmcraft-cli`
                // alone: this crate asks for none) has no GPU to offer, and `Instance::default()`
                // panics there.
                if wgpu::Instance::enabled_backend_features().is_empty() {
                    return None;
                }
                let instance = wgpu::Instance::default();
                let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
                let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
                // wgpu's default handler panics on a validation or out-of-memory error; log it and
                // send the remaining frames to the CPU instead.
                device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| mark_device_failed(&format!("GPU error: {e}"))));
                device.set_device_lost_callback(|reason, msg| mark_device_failed(&format!("GPU device lost ({reason:?}): {msg}")));
                Some((instance, device, queue))
            })
            .as_ref()?
            .clone();
        if device_failed() {
            return None;
        }
        let compositor = GpuCompositor::new(&device, &queue);
        Some(Self { _instance: instance, device, queue, compositor })
    }

    /// One frame as `render_sequence` would return it (premultiplied linear-light RGBA at the
    /// render size), or `None` when this frame must go to the CPU renderer.
    pub fn render(&mut self, project: &Project, seq: ItemId, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider) -> Option<Image> {
        let plan = plan_frame(project, seq, t, opts, sources);
        match &plan {
            // Already the CPU image: return it unchanged instead of rendering it twice.
            FramePlan::Image(img) => Some(img.clone()),
            // Layers, and transitions drawn by the same stages as on the monitors (Push, the
            // GL Transitions): an export matches the Program monitor. A transition this device
            // cannot draw makes `render_export_prepared` return None: the CPU renders the frame.
            FramePlan::Layers { .. } | FramePlan::Composite { .. } => {
                if device_failed() {
                    return None;
                }
                let prep = prepare(&plan);
                let (w, h, px) = self.compositor.render_export_prepared(&plan, Some(&prep))?;
                // an error reported while this frame was drawn leaves its pixels untrusted
                if device_failed() {
                    return None;
                }
                timing::add_frame();
                Some(Image { w: w as usize, h: h as usize, px })
            }
        }
    }

    /// One frame as `render(..)` followed by `Image::over_black_rgba8` would make it (opaque
    /// 8-bit sRGB RGBA at the render size), for an export that needs nothing else done to the
    /// picture: composited and encoded on the GPU, without the float frame. `None` when this
    /// frame must go to the CPU renderer.
    pub fn render_rgba8(
        &mut self,
        project: &Project,
        seq: ItemId,
        t: Tick,
        opts: RenderOptions,
        sources: &dyn SourceProvider,
    ) -> Option<(usize, usize, Vec<u8>)> {
        let plan = plan_frame(project, seq, t, opts, sources);
        match &plan {
            // Already the CPU image: encode it here.
            FramePlan::Image(img) => Some((img.w, img.h, img.over_black_rgba8())),
            FramePlan::Layers { .. } | FramePlan::Composite { .. } => {
                if device_failed() {
                    return None;
                }
                let prep = prepare(&plan);
                let (w, h, px) = self.compositor.render_export_rgba8_prepared(&plan, Some(&prep))?;
                // an error reported while this frame was drawn leaves its pixels untrusted
                if device_failed() {
                    return None;
                }
                timing::add_frame();
                Some((w as usize, h as usize, px))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{f16_to_f32, f32_to_f16};
    use filmcraft_export::FrameRenderer;
    use filmcraft_media::generators::GeneratorSource;
    use filmcraft_media::{Generator, MediaSource};
    use filmcraft_project::{ItemKind, Label, MediaClip, MediaRef, Project, SequenceSettings, TrackKind};
    use filmcraft_render::SourceMap;
    use filmcraft_time::{TICKS_PER_SECOND, Tick, TimeRange};
    use std::sync::Arc;

    const EPS: f32 = 1e-4;

    /// A solid colour generator at 320×180 plus a tone generator (mirrors the export tests'
    /// fixture-free project helper).
    fn project(transform: bool) -> (Arc<Project>, ItemId, SourceMap) {
        let mut p = Project::new("x");
        let g = GeneratorSource::new(
            Generator::ColorMatte { color: [0.2, 0.5, 0.9, 1.0] },
            320,
            180,
            filmcraft_time::FrameRate::FPS_24,
            Tick(2 * TICKS_PER_SECOND),
        );
        let tone = GeneratorSource::new(Generator::Tone { hz: 440.0, db: -6.0 }, 320, 180, filmcraft_time::FrameRate::FPS_24, Tick(2 * TICKS_PER_SECOND));
        let add = |p: &mut Project, g: &GeneratorSource| {
            let info = g.info().clone();
            p.add_item(
                &info.name.clone(),
                Label::Iris,
                ItemKind::Media(MediaClip {
                    media: MediaRef::Generator(g.generator.clone()),
                    info,
                    interpret: Default::default(),
                    mark_in: None,
                    mark_out: None,
                    markers: vec![],
                    offline: false,
                    proxy: None,
                    identity: None,
                }),
                None,
            )
        };
        let red = add(&mut p, &g);
        let t = add(&mut p, &tone);
        let seq =
            p.new_sequence("s", SequenceSettings { width: 320, height: 180, frame_rate: filmcraft_time::FrameRate::FPS_24, ..Default::default() }, 1, 1, None);
        let r = filmcraft_time::FrameRate::FPS_24;
        let _v = p.make_track_item(red, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, r.tick_of(24)), r).unwrap();
        let a = p.make_track_item(t, TrackKind::Audio, Tick::ZERO, TimeRange::new(Tick::ZERO, r.tick_of(24)), r).unwrap();
        if transform {
            // Motion + opacity as fixed effects (the way the golden transform scene does it).
            if let Some(clip) = p.sequence_mut(seq).unwrap().video_tracks[0].items.last_mut() {
                let m = clip.effect_mut("motion").unwrap();
                if let Some(pm) = m.param_mut("scale") {
                    pm.value = filmcraft_project::ParamValue::Float(45.0);
                }
                if let Some(pm) = m.param_mut("rotation") {
                    pm.value = filmcraft_project::ParamValue::Float(20.0);
                }
                if let Some(pm) = m.param_mut("position") {
                    pm.value = filmcraft_project::ParamValue::Vec2(filmcraft_geom::Vec2::new(215.0, 70.0));
                }
                if let Some(o) = clip.effect_mut("opacity")
                    && let Some(pm) = o.param_mut("opacity")
                {
                    pm.value = filmcraft_project::ParamValue::Float(70.0);
                }
            }
        }
        p.sequence_mut(seq).unwrap().audio_tracks[0].items.push(a);
        let mut m = SourceMap::default();
        m.0.insert(red, Arc::new(g));
        m.0.insert(t, Arc::new(tone));
        (Arc::new(p), seq, m)
    }

    /// GPU parity: p99 of the per-pixel max RGB difference (over black, 8-bit) ≤ 6 and mean < 1.5
    /// — the same criterion the golden GPU-vs-CPU parity tests use.
    fn assert_parity(cpu: &Image, gpu: &Image, what: &str) {
        assert_eq!((cpu.w, cpu.h), (gpu.w, gpu.h), "{what}: size mismatch");
        let to8 = |img: &Image| -> Vec<u8> { img.over_black_rgba8() };
        let (a, b) = (to8(cpu), to8(gpu));
        let mut d: Vec<u32> = a
            .as_chunks::<4>()
            .0
            .iter()
            .zip(b.as_chunks::<4>().0)
            .map(|(x, y)| (0..3).map(|k| (x[k] as u32).abs_diff(y[k] as u32)).max().unwrap_or(0))
            .collect();
        d.sort_unstable();
        let p99 = d[d.len() * 99 / 100];
        let mean = d.iter().sum::<u32>() as f64 / d.len() as f64;
        eprintln!("{what}: GPU vs CPU over-black p99 {p99}, mean {mean:.3}");
        assert!(p99 <= 6 && mean < 1.5, "{what}: GPU parity p99 {p99}, mean {mean:.3}");
    }

    #[test]
    fn solid_frame_matches_cpu() {
        let Some(mut r) = ExportRenderer::new() else {
            eprintln!("SKIPPED (gpu export parity): no adapter");
            return;
        };
        let (p, seq, m) = project(false);
        let t = Tick::ZERO;
        let cpu = filmcraft_render::render_sequence(&p, seq, t, RenderOptions::default(), &m);
        let gpu = r.render(&p, seq, t, RenderOptions::default(), &m).expect("GPU rendered the frame");
        // The linear premultiplied accumulator should be very close to the CPU float image.
        assert_eq!((cpu.w, cpu.h), (gpu.w, gpu.h));
        let max: f32 = cpu.px.iter().zip(&gpu.px).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        eprintln!("solid: max |GPU-CPU| linear {max:.6}");
        assert!(max < 0.02, "solid frame differs by {max}");
        assert_parity(&cpu, &gpu, "solid");
    }

    #[test]
    fn transformed_frame_matches_cpu() {
        let Some(mut r) = ExportRenderer::new() else {
            eprintln!("SKIPPED (gpu export parity): no adapter");
            return;
        };
        let (p, seq, m) = project(true);
        let t = filmcraft_time::FrameRate::FPS_24.tick_of(10);
        let cpu = filmcraft_render::render_sequence(&p, seq, t, RenderOptions::default(), &m);
        let gpu = r.render(&p, seq, t, RenderOptions::default(), &m).expect("GPU rendered the frame");
        assert_parity(&cpu, &gpu, "transformed");
    }

    /// Two demo clips (160×90) cut at frame 24 under transition `effect` (frames 18–30), on V2
    /// over a colour matte on V1; the outgoing clip at 70 % opacity, so the transition's result is
    /// composited over the matte. `nested`: the cut is inside a sequence nested (plainly) in the
    /// exported one.
    fn transition_project(effect: &str, nested: bool) -> (Arc<Project>, ItemId, SourceMap) {
        use filmcraft_media::DemoScene;
        let rate = filmcraft_time::FrameRate::FPS_24;
        let (w, h) = (160, 90);
        let mut p = Project::new("x");
        let mut m = SourceMap::default();
        let mut add = |p: &mut Project, g: Generator| {
            let g = GeneratorSource::new(g, w, h, rate, Tick(4 * TICKS_PER_SECOND));
            let info = g.info().clone();
            let id = p.add_item(
                &info.name.clone(),
                Label::Iris,
                ItemKind::Media(MediaClip {
                    media: MediaRef::Generator(g.generator.clone()),
                    info,
                    interpret: Default::default(),
                    mark_in: None,
                    mark_out: None,
                    markers: vec![],
                    offline: false,
                    proxy: None,
                    identity: None,
                }),
                None,
            );
            m.0.insert(id, Arc::new(g) as filmcraft_media::SharedSource);
            id
        };
        let ocean = add(&mut p, Generator::Demo(DemoScene::OceanSunset));
        let aurora = add(&mut p, Generator::Demo(DemoScene::Aurora));
        let matte = add(&mut p, Generator::ColorMatte { color: [0.1, 0.6, 0.2, 1.0] });
        let settings = SequenceSettings { width: w, height: h, frame_rate: rate, ..Default::default() };
        let seq = p.new_sequence("s", settings.clone(), 2, 1, None);
        let cut = if nested { p.new_sequence("inner", settings, 2, 1, None) } else { seq };
        let clip = |p: &mut Project, item, start: i64| {
            p.make_track_item(item, TrackKind::Video, rate.tick_of(start), TimeRange::new(Tick::ZERO, rate.tick_of(24)), rate).unwrap()
        };
        let mut a = clip(&mut p, ocean, 0);
        a.effect_mut("opacity").unwrap().param_mut("opacity").unwrap().value = filmcraft_project::ParamValue::Float(70.0);
        let b = clip(&mut p, aurora, 24);
        let (ia, ib) = (a.id, b.id);
        let track = if nested { 0 } else { 1 };
        let s = p.sequence_mut(cut).unwrap();
        s.video_tracks[track].items.extend([a, b]);
        s.video_tracks[track].transitions.push(filmcraft_project::Transition {
            id: filmcraft_project::TransitionId(1),
            effect: filmcraft_project::find_effect(effect).unwrap().instance(),
            start: rate.tick_of(18),
            duration: rate.tick_of(12),
            from: Some(ia),
            to: Some(ib),
            align: Default::default(),
            reverse: false,
            sound: None,
        });
        let under = p.make_track_item(matte, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, rate.tick_of(48)), rate).unwrap();
        p.sequence_mut(seq).unwrap().video_tracks[0].items.push(under);
        if nested {
            let nest = p.make_track_item(cut, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, rate.tick_of(48)), rate).unwrap();
            p.sequence_mut(seq).unwrap().video_tracks[1].items.push(nest);
        }
        (Arc::new(p), seq, m)
    }

    /// (99th percentile, mean) of the per-pixel max RGB difference over black, 8-bit.
    fn levels(cpu: &Image, gpu: &Image) -> (u32, f64) {
        let (a, b) = (cpu.over_black_rgba8(), gpu.over_black_rgba8());
        let mut d: Vec<u32> = a
            .as_chunks::<4>()
            .0
            .iter()
            .zip(b.as_chunks::<4>().0)
            .map(|(x, y)| (0..3).map(|k| (x[k] as u32).abs_diff(y[k] as u32)).max().unwrap_or(0))
            .collect();
        d.sort_unstable();
        (d[d.len() * 99 / 100], d.iter().sum::<u32>() as f64 / d.len() as f64)
    }

    /// Export draws GL Transitions (and Push) with the stages the Program monitor uses, so an
    /// exported frame is the monitor's picture exactly, and within a level of the CPU reference
    /// (the fallback when a transition cannot be drawn on the device); also inside a nest.
    #[test]
    fn transitions_export_as_the_monitor_shows_them() {
        let Some(mut r) = ExportRenderer::new() else {
            eprintln!("SKIPPED (gpu export transitions): no adapter");
            return;
        };
        let rate = filmcraft_time::FrameRate::FPS_24;
        for effect in ["gl_mosaic", "gl_crosszoom", "gl_ripple", "push"] {
            for nested in [false, true] {
                let (p, seq, m) = transition_project(effect, nested);
                for f in [19, 24, 28] {
                    let t = rate.tick_of(f);
                    let what = format!("{effect} nested {nested} frame {f}");
                    let opts = RenderOptions::default();
                    let plan = plan_frame(&p, seq, t, opts, &m);
                    let FramePlan::Composite { steps, .. } = &plan else { panic!("{what}: not planned as GPU steps") };
                    assert!(steps.iter().any(|s| matches!(s, filmcraft_render::plan::PlanStep::Transition { effect: e, .. } if e.effect == effect)), "{what}");
                    let before = (r.compositor.gpu_transitions, r.compositor.cpu_transitions);
                    let export = r.render(&p, seq, t, opts, &m).expect("GPU rendered the transition");
                    assert_eq!(r.compositor.gpu_transitions, before.0 + 1, "{what}: drawn by the GPU stage");
                    assert_eq!(r.compositor.cpu_transitions, before.1, "{what}: no CPU fallback");
                    // the Program monitor's path (composite, then the same accumulator)
                    r.compositor.composite_prepared(&plan, Some(&prepare(&plan)));
                    let (w, h, monitor) = r.compositor.read_accumulator().expect("read-back");
                    assert_eq!((w as usize, h as usize), (export.w, export.h));
                    assert!(monitor == export.px, "{what}: export differs from the monitor");
                    let cpu = filmcraft_render::render_sequence(&p, seq, t, opts, &m);
                    let (p99, mean) = levels(&cpu, &export);
                    eprintln!("{what}: export vs CPU reference p99 {p99}, mean {mean:.3}");
                    assert!(p99 <= 1 && mean <= 0.5, "{what}: export vs CPU reference p99 {p99}, mean {mean:.3}");
                    // the 8-bit export path draws the same transition on the GPU and encodes it
                    // exactly as the float frame is encoded on the CPU
                    let before8 = r.compositor.gpu_transitions;
                    let (w8, h8, gpu8) = r.render_rgba8(&p, seq, t, opts, &m).expect("GPU RGBA8 transition frame");
                    assert_eq!(r.compositor.gpu_transitions, before8 + 1, "{what}: render_rgba8 drawn by the GPU stage");
                    assert_eq!((w8, h8), (export.w, export.h));
                    assert!(gpu8 == export.over_black_rgba8(), "{what}: render_rgba8 differs from the float frame over black");
                }
            }
        }
    }

    /// The 8-bit frame encoded by the export LUT resolve shader matches the float frame encoded
    /// on the CPU (`over_black_rgba8`) byte-for-byte, and stays within the CPU parity bound.
    #[test]
    fn rgba8_frame_matches_float_frame_over_black() {
        let Some(mut r) = ExportRenderer::new() else {
            eprintln!("SKIPPED (gpu export parity): no adapter");
            return;
        };
        for transform in [false, true] {
            let (p, seq, m) = project(transform);
            let t = filmcraft_time::FrameRate::FPS_24.tick_of(10);
            let (w, h, gpu8) = r.render_rgba8(&p, seq, t, RenderOptions::default(), &m).expect("GPU rendered the frame");
            let float = r.render(&p, seq, t, RenderOptions::default(), &m).expect("GPU rendered the frame");
            assert_eq!((w, h), (float.w, float.h));
            assert_eq!(gpu8.len(), w * h * 4);
            assert!(gpu8 == float.over_black_rgba8(), "transform {transform}: 8-bit frame must match the encoded float frame byte-for-byte");
            let cpu = filmcraft_render::render_sequence(&p, seq, t, RenderOptions::default(), &m);
            let decoded = Image { w, h, px: gpu8.iter().map(|&c| filmcraft_color::srgb_u8_to_linear_table()[c as usize]).collect() };
            assert_parity(&cpu, &decoded, "rgba8");
        }
    }

    /// The bench's multi-layer effects stack over a demo scene: three scaled layers, one at 70 %
    /// opacity, with Proc Amp, Gaussian Blur, Vignette, Brightness & Contrast and Sharpen.
    fn demo_effects_project(scene: filmcraft_media::DemoScene, w: u32, h: u32, rate: filmcraft_time::FrameRate) -> (Project, ItemId, SourceMap) {
        use filmcraft_project::{ParamValue, find_effect};
        let mut p = Project::new("x");
        let g = GeneratorSource::new(Generator::Demo(scene), w, h, rate, Tick(2 * TICKS_PER_SECOND));
        let info = g.info().clone();
        let id = p.add_item(
            &info.name.clone(),
            Label::Iris,
            ItemKind::Media(MediaClip {
                media: MediaRef::Generator(g.generator.clone()),
                info,
                interpret: Default::default(),
                mark_in: None,
                mark_out: None,
                markers: vec![],
                offline: false,
                proxy: None,
                identity: None,
            }),
            None,
        );
        let seq = p.new_sequence("s", SequenceSettings { width: w, height: h, frame_rate: rate, ..Default::default() }, 3, 1, None);
        let layers: [(f64, (f64, f64), f64, &[(&str, &[(&str, f64)])]); 3] = [
            (100.0, (0.0, 0.0), 100.0, &[("proc_amp", &[("contrast", 110.0)]), ("gaussian_blur", &[("blurriness", 8.0)]), ("vignette", &[])]),
            (50.0, (160.0, -90.0), 100.0, &[("brightness_contrast", &[("brightness", 10.0), ("contrast", 15.0)])]),
            (33.0, (-186.0, 100.0), 70.0, &[("sharpen", &[("amount", 40.0)])]),
        ];
        for (i, (scale, (px, py), opacity, fx)) in layers.into_iter().enumerate() {
            let mut v = p.make_track_item(id, TrackKind::Video, Tick::ZERO, TimeRange::new(Tick::ZERO, rate.tick_of(24)), rate).unwrap();
            let motion = v.effect_mut("motion").unwrap();
            motion.param_mut("scale").unwrap().value = ParamValue::Float(scale);
            motion.param_mut("position").unwrap().value = ParamValue::Vec2(filmcraft_geom::Vec2::new(px, py));
            v.effect_mut("opacity").unwrap().param_mut("opacity").unwrap().value = ParamValue::Float(opacity);
            for (k, (fx_id, set)) in fx.iter().enumerate() {
                let mut e = find_effect(fx_id).unwrap().instance();
                for (name, val) in *set {
                    e.param_mut(name).unwrap().value = ParamValue::Float(*val);
                }
                v.effects.insert(k, e);
            }
            p.sequence_mut(seq).unwrap().video_tracks[i].items.push(v);
        }
        let mut m = SourceMap::default();
        m.0.insert(id, Arc::new(g));
        (p, seq, m)
    }

    /// Across all demo scenes and the bench's multi-layer effects stack, `render_rgba8` resolves
    /// the accumulator through the 4096-entry sRGB LUT and matches `render` +
    /// `over_black_rgba8()` byte-for-byte.
    #[test]
    fn rgba8_frame_with_effects_matches_float_frame_over_black() {
        let Some(mut r) = ExportRenderer::new() else {
            eprintln!("SKIPPED (gpu export parity): no adapter");
            return;
        };
        let rate = filmcraft_time::FrameRate::FPS_24;
        for scene in filmcraft_media::DemoScene::ALL {
            let (p, seq, m) = demo_effects_project(scene, 640, 360, rate);
            for frame in [3, 17] {
                let t = rate.tick_of(frame);
                let (_, _, gpu8) = r.render_rgba8(&p, seq, t, RenderOptions::default(), &m).expect("GPU rendered the frame");
                let float = r.render(&p, seq, t, RenderOptions::default(), &m).expect("GPU rendered the frame");
                assert!(gpu8 == float.over_black_rgba8(), "{scene:?} frame {frame}: GPU LUT resolve must match over_black_rgba8 byte-for-byte");
            }
        }
    }

    /// The 256×256 raw f16 accumulator fixture (all 65,536 halves in R, G and B) and its
    /// expected `over_black_rgba8()` output.
    fn raw_f16_lut_fixture() -> (u32, u32, Vec<u8>, Vec<u8>) {
        let (fw, fh) = (256u32, 256u32);
        let mut raw_f16_bytes = Vec::with_capacity((fw * fh * 8) as usize);
        let mut expected_f32 = Vec::with_capacity((fw * fh * 4) as usize);
        for i in 0..=u16::MAX {
            let r = i;
            let g = i.wrapping_add(21_845);
            let b = i.wrapping_add(43_690);
            let a = 0x3c00u16; // 1.0
            for h in [r, g, b, a] {
                raw_f16_bytes.extend_from_slice(&h.to_le_bytes());
                expected_f32.push(f16_to_f32(h));
            }
        }
        let expected_rgba8 = Image { w: fw as usize, h: fh as usize, px: expected_f32 }.over_black_rgba8();
        (fw, fh, raw_f16_bytes, expected_rgba8)
    }

    /// A prepared-plan fixture of the LUT's rounding midpoints `(k * 0.5 / 4095.0) ± 1 ULP` for
    /// `k in 0..=8190`, plus `NaN`, `±Inf`, `-0.0`, negatives and `> 1.0`. `prepare` quantizes the
    /// f32 inputs to f16 before either render path runs, so neighbours may collapse to one half;
    /// every half reaches the resolve pass in [`raw_f16_lut_fixture`].
    fn rounding_threshold_lut_fixture() -> (usize, usize, FramePlan, crate::PreparedPlan) {
        let mut vals = vec![f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -0.0, 0.0, 1.0, -1.0, 2.0, 1e-30];
        vals.extend(
            (0..=8190).map(|k| k as f32 * 0.5 / 4095.0).flat_map(|b| [b, f32::from_bits(b.to_bits() + 1), f32::from_bits(b.to_bits().saturating_sub(1))]),
        );
        let (tw, th) = (256usize, 36usize);
        let mut px = vec![1.0f32; tw * th * 4];
        for (idx, &v) in vals.iter().enumerate() {
            let p = idx % (tw * th);
            px[p * 4] = v;
            px[p * 4 + 1] = vals[(idx + 8191) % vals.len()];
            px[p * 4 + 2] = vals[(idx + 16382) % vals.len()];
            px[p * 4 + 3] = 1.0;
        }
        let plan = FramePlan::Image(Image { w: tw, h: th, px });
        let prep = prepare(&plan);
        (tw, th, plan, prep)
    }

    /// With `required` (`FILMCRAFT_REQUIRE_GPU` set) a run that rendered on no device fails;
    /// otherwise it is skipped.
    fn require_lut_backend_renders(rendered_count: usize, required: bool, reason: &str) -> bool {
        if rendered_count > 0 {
            return true;
        }
        assert!(!required, "required GPU LUT precision test did not run: {reason}");
        eprintln!("SKIPPED (gpu LUT precision): {reason}");
        false
    }

    /// The three LUT resolve precision checks on one device; returns the transformed scene's
    /// bytes for the cross-backend comparison.
    fn verify_lut_precision_on_device(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        info: &wgpu::AdapterInfo,
        (fw, fh, raw_f16_bytes, expected_f16_rgba8): (u32, u32, &[u8], &[u8]),
        (tw, th, threshold_plan, threshold_prep): (usize, usize, &FramePlan, &crate::PreparedPlan),
        (scene_p, scene_seq, scene_t, scene_m): (&Project, ItemId, Tick, &SourceMap),
    ) -> Vec<u8> {
        let mut c = GpuCompositor::new(device, queue);

        // 1. All 65,536 raw f16 bit patterns in the accumulator through `resolve_export_rgba8`.
        c.render_plan(&FramePlan::Layers { width: fw as usize, height: fh as usize, layers: Vec::new() }, None, true, false);
        let accum_tex = c.accum.as_ref().expect("accum allocated").0.clone();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &accum_tex, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            raw_f16_bytes,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(fw * 8), rows_per_image: Some(fh) },
            wgpu::Extent3d { width: fw, height: fh, depth_or_array_layers: 1 },
        );
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("test-lut-f16") });
        c.resolve_export_rgba8(&mut enc).expect("resolve_export_rgba8");
        let size = c.copy_output(&mut enc).expect("copy_output");
        let sub = queue.submit([enc.finish()]);
        let (_, _, got_f16_rgba8) = c.map_output(size, sub).expect("map_output");
        assert!(got_f16_rgba8 == expected_f16_rgba8, "{:?} ({}) failed raw f16 LUT parity", info.backend, info.name);

        // 2. The rounding thresholds (quantized to f16 by `prepare`) through both export paths.
        let (_, _, got_rgba8) = c.render_export_rgba8_prepared(threshold_plan, Some(threshold_prep)).expect("threshold rgba8");
        let (_, _, got_f32) = c.render_export_prepared(threshold_plan, Some(threshold_prep)).expect("threshold f32");
        assert!(got_rgba8 == Image { w: tw, h: th, px: got_f32 }.over_black_rgba8(), "{:?} ({}) failed rounding threshold LUT parity", info.backend, info.name);

        // 3. A transformed scene on this backend; its bytes go to the cross-backend check.
        let plan = plan_frame(scene_p, scene_seq, scene_t, RenderOptions::default(), scene_m);
        let prep = prepare(&plan);
        let (_, _, scene_rgba8) = c.render_export_rgba8_prepared(&plan, Some(&prep)).expect("scene rgba8");
        let (sw, sh, scene_f32) = c.render_export_prepared(&plan, Some(&prep)).expect("scene f32");
        assert!(
            scene_rgba8 == Image { w: sw as usize, h: sh as usize, px: scene_f32 }.over_black_rgba8(),
            "{:?} ({}) scene rgba8 != over_black_rgba8",
            info.backend,
            info.name
        );
        scene_rgba8
    }

    #[test]
    fn lut_precision_policy_skips_without_adapter_unless_required() {
        assert!(!require_lut_backend_renders(0, false, "no adapter"));
        assert!(require_lut_backend_renders(1, false, "unused"));
        assert!(require_lut_backend_renders(1, true, "unused"));
    }

    #[test]
    #[should_panic(expected = "required GPU LUT precision test did not run")]
    fn lut_precision_policy_fails_when_gpu_required_and_none_rendered() {
        let _ = require_lut_backend_renders(0, true, "no adapter");
    }

    /// Precision of the 4096-entry LUT export resolve pass, on every backend this machine has:
    /// 1. all 65,536 f16 bit patterns (signalling and quiet NaN, ±Inf, -0.0, negatives,
    ///    subnormals, > 1.0 and every half in [0, 1]) written into the accumulator resolve to the
    ///    bytes `Image::over_black_rgba8()` gives;
    /// 2. the LUT's rounding thresholds, quantized to f16 by `prepare`, give the same bytes
    ///    through `render_export_rgba8_prepared` as `render_export_prepared` + `over_black_rgba8()`;
    /// 3. backends agree within one level. Skips without an adapter unless `FILMCRAFT_REQUIRE_GPU`.
    #[test]
    fn rgba8_lut_matches_over_black_rgba8_across_all_f16_and_rounding_thresholds() {
        let required = std::env::var_os("FILMCRAFT_REQUIRE_GPU").is_some();
        if wgpu::Instance::enabled_backend_features().is_empty() {
            require_lut_backend_renders(0, required, "no wgpu backend enabled");
            return;
        }
        let (fw, fh, raw_f16_bytes, expected_f16_rgba8) = raw_f16_lut_fixture();
        let (tw, th, threshold_plan, threshold_prep) = rounding_threshold_lut_fixture();
        let (scene_p, scene_seq, scene_m) = project(true);
        let scene_t = filmcraft_time::FrameRate::FPS_24.tick_of(10);

        // the renderer's own instance (`ExportRenderer::new` uses the default one)
        let instance = wgpu::Instance::default();
        let power = wgpu::PowerPreference::from_env().unwrap_or(wgpu::PowerPreference::HighPerformance);
        let adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        let mut backends: Vec<wgpu::Backend> = Vec::new();
        for adapter in &adapters {
            let backend = adapter.get_info().backend;
            if !backends.contains(&backend) {
                backends.push(backend);
            }
        }
        let mut renders: Vec<(wgpu::Backend, String, Vec<u8>)> = Vec::new();
        for backend in backends {
            let group: Vec<wgpu::Adapter> = adapters.iter().filter(|a| a.get_info().backend == backend).cloned().collect();
            let Ok(adapter) = crate::adapter::pick(&group, None, power) else {
                continue;
            };
            if !adapter.get_texture_format_features(crate::ACCUM_FORMAT).allowed_usages.contains(wgpu::TextureUsages::RENDER_ATTACHMENT) {
                continue;
            }
            let info = adapter.get_info();
            let Ok((device, queue)) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())) else {
                continue;
            };
            eprintln!("GPU test adapter (LUT precision): {} ({:?}, {:?})", info.name, info.backend, info.device_type);
            let scene = verify_lut_precision_on_device(
                &device,
                &queue,
                &info,
                (fw, fh, &raw_f16_bytes, &expected_f16_rgba8),
                (tw, th, &threshold_plan, &threshold_prep),
                (&scene_p, scene_seq, scene_t, &scene_m),
            );
            renders.push((info.backend, info.name.clone(), scene));
        }
        let reason = if adapters.is_empty() { "no GPU adapter available" } else { "no usable GPU device could be created" };
        if !require_lut_backend_renders(renders.len(), required, reason) {
            return;
        }
        for w in renders.windows(2) {
            let ((b0, n0, r0), (b1, n1, r1)) = (&w[0], &w[1]);
            let max_diff = r0.iter().zip(r1).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
            eprintln!("cross-backend {b0:?} ({n0}) vs {b1:?} ({n1}): max channel diff = {max_diff}");
            assert!(max_diff <= 1, "cross-backend {b0:?} vs {b1:?} differed by {max_diff}");
        }
    }

    #[test]
    fn gpu_render_impls_frame_renderer() {
        // dev-dependency: the same wrapper the app and CLI register.
        struct R(ExportRenderer);
        impl FrameRenderer for R {
            fn render(&mut self, project: &Project, seq: ItemId, t: Tick, opts: RenderOptions, sources: &dyn SourceProvider) -> Option<Image> {
                self.0.render(project, seq, t, opts, sources)
            }
        }
        let Some(r) = ExportRenderer::new() else {
            eprintln!("SKIPPED (gpu export parity): no adapter");
            return;
        };
        let (p, seq, m) = project(false);
        let gpu = R(r).render(&p, seq, Tick::ZERO, RenderOptions::default(), &m).expect("rendered");
        let cpu = filmcraft_render::render_sequence(&p, seq, Tick::ZERO, RenderOptions::default(), &m);
        assert_parity(&cpu, &gpu, "frame renderer");
    }

    #[test]
    fn f16_roundtrip() {
        for v in [0.0f32, 1.0, 0.5, 2.0, 1e-3, 65504.0, -1.0, -0.25] {
            assert!((f16_to_f32(f32_to_f16(v)) - v).abs() <= EPS * v.abs().max(1.0), "f16 {v}");
        }
    }
}
