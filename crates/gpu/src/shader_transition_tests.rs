//! GL Transitions on the GPU against their Rust references (`execute_cpu`). Skipped without a GPU
//! adapter, like the other GPU tests.

use super::*;
use filmcraft_geom::Affine;
use filmcraft_project::ParamValue;
use filmcraft_render::Blend;
use filmcraft_render::plan::execute_cpu;
use filmcraft_render::transitions::shader;

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

/// A textured test picture: smooth ramps, a checker and a disc (`side` 0 warm, 1 cool), with
/// `alpha` coverage.
fn picture(w: usize, h: usize, side: usize, alpha: f32) -> Arc<VideoFrame> {
    let px = (0..w * h)
        .flat_map(|i| {
            let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            let chk = if ((x * 8.0).floor() as i32 + (y * 6.0).floor() as i32) % 2 == 0 { 0.15 } else { 0.0 };
            let disc = if ((x - 0.6).powi(2) + (y - 0.4).powi(2)).sqrt() < 0.2 { 0.5 } else { 0.0 };
            let c = if side == 0 { [0.2 + 0.7 * x + chk, 0.1 + 0.3 * y, 0.05 + chk, 1.0] } else { [0.05 + disc, 0.2 + 0.2 * y + disc, 0.4 + 0.5 * y, 1.0] };
            [c[0] * alpha, c[1] * alpha, c[2] * alpha, alpha]
        })
        .collect();
    Arc::new(VideoFrame::rgba_f32(w as u32, h as u32, px))
}

/// The working image of a test picture (what the CPU reference draws a layer of it as).
fn image(frame: &VideoFrame) -> filmcraft_render::Image {
    let (w, h, px) = frame.to_linear_f32_decimated_with(1, None);
    filmcraft_render::Image { w, h, px }
}

/// The content field a content-aware transition reads (None for the others), from the two test
/// pictures as its anchors.
fn content_of(effect: &filmcraft_project::EffectInstance, w: usize, h: usize, alpha: f32) -> Option<Arc<filmcraft_render::transitions::content::Field>> {
    filmcraft_render::transitions::content::for_frames(effect, &image(&picture(w, h, 0, alpha)), &image(&picture(w, h, 1, alpha)))
}

fn transition_plan(w: usize, h: usize, effect: filmcraft_project::EffectInstance, progress: f32, alpha: f32) -> FramePlan {
    let layer = |side| PlanLayer::new(picture(w, h, side, alpha), Affine::IDENTITY, 1.0, Blend::Normal);
    let content = content_of(&effect, w, h, alpha);
    let mut above = PlanLayer::new(picture(w, h, 1, 0.5), Affine::IDENTITY, 0.4, Blend::Screen);
    above.opacity = 0.4;
    FramePlan::Composite {
        width: w,
        height: h,
        steps: vec![
            PlanStep::Layers(vec![PlanLayer::new(picture(w, h, 0, 1.0), Affine::IDENTITY, 1.0, Blend::Normal)]),
            PlanStep::Transition { inputs: [vec![layer(0)], vec![layer(1)]], effect, progress, scale: 1.0, content },
            PlanStep::Layers(vec![above]),
        ],
    }
}

/// (99th percentile, mean) of the per-pixel max RGB difference.
fn compare(cpu: &[u8], gpu: &[u8]) -> (u32, f64) {
    let mut d: Vec<u32> =
        cpu.as_chunks::<4>().0.iter().zip(gpu.as_chunks::<4>().0).map(|(a, b)| (0..3).map(|k| a[k].abs_diff(b[k]) as u32).max().unwrap()).collect();
    d.sort_unstable();
    (d[d.len() * 99 / 100], d.iter().sum::<u32>() as f64 / d.len() as f64)
}

#[test]
fn gpu_shader_transitions_match_the_cpu_reference() {
    let Some((dev, q)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut c = GpuCompositor::new(&dev, &q);
    let mut failures = Vec::new();
    let mut cases = 0u64;
    // an odd aspect ratio and a 16:9 frame; opaque and translucent clips
    for (w, h, alpha) in [(97usize, 41usize, 1.0f32), (128, 72, 0.6)] {
        for id in shader::transition_ids() {
            let mut worst = (0u32, 0.0f64);
            for progress in [0.0f32, 0.13, 0.37, 0.5, 0.81, 1.0] {
                let mut effect = filmcraft_project::find_effect(id).unwrap().instance();
                // exercise the host's easing on half the cases
                if progress > 0.4 {
                    effect.param_mut("easing").unwrap().value = ParamValue::Choice(9);
                }
                let plan = transition_plan(w, h, effect, progress, alpha);
                let cpu = execute_cpu(&plan).over_black_rgba8();
                c.composite_prepared(&plan, Some(&prepare(&plan)));
                let (_, _, gpu) = c.read_output().expect("readback");
                cases += 1;
                let (p99, mean) = compare(&cpu, &gpu);
                worst = (worst.0.max(p99), worst.1.max(mean));
                // the endpoints pass the inputs through: half-float rounding only
                // FilmCraft's content-aware engines (the flow transitions, the Content-Aware Wipe)
                // agree within one 8-bit level
                let limit = if progress == 0.0 || progress == 1.0 || shader::reads_content(id) { (1, 0.5) } else { (6, 1.5) };
                if p99 > limit.0 || mean >= limit.1 {
                    failures.push(format!("{id} {w}x{h} alpha {alpha} p={progress}: p99 {p99}, mean {mean:.3}"));
                }
            }
            eprintln!("{id} {w}x{h}: worst p99 {}, mean {:.3}", worst.0, worst.1);
        }
    }
    assert!(failures.is_empty(), "GPU differs from the CPU reference:\n{}", failures.join("\n"));
    assert_eq!(c.gpu_transitions, cases, "every case ran on the GPU");
    assert_eq!(c.cpu_transitions, 0);
}

#[test]
fn shader_stage_only_takes_shader_transitions() {
    let Some((dev, _q)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let mut stage = shader_transition::ShaderStage::new(&dev);
    assert!(!stage.ready(&dev, "cross_dissolve"));
    assert!(!stage.ready(&dev, "push"));
    assert!(!stage.ready(&dev, ""));
    for id in shader::transition_ids() {
        assert!(stage.ready(&dev, id), "{id} builds");
        assert!(stage.ready(&dev, id), "{id} is cached");
    }
    // the presets of an engine share its one pipeline
    assert_eq!(stage.pipelines.len(), shader::shaders().len() + shader::engines().len());
}

/// GPU vs CPU speed of every shader transition on a 1080p frame (prints a table):
/// `cargo test -p filmcraft-gpu --release gl_transition_speed -- --ignored --nocapture`.
/// Hold the GPU lock while it runs on a shared machine.
#[test]
#[ignore]
fn gl_transition_speed() {
    let Some((dev, q)) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let (w, h) = (1920usize, 1080usize);
    let mut c = GpuCompositor::new(&dev, &q);
    let a = filmcraft_render::Image { w, h, px: picture(w, h, 0, 1.0).to_linear_f32_decimated_with(1, None).2 };
    let b = filmcraft_render::Image { w, h, px: picture(w, h, 1, 1.0).to_linear_f32_decimated_with(1, None).2 };
    eprintln!("{:<20} {:>12} {:>12} {:>12}", "transition", "CPU ms", "GPU ms", "GPU frame ms");
    for id in shader::transition_ids() {
        let effect = filmcraft_project::find_effect(id).unwrap().instance();
        // CPU reference alone (rayon over rows)
        let runs = 3;
        let t = std::time::Instant::now();
        for _ in 0..runs {
            let field = content_of(&effect, w, h, 1.0);
            std::hint::black_box(filmcraft_render::transitions::apply_with(&effect, &a, &b, 0.43, 1.0, field.as_deref()));
        }
        let cpu_ms = t.elapsed().as_secs_f64() * 1000.0 / runs as f64;
        // GPU: the transition pass alone on resident inputs, waited for
        let plan = FramePlan::Composite {
            width: w,
            height: h,
            steps: vec![PlanStep::Transition {
                inputs: [
                    vec![PlanLayer::new(picture(w, h, 0, 1.0), Affine::IDENTITY, 1.0, Blend::Normal)],
                    vec![PlanLayer::new(picture(w, h, 1, 1.0), Affine::IDENTITY, 1.0, Blend::Normal)],
                ],
                effect: effect.clone(),
                progress: 0.43,
                scale: 1.0,
                content: content_of(&effect, w, h, 1.0),
            }],
        };
        let prep = prepare(&plan);
        c.composite_prepared(&plan, Some(&prep));
        let _ = c.read_output();
        let views = [c.transition_inputs[0].as_ref().map(|t| t.1.clone()), c.transition_inputs[1].as_ref().map(|t| t.1.clone())];
        let canvas = c.accum.as_ref().map(|t| t.1.clone());
        let (Some(va), Some(vb), Some(canvas)) = (views[0].clone(), views[1].clone(), canvas) else { panic!("targets") };
        let gpu_runs = 20;
        let _ = dev.poll(wgpu::PollType::wait_indefinitely());
        let t = std::time::Instant::now();
        for _ in 0..gpu_runs {
            let field = content_of(&effect, w, h, 1.0);
            assert!(c.shader_transitions.draw(&dev, &q, &canvas, [&va, &vb], (w as u32, h as u32), &effect, 0.43, 1.0, field.as_deref()));
        }
        let _ = dev.poll(wgpu::PollType::wait_indefinitely());
        let gpu_ms = t.elapsed().as_secs_f64() * 1000.0 / gpu_runs as f64;
        // a whole GPU frame: both inputs drawn, the transition, the resolve and a read-back
        let frames = 5;
        let t = std::time::Instant::now();
        for _ in 0..frames {
            c.composite_prepared(&plan, Some(&prep));
            let _ = c.read_output();
        }
        let frame_ms = t.elapsed().as_secs_f64() * 1000.0 / frames as f64;
        eprintln!("{:<20} {cpu_ms:>12.1} {gpu_ms:>12.2} {frame_ms:>12.1}", id);
    }
}
