//! 360° layers drawn through their virtual camera on the GPU against the CPU reference
//! (`filmcraft_render::vr`).

use super::*;
use filmcraft_geom::Affine;
use filmcraft_media::Spherical;
use filmcraft_media::spherical::FisheyeLens;
use filmcraft_render::Blend;
use filmcraft_render::plan::{VrLayer, execute_cpu};
use filmcraft_render::vr::{SourceModel, View};

fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
    let instance = wgpu::Instance::default();
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
    pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
}

/// A 4:2:0 picture with smooth detail in every direction (luma waves, chroma ramps).
fn pattern(w: u32, h: u32) -> Arc<VideoFrame> {
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let y: Vec<u8> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32 / w as f32, (i / w) as f32 / h as f32);
            (128.0 + 80.0 * (x * 37.0).sin() * (y * 23.0).cos() + 20.0 * (y * 5.0).sin()) as u8
        })
        .collect();
    let u: Vec<u8> = (0..cw * ch).map(|i| (64 + (i / cw) * 128 / ch) as u8).collect();
    let v: Vec<u8> = (0..cw * ch).map(|i| (200 - (i % cw) * 100 / cw) as u8).collect();
    Arc::new(VideoFrame {
        width: w,
        height: h,
        data: PixelData::Yuv8 { planes: [Arc::new(y), Arc::new(u), Arc::new(v)], chroma: Chroma::C420, alpha: None },
        color: filmcraft_color::ColorInfo::REC709,
        par: (1, 1),
        pts: Default::default(),
    })
}

fn dual() -> Spherical {
    let lens = |k: usize, yaw: f64, cx: f64| FisheyeLens {
        rect: [k as f64 * 0.5, 0.0, 0.5, 1.0],
        cx,
        cy: 0.49,
        radius: 0.49,
        half_fov: 100.0,
        poly: if k == 0 { vec![] } else { vec![1.02, -0.05, 0.03] },
        yaw,
        pitch: 0.5,
        roll: -0.3,
    };
    Spherical::dual_fisheye(vec![lens(0, 180.0, 0.5), lens(1, 0.0, 0.51)], "test")
}

/// GPU and CPU pictures of one 360° layer, compared (8-bit: p99, mean of the largest channel
/// difference).
fn compare(frame: Arc<VideoFrame>, spherical: &Spherical, view: View, out: (u32, u32), matrix: Affine, canvas: (usize, usize)) -> Option<(u32, f64)> {
    let (dev, q) = device()?;
    let mut c = GpuCompositor::new(&dev, &q);
    let model = SourceModel::new(spherical, (frame.width as usize, frame.height as usize));
    let taps = filmcraft_render::vr::taps(&model, &view, out.0 as usize) as u32;
    let vr = Some(Arc::new(VrLayer { view, model, out, taps }));
    let plan = FramePlan::Layers { width: canvas.0, height: canvas.1, layers: vec![PlanLayer { vr, ..PlanLayer::new(frame, matrix, 1.0, Blend::Normal) }] };
    let cpu = execute_cpu(&plan).over_black_rgba8();
    c.composite(&plan);
    let (gw, gh, gpu) = c.read_output()?;
    assert_eq!((gw as usize, gh as usize), canvas);
    let mut diffs: Vec<u32> =
        cpu.chunks(4).zip(gpu.chunks(4)).map(|(a, b)| (0..3).map(|k| (a[k] as i32 - b[k] as i32).unsigned_abs()).max().unwrap_or(0)).collect();
    diffs.sort_unstable();
    let p99 = diffs[diffs.len() * 99 / 100];
    let mean = diffs.iter().sum::<u32>() as f64 / diffs.len() as f64;
    Some((p99, mean))
}

#[test]
fn equirect_views_match_the_cpu() {
    let src = pattern(720, 360);
    let s = Spherical::equirect();
    let cases = [
        // across the ±180° seam, looking up
        (View::flat(178.0, 20.0, 5.0, 100.0, 0.0, 320, 180), (320, 180)),
        // pulled back into a tiny planet
        (View::flat(30.0, -90.0, 0.0, 120.0, 1.0, 180, 180), (180, 180)),
        // a vertical frame
        (View::flat(-60.0, -10.0, 0.0, 70.0, 0.3, 108, 192), (108, 192)),
        // the sphere turned
        (View::sphere(45.0, 15.0, -10.0), (360, 180)),
    ];
    for (view, out) in cases {
        let Some((p99, mean)) = compare(src.clone(), &s, view, out, Affine::IDENTITY, (out.0 as usize, out.1 as usize)) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!(p99 <= 6 && mean < 1.5, "{view:?}: p99 {p99}, mean {mean}");
    }
}

#[test]
fn dual_fisheye_views_match_the_cpu() {
    let src = pattern(800, 400);
    let s = dual();
    for (yaw, pitch) in [(0.0, 0.0), (90.0, 10.0), (-95.0, -30.0), (180.0, 45.0)] {
        let view = View::flat(yaw, pitch, 0.0, 90.0, 0.0, 256, 144);
        let Some((p99, mean)) = compare(src.clone(), &s, view, (256, 144), Affine::IDENTITY, (256, 144)) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        assert!(p99 <= 6 && mean < 1.5, "yaw {yaw} pitch {pitch}: p99 {p99}, mean {mean}");
    }
}

#[test]
fn a_placed_and_shrunk_view_matches_the_cpu() {
    // Motion places the camera's picture smaller, off centre: the supersampling covers it
    let src = pattern(1440, 720);
    let view = View::flat(20.0, 0.0, 0.0, 120.0, 0.0, 400, 300);
    let m = Affine::translate(30.0, 20.0).then_apply(&Affine::scale(0.5, 0.5));
    let Some((p99, mean)) = compare(src, &Spherical::equirect(), view, (400, 300), m, (260, 190)) else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    assert!(p99 <= 6 && mean < 1.5, "p99 {p99}, mean {mean}");
}
