use super::*;

fn models() -> Option<std::path::PathBuf> {
    let dir = std::path::PathBuf::from(std::env::var_os("HOME")?).join("Library/Application Support/FilmCraft/models");
    catalog::inpainter_installed(&dir).then_some(dir)
}

/// A drawn scene: a sky gradient over a striped ground, a dark disc on it.
fn scene(w: usize, h: usize, disc: bool) -> Vec<u8> {
    let mut px = vec![0u8; w * h * 4];
    for y in 0..h {
        for x in 0..w {
            let i = (y * w + x) * 4;
            let ground = y > h / 2;
            let mut c = if ground {
                if (x / 24 + y / 24) % 2 == 0 { [90, 140, 60] } else { [70, 115, 50] }
            } else {
                let t = y as f32 / (h / 2) as f32;
                [(120.0 + 80.0 * t) as u8, (160.0 + 60.0 * t) as u8, 230]
            };
            if disc && (x as f32 - w as f32 * 0.5).hypot(y as f32 - h as f32 * 0.55) < h as f32 * 0.12 {
                c = [30, 20, 25];
            }
            px[i..i + 4].copy_from_slice(&[c[0], c[1], c[2], 255]);
        }
    }
    px
}

/// The downloaded model fills a hole where a disc was with what is round it.
/// (Skips when MI-GAN is not downloaded.)
#[test]
fn a_hole_is_filled_like_what_is_round_it() {
    let Some(dir) = models() else {
        eprintln!("MI-GAN not downloaded; skipping");
        return;
    };
    let net = Inpainter::load(&dir, std::env::var_os("INPAINT_CPU").is_none()).unwrap();
    let (w, h) = (960usize, 540usize);
    let (with, without) = (scene(w, h, true), scene(w, h, false));
    let hole: Vec<u8> = (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f32, (i / w) as f32);
            if (x - w as f32 * 0.5).hypot(y - h as f32 * 0.55) < h as f32 * 0.12 + 6.0 { 255 } else { 0 }
        })
        .collect();
    let t = std::time::Instant::now();
    let out = net.inpaint(&with, w, h, &hole).unwrap();
    eprintln!("MI-GAN on the {}: {:.0} ms", net.device_name(), t.elapsed().as_secs_f64() * 1000.0);
    let t = std::time::Instant::now();
    let _ = net.inpaint(&with, w, h, &hole).unwrap();
    eprintln!("then {:.0} ms", t.elapsed().as_secs_f64() * 1000.0);
    // inside the hole: near the scene without the disc, far from the disc
    let (mut to_truth, mut to_disc, mut n) = (0f64, 0f64, 0f64);
    for (i, a) in hole.iter().enumerate() {
        if *a == 255 {
            for c in 0..3 {
                to_truth += f64::from(out[i * 4 + c].abs_diff(without[i * 4 + c]));
                to_disc += f64::from(out[i * 4 + c].abs_diff(with[i * 4 + c]));
            }
            n += 3.0;
        }
    }
    eprintln!("in the hole: {:.1} levels from the scene without the disc, {:.1} from the disc", to_truth / n, to_disc / n);
    assert!(to_truth / n < 25.0 && to_disc / n > 50.0);
    // outside it, untouched
    assert!(hole.iter().enumerate().filter(|(_, a)| **a == 0).all(|(i, _)| out[i * 4..i * 4 + 3] == with[i * 4..i * 4 + 3]));
    if let Some(d) = std::env::var_os("INPAINT_OUT") {
        let d = std::path::PathBuf::from(d);
        for (name, px) in [("inpaint-in.png", &with), ("inpaint-out.png", &out)] {
            let img = image::RgbaImage::from_raw(w as u32, h as u32, px.clone()).unwrap();
            img.save(d.join(name)).unwrap();
        }
    }
    assert!(net.inpaint(&with, w, h, &hole[..10]).is_err());
}

#[test]
fn the_cut_round_a_hole_stays_in_the_picture() {
    assert_eq!(square(100, 100, 140, 140, 1920, 1080), (0, 0, 512, 512));
    assert_eq!(square(900, 500, 1000, 600, 1920, 1080), (694, 294, 512, 512));
    let (x, y, w, h) = square(0, 0, 1900, 900, 1920, 1080);
    assert!(x + w <= 1920 && y + h <= 1080 && w >= 1900 && h >= 900);
    let (x, y, w, h) = square(10, 10, 20, 20, 300, 200);
    assert_eq!((x, y, w, h), (0, 0, 300, 200), "a small picture: all of it");
    assert!(bounds(&[0; 16], 4, 4).is_none());
    assert_eq!(bounds(&[0, 0, 0, 0, 0, 255, 255, 0, 0, 0, 0, 0], 4, 3), Some((1, 1, 3, 2)));
    assert_eq!(area(&[0, 100, 200, 100], 2, 2, 1, 1, 1), vec![100]);
}

/// The GPU's passes give the CPU's picture. (Skips without MI-GAN or a GPU.)
#[test]
fn the_gpu_matches_the_cpu() {
    let Some(dir) = models() else { return };
    let net = Inpainter::load(&dir, true).unwrap();
    let Some(gpu) = &net.gpu else {
        eprintln!("no GPU; skipping");
        return;
    };
    let pic = area(&scene(960, 540, true), 960, 540, 4, SIDE, SIDE);
    let keep: Vec<f32> = (0..SIDE * SIDE).map(|i| if (i % SIDE).abs_diff(SIDE / 2) < 60 && (i / SIDE).abs_diff(SIDE / 2) < 60 { 0.0 } else { 1.0 }).collect();
    let x = input(&pic, &keep);
    let t = std::time::Instant::now();
    let cpu = forward(&net.w, &x).unwrap();
    let tc = t.elapsed().as_secs_f64() * 1000.0;
    let _ = gpu.run(&x).unwrap();
    let t = std::time::Instant::now();
    let g = gpu.run(&x).unwrap();
    let tg = t.elapsed().as_secs_f64() * 1000.0;
    let worst = cpu.iter().zip(&g).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("MI-GAN: CPU {tc:.0} ms, GPU {tg:.1} ms; the GPU within {worst:.2e} of the CPU");
    assert!(worst < 1e-3, "{worst}");
}
