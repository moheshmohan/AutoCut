//! Draw the AutoCut placeholder app icon (branding placeholder — replace with real art before
//! any distribution). Usage: `cargo run -p filmcraft-autocut --example make_icon -- <dir>` writes
//! `autocut-icon-1024.png` plus iconset sizes into `dir`.

use std::path::Path;

fn main() {
    let dir = std::env::args().nth(1).unwrap_or_else(|| ".".into());
    let dir = Path::new(&dir);
    std::fs::create_dir_all(dir).expect("create dir");
    let master = draw(1024);
    master.save(dir.join("autocut-icon-1024.png")).expect("save master");
    for size in [16, 32, 64, 128, 256, 512, 1024] {
        let img = image::imageops::resize(&master, size, size, image::imageops::FilterType::Lanczos3);
        let name = if size == 1024 { "icon_512x512@2x.png".to_string() } else { format!("icon_{size}x{size}.png") };
        // 16 and 32 also exist at 2x; 64 is only @2x of 32, 512 also @2x of 256 — iconutil wants:
        // 16, 16@2x(=32), 32, 32@2x(=64), 128, 128@2x(=256), 256, 256@2x(=512), 512, 512@2x(=1024)
        img.save(dir.join(&name)).expect("save size");
    }
    for (half, dbl) in
        [(16u32, "icon_16x16@2x.png"), (32, "icon_32x32@2x.png"), (128, "icon_128x128@2x.png"), (256, "icon_256x256@2x.png"), (512, "icon_512x512@2x.png")]
    {
        let img = image::imageops::resize(&master, half * 2, half * 2, image::imageops::FilterType::Lanczos3);
        img.save(dir.join(dbl)).expect("save @2x");
    }
    println!("wrote the AutoCut icon set to {}", dir.display());
}

/// A rounded-square indigo tile with a white diagonal "cut" band and notch — a placeholder that
/// is clearly not FilmCraft's mark.
fn draw(size: u32) -> image::RgbaImage {
    use image::Rgba;
    let s = size as f32;
    let radius = s * 0.22;
    let mut img = image::RgbaImage::new(size, size);
    for y in 0..size {
        for x in 0..size {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            // rounded-rect mask
            let cx = fx.clamp(radius, s - radius);
            let cy = fy.clamp(radius, s - radius);
            let inside = (fx - cx).powi(2) + (fy - cy).powi(2) <= radius * radius || (fx >= radius && fx <= s - radius) || (fy >= radius && fy <= s - radius);
            if !inside {
                continue;
            }
            // vertical indigo gradient
            let t = fy / s;
            let (r, g, b) = ((70.0 + 30.0 * t) as u8, (40.0 + 24.0 * t) as u8, (200.0 + 40.0 * t) as u8);
            // the diagonal cut band: |x - y| within a band, plus a gap (the "cut")
            let d = ((fx - fy) / s + 0.5).abs(); // 0 at the main diagonal
            let band = d < 0.055;
            let gap = d > 0.055 && d < 0.10;
            let px = if band {
                Rgba([245, 245, 250, 255])
            } else if gap {
                Rgba([0, 0, 0, 0]) // transparency reads as the cut
            } else {
                Rgba([r, g, b, 255])
            };
            img.put_pixel(x, y, px);
        }
    }
    img
}
