use super::*;

/// A JPEG `w` × `h`, its left half red and its right half blue.
fn halves_jpeg(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbImage::from_fn(w, h, |x, _| if x < w / 2 { image::Rgb([230, 20, 20]) } else { image::Rgb([20, 20, 230]) });
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img).write_to(&mut out, image::ImageFormat::Jpeg).unwrap();
    out.into_inner()
}

fn png(w: u32, h: u32) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(w, h, image::Rgba([10, 200, 10, 255]));
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img).write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn px(r: &Raster, x: u32, y: u32) -> [u8; 4] {
    let i = ((y * r.width + x) * 4) as usize;
    [r.rgba[i], r.rgba[i + 1], r.rgba[i + 2], r.rgba[i + 3]]
}

#[test]
fn a_sideways_phone_photo_is_turned_upright() {
    let plain = halves_jpeg(64, 32);
    let r = decode_raster("plain.jpg", &plain).unwrap();
    assert_eq!((r.width, r.height), (64, 32));
    // orientation 6: stored turned a quarter anticlockwise, shown a quarter clockwise — the left
    // (red) half ends up on top
    let tagged = crate::exif::build::into_jpeg(&plain, &crate::exif::build::block(Some(6), None, None, None));
    let r = decode_raster("tagged.jpg", &tagged).unwrap();
    assert_eq!((r.width, r.height), (32, 64), "portrait now");
    let (top, bottom) = (px(&r, 16, 8), px(&r, 16, 56));
    assert!(top[0] > 150 && top[2] < 100, "top is red: {top:?}");
    assert!(bottom[2] > 150 && bottom[0] < 100, "bottom is blue: {bottom:?}");
    let s = StillSource::decode("tagged.jpg", &tagged).unwrap();
    let v = s.info().video.as_ref().unwrap();
    assert_eq!((v.width, v.height), (32, 64), "the media's size is the upright one");
}

#[test]
fn huge_stills_are_held_at_the_cap() {
    assert_eq!(capped_size(8064, 6048), (8064, 6048), "a 48-megapixel phone photo stays whole");
    assert_eq!(capped_size(16384, 8192), (8192, 4096));
    assert_eq!(capped_size(9000, 3), (8192, 3));
    assert_eq!(capped_size(1, 100_000), (1, 8192));
    let r = decode_raster("wide.png", &png(9000, 4)).unwrap();
    assert_eq!((r.width, r.height), (8192, 4));
    assert_eq!(r.rgba.len(), 8192 * 4 * 4);
    assert_eq!(px(&r, 5000, 2), [10, 200, 10, 255]);
    // a buffer that is not w × h × 4 is left as it is
    assert_eq!(cap_rgba(9000, 2, vec![0; 7]), (9000, 2, vec![0; 7]));
}

#[test]
fn hostile_stills_are_errors_not_crashes() {
    // a PNG whose header claims 100 000 × 100 000 pixels (40 GB): refused before allocating
    let mut b = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut ihdr = b"IHDR".to_vec();
    ihdr.extend(100_000u32.to_be_bytes());
    ihdr.extend(100_000u32.to_be_bytes());
    ihdr.extend([8, 6, 0, 0, 0]);
    b.extend(13u32.to_be_bytes());
    b.extend(&ihdr);
    b.extend(crc32(&ihdr).to_be_bytes());
    b.extend(0u32.to_be_bytes());
    b.extend(b"IEND");
    b.extend(crc32(b"IEND").to_be_bytes());
    let e = std::panic::catch_unwind(|| decode_raster("huge.png", &b).map(|r| r.width)).expect("no panic");
    assert!(e.is_err(), "{e:?}");
    // truncated and bit-flipped photos, the EXIF block included: an error or a picture
    let good = crate::exif::build::into_jpeg(&halves_jpeg(24, 16), &crate::exif::build::block(Some(8), Some("2022:11:02 19:39:53"), None, Some((1.0, 2.0))));
    for n in (0..good.len()).step_by(7) {
        let _ = std::panic::catch_unwind(|| decode_raster("cut.jpg", &good[..n]).map(|r| r.width)).expect("no panic on truncation");
    }
    for i in (0..good.len()).step_by(3) {
        let mut bad = good.clone();
        bad[i] ^= 0x5A;
        let _ = std::panic::catch_unwind(|| decode_raster("flip.jpg", &bad).map(|r| r.width)).expect("no panic on a bit flip");
    }
    assert!(decode_raster("empty.jpg", &[]).is_err());
}

/// CRC-32 (ISO 3309), as PNG chunks carry it.
fn crc32(b: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &x in b {
        c ^= u32::from(x);
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}
