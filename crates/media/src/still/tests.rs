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

/// EXIF with only an Orientation tag of value `o`.
fn orientation_exif(o: u16) -> Vec<u8> {
    crate::exif::build::block(Some(o), None, None, None)
}

/// A 6 × 4 PNG or lossless WebP carrying EXIF Orientation `o` (in its eXIf / EXIF chunk).
fn tagged(format: image::ImageFormat, o: u16) -> Vec<u8> {
    use image::ImageEncoder as _;
    let px: Vec<u8> = (0..6 * 4).flat_map(|i| [(i * 10) as u8, 90, 160]).collect();
    let mut out = Vec::new();
    match format {
        image::ImageFormat::WebP => {
            let mut enc = image::codecs::webp::WebPEncoder::new_lossless(&mut out);
            enc.set_exif_metadata(orientation_exif(o)).unwrap();
            enc.write_image(&px, 6, 4, image::ExtendedColorType::Rgb8).unwrap();
        }
        _ => {
            let mut enc = image::codecs::png::PngEncoder::new(&mut out);
            enc.set_exif_metadata(orientation_exif(o)).unwrap();
            enc.write_image(&px, 6, 4, image::ExtendedColorType::Rgb8).unwrap();
        }
    }
    out
}

/// A little-endian TIFF of `w` × `h`, 8-bit RGB (`photometric` 2) or YCbCr (6), uncompressed in
/// one strip, with an Orientation tag `o`; its image directory after the pixels when `ifd_last`
/// (as libtiff writes it).
fn tiff(w: u32, h: u32, photometric: u16, o: u16, ifd_last: bool) -> Vec<u8> {
    let data = vec![128u8; (w * h * 3) as usize];
    // (tag, type, count, value); BitsPerSample's three SHORTs sit after the directory
    let entries: [(u16, u16, u32, u32); 10] = [
        (256, 4, 1, w),
        (257, 4, 1, h),
        (258, 3, 3, 0),
        (259, 3, 1, 1),
        (262, 3, 1, u32::from(photometric)),
        (273, 4, 1, 0),
        (274, 3, 1, u32::from(o)),
        (277, 3, 1, 3),
        (278, 4, 1, h),
        (279, 4, 1, data.len() as u32),
    ];
    let ifd_len = 2 + 12 * entries.len() as u32 + 4 + 6;
    let (ifd_at, data_at) = if ifd_last { (8 + data.len() as u32, 8) } else { (8, 8 + ifd_len) };
    let bits_at = ifd_at + ifd_len - 6;
    let mut ifd = (entries.len() as u16).to_le_bytes().to_vec();
    for (tag, ty, count, value) in entries {
        let value = match tag {
            273 => data_at,
            258 => bits_at,
            _ => value,
        };
        ifd.extend(tag.to_le_bytes());
        ifd.extend(ty.to_le_bytes());
        ifd.extend(count.to_le_bytes());
        if ty == 3 && count == 1 {
            ifd.extend((value as u16).to_le_bytes());
            ifd.extend([0, 0]);
        } else {
            ifd.extend(value.to_le_bytes());
        }
    }
    ifd.extend(0u32.to_le_bytes());
    for _ in 0..3 {
        ifd.extend(8u16.to_le_bytes());
    }
    let mut out = b"II*\0".to_vec();
    out.extend(ifd_at.to_le_bytes());
    if ifd_last {
        out.extend(&data);
        out.extend(ifd);
    } else {
        out.extend(ifd);
        out.extend(&data);
    }
    out
}

/// The engine's load-time still migration decides from `probe_geometry` alone, so the probe must
/// report what the decode does: the upright size and the orientation it turned the picture by.
#[test]
fn probe_geometry_agrees_with_the_decode() {
    for o in 1..=8u16 {
        let jpeg = crate::exif::build::into_jpeg(&halves_jpeg(24, 16), &orientation_exif(o));
        let cases = [
            ("photo.jpg", jpeg),
            ("shot.png", tagged(image::ImageFormat::Png, o)),
            ("shot.webp", tagged(image::ImageFormat::WebP, o)),
            ("scan.tif", tiff(6, 4, 2, o, false)),
        ];
        for (name, bytes) in cases {
            let r = decode_raster(name, &bytes).unwrap_or_else(|e| panic!("{name} orientation {o}: {e}"));
            let want = Orientation::from_exif(o as u8).unwrap();
            assert_eq!(probe_geometry(name, &bytes), Some((r.width, r.height, want)), "{name} orientation {o}");
            assert_eq!(transposes(want), (5..=8).contains(&o));
        }
    }
    // held at the cap, as the decode holds it
    let r = decode_raster("wide.png", &png(9000, 4)).unwrap();
    assert_eq!(probe_geometry("wide.png", &png(9000, 4)), Some((r.width, r.height, Orientation::NoTransforms)));
}

#[test]
fn stills_whose_decode_never_turns_are_not_probed() {
    // a YCbCr TIFF decodes unturned whatever its Orientation tag says
    let ycbcr = tiff(4, 2, 6, 6, false);
    let r = decode_raster("ycbcr.tif", &ycbcr).unwrap();
    assert_eq!((r.width, r.height), (4, 2));
    assert_eq!(probe_geometry("ycbcr.tif", &ycbcr), None);
    assert_eq!(probe_geometry("notes.txt", b"not a picture"), None);
    assert_eq!(probe_geometry("empty.jpg", &[]), None);
}

/// A file behind a [`crate::ByteReader`] that counts the bytes read.
struct Counting {
    bytes: Vec<u8>,
    read: std::sync::atomic::AtomicU64,
}

impl crate::ByteReader for Counting {
    fn len(&self) -> u64 {
        self.bytes.len() as u64
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        let at = offset as usize;
        let src = self.bytes.get(at..at + buf.len()).ok_or(std::io::ErrorKind::UnexpectedEof)?;
        buf.copy_from_slice(src);
        self.read.fetch_add(buf.len() as u64, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }
}

#[test]
fn a_large_tiff_is_probed_from_its_headers() {
    // 400 × 300 RGB with its directory at the end, as libtiff writes it: 360 kB of pixels
    let big = Counting { bytes: tiff(400, 300, 2, 6, true), read: Default::default() };
    assert_eq!(probe_geometry_reader("scan.tif", &big), Some((300, 400, Orientation::Rotate90)));
    let read = big.read.load(std::sync::atomic::Ordering::Relaxed);
    assert!(read < big.bytes.len() as u64 / 2, "read {read} of {} bytes", big.bytes.len());
    // a YCbCr one is left alone even when its directory is past the bytes read first
    let ycbcr = Counting { bytes: tiff(400, 300, 6, 6, true), read: Default::default() };
    assert_eq!(probe_geometry_reader("ycbcr.tif", &ycbcr), None);
    // a short file is probed whole
    let small = Counting { bytes: tagged(image::ImageFormat::Png, 8), read: Default::default() };
    assert_eq!(probe_geometry_reader("shot.png", &small), Some((4, 6, Orientation::Rotate270)));
}

#[test]
fn hostile_headers_probe_to_none_without_a_panic() {
    let good = crate::exif::build::into_jpeg(&halves_jpeg(24, 16), &orientation_exif(6));
    let tiff = tiff(400, 300, 2, 6, true);
    for (name, file) in [("cut.jpg", &good), ("cut.tif", &tiff)] {
        for n in (0..file.len()).step_by(97) {
            let _ = std::panic::catch_unwind(|| probe_geometry(name, &file[..n])).expect("no panic on truncation");
            let r = Counting { bytes: file[..n].to_vec(), read: Default::default() };
            let _ = std::panic::catch_unwind(|| probe_geometry_reader(name, &r)).expect("no panic on a truncated reader");
        }
        for i in (0..file.len().min(4096)).step_by(5) {
            let mut bad = file.clone();
            bad[i] ^= 0x5A;
            let _ = std::panic::catch_unwind(|| probe_geometry(name, &bad)).expect("no panic on a bit flip");
        }
    }
}
