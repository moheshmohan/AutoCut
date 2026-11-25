//! Lossy JPEG DNG (Compression 34892, DNG 1.4): 8-bit LinearRaw tiles or strips of one or three
//! components, usually with a LinearizationTable taking the 8-bit codes back to linear values.
//! `filmcraft-dng` reads the container; the baseline JPEG tiles go through the image crate's
//! decoder, which `filmcraft_media::still::dng` registers. The files are written here: the DNG
//! test writer lays out the tiles and the image crate's encoder codes them.

use filmcraft_dng::tiff::tags;
use filmcraft_dng::writer::{Compression, Layout, Options, Value, chunk_samples, write};
use filmcraft_dng::{Dng, Error, probe};
use filmcraft_media::still::dng::{decode, register_lossy_jpeg};

/// A smooth 8-bit picture (what JPEG is good at).
fn smooth(w: usize, h: usize, spp: usize) -> Vec<u16> {
    (0..w * h * spp)
        .map(|i| {
            let (x, y, s) = ((i / spp) % w, (i / spp) / w, i % spp);
            (40.0 + 80.0 * (1.0 + ((x as f32 + 3.0 * s as f32) / 9.0).sin() * ((y as f32) / 7.0).cos())) as u16
        })
        .collect()
}

/// `w` × `h` × `spp` 8-bit samples (1 or 3 a pixel) as a baseline JPEG.
fn jpeg(samples: &[u16], w: usize, h: usize, spp: usize, quality: u8) -> Vec<u8> {
    use image::ImageEncoder;
    let bytes: Vec<u8> = samples.iter().map(|&v| v.min(255) as u8).collect();
    let color = if spp == 3 { image::ExtendedColorType::Rgb8 } else { image::ExtendedColorType::L8 };
    let mut out = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality).write_image(&bytes, w as u32, h as u32, color).unwrap();
    out
}

/// A lossy JPEG DNG of `px`: its tiles or strips coded at `quality`, `jpeg_spp` components each.
fn lossy_with(px: &[u16], w: usize, h: usize, spp: usize, jpeg_spp: usize, layout: Layout, quality: u8, raw_tags: Vec<(u16, Value)>) -> Vec<u8> {
    let mut o = Options { width: w, height: h, samples_per_pixel: jpeg_spp, bits: 8, cfa: None, layout, raw_tags, ..Default::default() };
    let chunks = chunk_samples(px, &o);
    o.precoded = chunks.iter().map(|(cw, rows, s)| jpeg(s, *cw, *rows, jpeg_spp, quality)).collect();
    o.compression = Compression::Precoded { compression: 34892 };
    o.samples_per_pixel = spp;
    write(px, &o)
}

fn lossy(px: &[u16], w: usize, h: usize, spp: usize, layout: Layout, quality: u8, raw_tags: Vec<(u16, Value)>) -> Vec<u8> {
    lossy_with(px, w, h, spp, spp, layout, quality, raw_tags)
}

/// Lossy JPEG DNGs were refused ("lossy JPEG DNG"); one and three components, strips and tiles
/// decode, close to the original at quality 100.
#[test]
fn lossy_jpeg_tiles_and_strips_one_and_three_components() {
    register_lossy_jpeg();
    let (w, h) = (40, 28);
    for spp in [1, 3] {
        let px = smooth(w, h, spp);
        for layout in [Layout::Strips { rows: h }, Layout::Strips { rows: 9 }, Layout::Tiles { width: 16, height: 16 }] {
            let f = lossy(&px, w, h, spp, layout, 100, vec![]);
            let info = probe(&f).unwrap();
            assert_eq!((info.compression, info.bits, info.cfa, info.samples_per_pixel), ("lossy JPEG", 8, false, spp));
            let raw = Dng::parse(&f).unwrap().raw().unwrap();
            assert_eq!(raw.data.len(), px.len());
            // quality 100 is close to the original (colour goes through YCbCr)
            let worst = raw.data.iter().zip(&px).map(|(a, b)| (i32::from(*a) - i32::from(*b)).abs()).max().unwrap();
            assert!(worst <= 4, "spp {spp} {layout:?}: off by {worst}");
            let d = decode("lossy.dng", &f).unwrap();
            assert_eq!((d.frame.width, d.frame.height), (w as u32, h as u32));
            assert!(d.pixel_format.contains("lossy JPEG"), "{}", d.pixel_format);
        }
    }
}

#[test]
fn samples_are_the_jpeg_decoders_exactly() {
    register_lossy_jpeg();
    // one strip = one JPEG: our samples are the decoded JPEG's, in raster order
    let (w, h) = (24, 16);
    for spp in [1, 3] {
        let px = smooth(w, h, spp);
        let f = lossy(&px, w, h, spp, Layout::Strips { rows: h }, 75, vec![]);
        let want: Vec<u16> = match image::load_from_memory(&jpeg(&px, w, h, spp, 75)).unwrap() {
            image::DynamicImage::ImageLuma8(g) => g.into_raw().into_iter().map(u16::from).collect(),
            image::DynamicImage::ImageRgb8(c) => c.into_raw().into_iter().map(u16::from).collect(),
            other => panic!("{:?}", other.color()),
        };
        assert_eq!(Dng::parse(&f).unwrap().raw().unwrap().data, want, "spp {spp}");
    }
}

#[test]
fn linearisation_table_expands_the_codes() {
    register_lossy_jpeg();
    // grey 8-bit codes through a square-law table to 16 bits, as lossy DNG writers do
    let (w, h) = (16, 16);
    let px = vec![128u16; w * h];
    let table: Vec<u16> = (0..256u32).map(|v| (v * v * 65_535 / (255 * 255)) as u16).collect();
    let tags = vec![(tags::LINEARIZATION_TABLE, Value::Short(table.clone())), (tags::WHITE_LEVEL, Value::Long(vec![65_535]))];
    let f = lossy(&px, w, h, 1, Layout::Strips { rows: h }, 90, tags);
    let img = filmcraft_dng::decode(&f).unwrap();
    let code = Dng::parse(&f).unwrap().raw().unwrap().data[0];
    let want = f32::from(table[usize::from(code)]) / 65_535.0;
    assert!((127..=129).contains(&code), "{code}");
    assert!((img.rgba[0] - want).abs() < 1e-6 && (img.rgba[0] - 0.252).abs() < 0.01, "{}", img.rgba[0]);
}

#[test]
fn only_8_bit_linear_raw_and_hostile_lossy_files() {
    register_lossy_jpeg();
    let (w, h) = (16, 16);
    // CFA or 12-bit lossy JPEG is outside the specification
    let o = Options { width: w, height: h, bits: 12, raw_tags: vec![(tags::COMPRESSION, Value::Short(vec![34892]))], ..Default::default() };
    let e = filmcraft_dng::decode(&write(&vec![100; w * h], &o)).err();
    assert!(matches!(e, Some(Error::Unsupported(ref m)) if m.contains("lossy JPEG") && m.contains("CFA")), "{e:?}");
    let o = Options { width: w, height: h, bits: 12, cfa: None, raw_tags: vec![(tags::COMPRESSION, Value::Short(vec![34892]))], ..Default::default() };
    assert!(matches!(filmcraft_dng::decode(&write(&vec![100; w * h], &o)), Err(Error::Unsupported(m)) if m.contains("12-bit")));
    // a tiny file claiming a huge picture is refused before allocating it
    let mut big = lossy(&smooth(w, h, 3), w, h, 3, Layout::Strips { rows: h }, 80, vec![]);
    let ifd = u32::from_le_bytes(big[4..8].try_into().unwrap()) as usize;
    let n = u16::from_le_bytes(big[ifd..ifd + 2].try_into().unwrap()) as usize;
    for e in (0..n).map(|i| ifd + 2 + 12 * i) {
        let tag = u16::from_le_bytes(big[e..e + 2].try_into().unwrap());
        if tag == tags::IMAGE_WIDTH || tag == tags::IMAGE_LENGTH || tag == tags::ROWS_PER_STRIP {
            big[e + 8..e + 12].copy_from_slice(&60_000u32.to_le_bytes());
        }
    }
    assert!(filmcraft_dng::decode(&big).is_err());
    // a JPEG coding a larger picture than its tile is refused by the decoder's budget
    let o = Options { width: w, height: h, bits: 8, cfa: None, layout: Layout::Strips { rows: h }, ..Default::default() };
    let huge = jpeg(&vec![90; 512 * 512], 512, 512, 1, 50);
    let f = write(&vec![0; w * h], &Options { compression: Compression::Precoded { compression: 34892 }, precoded: vec![huge], ..o });
    assert!(matches!(Dng::parse(&f).and_then(|d| d.raw()), Err(Error::Malformed(_))));
    // truncations and mutations
    let f = lossy(&smooth(w, h, 3), w, h, 3, Layout::Tiles { width: 16, height: 8 }, 85, vec![]);
    for n in 0..f.len() {
        assert!(std::panic::catch_unwind(|| filmcraft_dng::decode(&f[..n]).map(|_| ())).is_ok(), "truncated to {n}");
    }
    let mut s = 4321u32;
    for i in 0..3000 {
        let mut g = f.clone();
        for _ in 0..1 + i % 4 {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let at = s as usize % g.len();
            g[at] = if i % 3 == 0 { 0xFF } else { g[at] ^ (1 << (s >> 29)) };
        }
        assert!(std::panic::catch_unwind(|| filmcraft_dng::decode(&g).map(|_| ())).is_ok(), "mutation {i}");
    }
}

#[test]
fn jpeg_components_must_match_samples_per_pixel() {
    register_lossy_jpeg();
    // a colour JPEG in a one-sample file, and a grey one in a three-sample file, is no picture
    let (w, h) = (16, 16);
    for (jpeg_spp, declared) in [(3, 1), (1, 3)] {
        let f = lossy_with(&smooth(w, h, jpeg_spp), w, h, declared, jpeg_spp, Layout::Strips { rows: h }, 90, vec![]);
        let r = Dng::parse(&f).and_then(|d| d.raw());
        assert!(matches!(r, Err(Error::Malformed(_))), "{jpeg_spp} component JPEG, SamplesPerPixel {declared}: {:?}", r.map(|r| r.data.len()));
    }
}
