use super::*;

/// Channel `(name, pixel type code)`.
type Chan = (&'static str, u32);

fn attr(h: &mut Vec<u8>, name: &str, ty: &str, value: &[u8]) {
    for s in [name, ty] {
        h.extend_from_slice(s.as_bytes());
        h.push(0);
    }
    h.extend_from_slice(&(value.len() as u32).to_le_bytes());
    h.extend_from_slice(value);
}

fn box2i(w: [i32; 4]) -> Vec<u8> {
    w.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The test-side preprocessing for RLE / ZIP: interleave even and odd bytes, then byte deltas.
fn predict(raw: &[u8]) -> Vec<u8> {
    let mut t: Vec<u8> = raw.iter().step_by(2).chain(raw.iter().skip(1).step_by(2)).copied().collect();
    for i in (1..t.len()).rev() {
        t[i] = t[i].wrapping_sub(t[i - 1]).wrapping_add(128);
    }
    t
}

/// The test-side PXR24 packing of the `lines` of `raw` (plain layout): per line and channel, the
/// differences to the previous sample as byte planes, floats cut to their top 24 bits.
fn pxr24(raw: &[u8], chans: &[Chan], width: usize, lines: usize) -> Vec<u8> {
    let mut at = 0;
    let mut out = Vec::new();
    for _ in 0..lines {
        for (_, t) in chans {
            let size = if *t == 1 { 2 } else { 4 };
            let mut prev = 0u32;
            let mut planes = vec![Vec::new(); size];
            for x in 0..width {
                let b = &raw[at + x * size..at + (x + 1) * size];
                let v = match *t {
                    1 => u32::from(u16::from_le_bytes([b[0], b[1]])),
                    2 => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) >> 8,
                    _ => u32::from_le_bytes([b[0], b[1], b[2], b[3]]),
                };
                let d = v.wrapping_sub(prev);
                prev = v;
                let n = if *t == 2 { 3 } else { size };
                planes[0..n].iter_mut().enumerate().for_each(|(k, p)| p.push((d >> (8 * (n - 1 - k))) as u8));
            }
            at += width * size;
            out.extend(planes.concat());
        }
    }
    out
}

/// OpenEXR RLE: runs of 3+ equal bytes as runs, the rest as literals.
fn rle(t: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < t.len() {
        let run = t[i..].iter().take(128).take_while(|&&b| b == t[i]).count();
        if run >= 3 {
            out.push((run - 1) as u8);
            out.push(t[i]);
            i += run;
        } else {
            let mut j = i;
            while j < t.len() && j - i < 127 && !(j + 2 < t.len() && t[j] == t[j + 1] && t[j] == t[j + 2]) {
                j += 1;
            }
            let j = j.max(i + 1);
            out.push((-((j - i) as i32)) as u8);
            out.extend_from_slice(&t[i..j]);
            i = j;
        }
    }
    out
}

/// A zlib stream of stored blocks.
fn zlib_stored(t: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01];
    let blocks: Vec<&[u8]> = t.chunks(65535).collect();
    for (k, b) in blocks.iter().enumerate() {
        out.push(u8::from(k + 1 == blocks.len()));
        out.extend_from_slice(&(b.len() as u16).to_le_bytes());
        out.extend_from_slice(&(!(b.len() as u16)).to_le_bytes());
        out.extend_from_slice(b);
    }
    // Adler-32 (not checked by our reader; other readers do)
    let (mut a, mut b) = (1u32, 0u32);
    for &x in t {
        a = (a + u32::from(x)) % 65521;
        b = (b + a) % 65521;
    }
    out.extend_from_slice(&(b << 16 | a).to_be_bytes());
    out
}

/// Encode one sample of type `ty` (0 uint, 1 half, 2 float).
fn put(out: &mut Vec<u8>, ty: u32, v: f32) {
    match ty {
        0 => out.extend_from_slice(&(v as u32).to_le_bytes()),
        1 => out.extend_from_slice(&f16(v).to_le_bytes()),
        _ => out.extend_from_slice(&v.to_le_bytes()),
    }
}

/// f32 → half for exactly representable test values.
fn f16(v: f32) -> u16 {
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    if v == 0.0 {
        return sign;
    }
    let exp = ((b >> 23) & 0xFF) as i32 - 127 + 15;
    sign | ((exp as u16) << 10) | ((b >> 13) & 0x3FF) as u16
}

/// An EXR file: `chans` (alphabetical, as EXR stores them), data window `data`, display window
/// `display`, compression `comp`; `value(channel index, x, y)` gives each sample.
fn exr_file(chans: &[Chan], data: [i32; 4], display: [i32; 4], comp: u8, value: impl Fn(usize, i32, i32) -> f32) -> Vec<u8> {
    let mut f = MAGIC.to_vec();
    f.extend_from_slice(&2u32.to_le_bytes());
    let mut cl = Vec::new();
    for (n, t) in chans {
        cl.extend_from_slice(n.as_bytes());
        cl.push(0);
        cl.extend_from_slice(&t.to_le_bytes());
        cl.extend_from_slice(&[0; 4]);
        cl.extend_from_slice(&1u32.to_le_bytes());
        cl.extend_from_slice(&1u32.to_le_bytes());
    }
    cl.push(0);
    attr(&mut f, "channels", "chlist", &cl);
    attr(&mut f, "compression", "compression", &[comp]);
    attr(&mut f, "dataWindow", "box2i", &box2i(data));
    attr(&mut f, "displayWindow", "box2i", &box2i(display));
    attr(&mut f, "lineOrder", "lineOrder", &[0]);
    attr(&mut f, "pixelAspectRatio", "float", &1f32.to_le_bytes());
    f.push(0);
    let lpc = if comp == 3 || comp == 5 { 16 } else { 1 };
    let ys: Vec<i32> = (data[1]..=data[3]).step_by(lpc).collect();
    let table = f.len();
    f.resize(table + ys.len() * 8, 0);
    for (i, &y0) in ys.iter().enumerate() {
        let mut raw = Vec::new();
        for y in y0..=(y0 + lpc as i32 - 1).min(data[3]) {
            for (c, (_, t)) in chans.iter().enumerate() {
                for x in data[0]..=data[2] {
                    put(&mut raw, *t, value(c, x, y));
                }
            }
        }
        let packed = match comp {
            0 => raw,
            1 => rle(&predict(&raw)),
            5 => {
                let lines = (data[3] - y0 + 1).min(16) as usize;
                zlib_stored(&pxr24(&raw, chans, (data[2] - data[0] + 1) as usize, lines))
            }
            _ => zlib_stored(&predict(&raw)),
        };
        let off = f.len() as u64;
        f[table + i * 8..table + i * 8 + 8].copy_from_slice(&off.to_le_bytes());
        f.extend_from_slice(&y0.to_le_bytes());
        f.extend_from_slice(&(packed.len() as u32).to_le_bytes());
        f.extend_from_slice(&packed);
    }
    f
}

/// The plain bytes of a block: lines `y0..y0 + lines`, columns `x0..x0 + width`, every channel.
fn block_raw(chans: &[Chan], x0: i32, width: usize, y0: i32, lines: usize, value: &dyn Fn(usize, i32, i32) -> f32) -> Vec<u8> {
    let mut raw = Vec::new();
    for y in y0..y0 + lines as i32 {
        for (c, (_, t)) in chans.iter().enumerate() {
            for x in x0..x0 + width as i32 {
                put(&mut raw, *t, value(c, x, y));
            }
        }
    }
    raw
}

/// A block's bytes as compression `comp` stores them. An RLE block that does not shrink is stored
/// raw, as the format says; ZIP and PXR24 blocks are stored deflate blocks, larger than the data,
/// kept so that our reader's inflation is exercised (other readers take them to be raw).
fn pack(comp: u8, raw: Vec<u8>, chans: &[Chan], width: usize, lines: usize) -> Vec<u8> {
    match comp {
        0 => raw,
        1 => Some(rle(&predict(&raw))).filter(|p| p.len() < raw.len()).unwrap_or(raw),
        5 => zlib_stored(&pxr24(&raw, chans, width, lines)),
        _ => zlib_stored(&predict(&raw)),
    }
}

/// One part of a test file.
struct TestPart<'a> {
    name: &'a str,
    /// `scanlineimage`, `tiledimage` or a deep type (whose single chunk is junk).
    kind: &'a str,
    chans: &'a [Chan],
    data: [i32; 4],
    comp: u8,
    /// Tile size and level mode of a tiled part.
    tile: (usize, usize, u8),
    value: &'a dyn Fn(usize, i32, i32) -> f32,
}

/// The header attributes of a part (without the terminating NUL).
fn part_header(p: &TestPart, multipart: bool, chunks: usize) -> Vec<u8> {
    let mut h = Vec::new();
    let mut cl = Vec::new();
    for (n, t) in p.chans {
        cl.extend_from_slice(n.as_bytes());
        cl.push(0);
        cl.extend_from_slice(&t.to_le_bytes());
        cl.extend_from_slice(&[0; 4]);
        cl.extend_from_slice(&1u32.to_le_bytes());
        cl.extend_from_slice(&1u32.to_le_bytes());
    }
    cl.push(0);
    attr(&mut h, "channels", "chlist", &cl);
    if multipart {
        attr(&mut h, "chunkCount", "int", &(chunks as i32).to_le_bytes());
    }
    attr(&mut h, "compression", "compression", &[p.comp]);
    attr(&mut h, "dataWindow", "box2i", &box2i(p.data));
    attr(&mut h, "displayWindow", "box2i", &box2i(p.data));
    attr(&mut h, "lineOrder", "lineOrder", &[0]);
    if multipart {
        attr(&mut h, "name", "string", p.name.as_bytes());
    }
    attr(&mut h, "pixelAspectRatio", "float", &1f32.to_le_bytes());
    if p.kind == "tiledimage" {
        let mut t = (p.tile.0 as u32).to_le_bytes().to_vec();
        t.extend_from_slice(&(p.tile.1 as u32).to_le_bytes());
        t.push(p.tile.2);
        attr(&mut h, "tiles", "tiledesc", &t);
    }
    if multipart {
        attr(&mut h, "type", "string", p.kind.as_bytes());
    }
    h
}

/// A part's chunks, in table order: the coordinate fields and the stored data. A tiled part's
/// level 0 comes first (tiles row by row); a mip-mapped one then has a level 1 of junk tiles.
fn part_chunks(p: &TestPart) -> Vec<(Vec<i32>, Vec<u8>)> {
    let (w, h) = ((p.data[2] - p.data[0] + 1) as usize, (p.data[3] - p.data[1] + 1) as usize);
    let mut out = Vec::new();
    match p.kind {
        "scanlineimage" => {
            let lpc = if p.comp == 3 || p.comp == 5 { 16 } else { 1 };
            for y0 in (p.data[1]..=p.data[3]).step_by(lpc) {
                let lines = lpc.min((p.data[3] - y0 + 1) as usize);
                let raw = block_raw(p.chans, p.data[0], w, y0, lines, p.value);
                out.push((vec![y0], pack(p.comp, raw, p.chans, w, lines)));
            }
        }
        "tiledimage" => {
            let (tw, th, mode) = p.tile;
            for ty in 0..h.div_ceil(th) {
                for tx in 0..w.div_ceil(tw) {
                    let (bw, bl) = (tw.min(w - tx * tw), th.min(h - ty * th));
                    let raw = block_raw(p.chans, p.data[0] + (tx * tw) as i32, bw, p.data[1] + (ty * th) as i32, bl, p.value);
                    out.push((vec![tx as i32, ty as i32, 0, 0], pack(p.comp, raw, p.chans, bw, bl)));
                }
            }
            if mode == 1 {
                out.push((vec![0, 0, 1, 1], vec![0xEE; 7]));
            }
        }
        _ => out.push((vec![p.data[1], 0, 0, 0], vec![0xDD; 9])),
    }
    out
}

/// A file of `parts`: a single-part file (tiled or scan-line, by the version flag) when there is
/// one, else a multi-part file.
fn exr_parts(parts: &[TestPart]) -> Vec<u8> {
    let multipart = parts.len() > 1;
    let chunks: Vec<Vec<(Vec<i32>, Vec<u8>)>> = parts.iter().map(part_chunks).collect();
    let mut f = MAGIC.to_vec();
    let mut flags = 2u32;
    if multipart {
        flags |= 0x1000;
    } else if parts[0].kind == "tiledimage" {
        flags |= 0x200;
    }
    f.extend_from_slice(&flags.to_le_bytes());
    for (p, c) in parts.iter().zip(&chunks) {
        f.extend(part_header(p, multipart, c.len()));
        f.push(0);
    }
    if multipart {
        f.push(0);
    }
    let table = f.len();
    let entries: usize = chunks.iter().map(Vec::len).sum();
    f.resize(table + entries * 8, 0);
    let mut k = 0;
    for (i, part) in chunks.iter().enumerate() {
        for (fields, data) in part {
            let off = f.len() as u64;
            f[table + k * 8..table + k * 8 + 8].copy_from_slice(&off.to_le_bytes());
            k += 1;
            if multipart {
                f.extend_from_slice(&(i as i32).to_le_bytes());
            }
            for v in fields {
                f.extend_from_slice(&v.to_le_bytes());
            }
            f.extend_from_slice(&(data.len() as u32).to_le_bytes());
            f.extend_from_slice(data);
        }
    }
    f
}

fn tiled(chans: &'static [Chan], data: [i32; 4], comp: u8, tile: (usize, usize, u8), value: &'static dyn Fn(usize, i32, i32) -> f32) -> Vec<u8> {
    exr_parts(&[TestPart { name: "", kind: "tiledimage", chans, data, comp, tile, value }])
}

fn floats(d: &Decoded) -> &[f32] {
    match &d.frame.data {
        filmcraft_frame::PixelData::RgbaF32(p) => p,
        _ => panic!("EXR frames are float"),
    }
}

const RGBA_HALF: &[Chan] = &[("A", 1), ("B", 1), ("G", 1), ("R", 1)];

/// A recognisable half-exact pattern: channel c of pixel (x, y).
fn pattern(c: usize, x: i32, y: i32) -> f32 {
    match c {
        0 => [1.0, 0.5, 0.25, 0.0][((x + y) & 3) as usize],
        _ => (c as f32) * 0.125 + (x as f32) * 0.0625 + (y as f32) * 2.0,
    }
}

#[test]
fn half_floats() {
    assert_eq!(half(0x3C00), 1.0);
    assert_eq!(half(0x3800), 0.5);
    assert_eq!(half(0xC000), -2.0);
    assert_eq!(half(0x7BFF), 65504.0);
    assert_eq!(half(0x0001), 2f32.powi(-24));
    assert_eq!(half(0x03FF), 1023.0 * 2f32.powi(-24));
    assert_eq!(half(0x8000).to_bits(), (-0.0f32).to_bits());
    assert_eq!(half(0x7C00), f32::INFINITY);
    assert!(half(0x7E00).is_nan());
    for v in [0.0, 1.0, 0.5, 2.0, 0.0625, 1000.0] {
        assert_eq!(half(f16(v)), v);
    }
}

#[test]
fn rgba_half_every_compression() {
    let (w, h) = (7, 21); // two ZIP chunks, the second partial
    let want = exr_file(RGBA_HALF, [0, 0, w - 1, h - 1], [0, 0, w - 1, h - 1], 0, pattern);
    let base = decode("a.exr", &want).unwrap();
    assert_eq!((base.frame.width, base.frame.height), (7, 21));
    assert!(base.has_alpha);
    assert_eq!(base.pixel_format, "RGBA 16-bit float");
    assert_eq!(base.frame.color.transfer, Transfer::Linear);
    let px = floats(&base);
    // pixel (3, 2): R G B A
    let p = &px[(2 * 7 + 3) * 4..][..4];
    assert_eq!(p, &[pattern(3, 3, 2), pattern(2, 3, 2), pattern(1, 3, 2), pattern(0, 3, 2)]);
    for comp in [1, 2, 3, 5] {
        let f = exr_file(RGBA_HALF, [0, 0, w - 1, h - 1], [0, 0, w - 1, h - 1], comp, pattern);
        assert_ne!(f, want, "compression {comp} packs differently");
        let d = decode("a.exr", &f).unwrap_or_else(|e| panic!("compression {comp}: {e}"));
        assert_eq!(floats(&d), px, "compression {comp}");
    }
}

#[test]
fn pxr24_keeps_half_and_uint_exactly_and_floats_to_24_bits() {
    // 40 wide, 35 tall: three chunks, the last of 3 lines; one channel of every type
    let chans: &[Chan] = &[("B", 2), ("G", 1), ("R", 2), ("Z", 0)];
    let value = |c: usize, x: i32, y: i32| match c {
        0 => 1.0 / 3.0 + x as f32 * 0.37 - y as f32 * 1.5, // not exact in 24 bits
        1 => x as f32 * 0.0625 + y as f32,
        2 => 6000.7 - (x * y) as f32 * 1.25,
        _ => (x * 1000 + y) as f32,
    };
    let f = exr_file(chans, [0, 0, 39, 34], [0, 0, 39, 34], 5, value);
    let d = decode("p.exr", &f).unwrap();
    assert_eq!((d.frame.width, d.frame.height), (40, 35));
    assert_eq!(d.pixel_format, "RGB 32-bit float");
    let px = floats(&d);
    for (x, y) in [(0, 0), (1, 0), (39, 0), (0, 16), (7, 17), (39, 34), (13, 33)] {
        let p = &px[(y * 40 + x) * 4..][..4];
        let (x, y) = (x as i32, y as i32);
        let cut = |v: f32| f32::from_bits(v.to_bits() & 0xFFFF_FF00);
        assert_eq!(p, &[cut(value(2, x, y)), half(f16(value(1, x, y))), cut(value(0, x, y)), 1.0], "pixel ({x}, {y})");
    }
}

#[test]
fn damaged_pxr24_is_an_error() {
    let mut f = exr_file(RGBA_HALF, [0, 0, 9, 19], [0, 0, 9, 19], 5, pattern);
    assert!(decode("p.exr", &f).is_ok());
    // cut the file inside the last chunk
    f.truncate(f.len() - 20);
    assert!(decode("p.exr", &f).is_err());
    // a chunk whose deflate data is garbage
    let mut g = exr_file(RGBA_HALF, [0, 0, 9, 19], [0, 0, 9, 19], 5, pattern);
    let at = g.windows(2).rposition(|w| w == [0x78, 0x01]).unwrap();
    g[at] = 0;
    assert!(decode("p.exr", &g).is_err());
}

#[test]
fn luminance_only_is_grey_and_opaque() {
    let f = exr_file(&[("Y", 2)], [0, 0, 3, 1], [0, 0, 3, 1], 2, |_, x, y| x as f32 * 0.25 + y as f32 * 4.0);
    let d = decode("y.exr", &f).unwrap();
    assert!(!d.has_alpha);
    assert_eq!(d.pixel_format, "Y 32-bit float");
    assert_eq!(&floats(&d)[(4 + 2) * 4..][..4], &[4.5, 4.5, 4.5, 1.0]);
}

#[test]
fn rgb_without_alpha_and_extra_channels() {
    // an unknown channel ("Z", uint) is skipped; missing alpha is opaque
    let chans: &[Chan] = &[("B", 1), ("G", 1), ("R", 1), ("Z", 0)];
    let f = exr_file(chans, [0, 0, 1, 0], [0, 0, 1, 0], 1, |c, x, _| [0.25, 0.5, 2.0, 7.0][c] + x as f32);
    let d = decode("rgb.exr", &f).unwrap();
    assert!(!d.has_alpha);
    assert_eq!(d.pixel_format, "RGB 16-bit float");
    assert_eq!(floats(&d), &[2.0, 0.5, 0.25, 1.0, 3.0, 1.5, 1.25, 1.0]);
}

#[test]
fn data_window_is_placed_in_the_display_window() {
    // data window 2×1 at (1, 1) of a 4×3 display window starting at (-1, -1)... and one column outside
    let f = exr_file(&[("B", 2), ("G", 2), ("R", 2)], [2, 0, 3, 0], [-1, -1, 2, 1], 0, |_, x, _| x as f32);
    let d = decode("dw.exr", &f).unwrap();
    assert_eq!((d.frame.width, d.frame.height), (4, 3));
    assert!(d.has_alpha, "uncovered pixels are transparent");
    let px = floats(&d);
    // display (x, y) = data (x - 1, y - 1): data x 2 lands at display column 3; data x 3 is outside
    for (i, p) in px.as_chunks::<4>().0.iter().enumerate() {
        let want: &[f32] = if i == 4 + 3 { &[2.0, 2.0, 2.0, 1.0] } else { &[0.0; 4] };
        assert_eq!(&p[..], want, "pixel {i}");
    }
}

#[test]
fn still_source_reports_exr() {
    let f = exr_file(RGBA_HALF, [0, 0, 3, 3], [0, 0, 3, 3], 3, pattern);
    assert!(sniff(&f));
    let s = crate::open_bytes("shot.0001.exr", f.into(), &[]).unwrap();
    let v = s.info().video.clone().unwrap();
    assert_eq!(s.info().kind, MediaKind::Still);
    assert_eq!((v.width, v.height, v.codec.as_str(), v.has_alpha), (4, 4, "EXR", true));
    assert_eq!(v.pixel_format, "RGBA 16-bit float");
    let fr = s.video_frame(crate::FrameRequest::full(filmcraft_time::Tick(0))).unwrap();
    assert!(matches!(fr.data, filmcraft_frame::PixelData::RgbaF32(_)));
}

#[test]
fn unsupported_files_say_why() {
    let f = exr_file(RGBA_HALF, [0, 0, 1, 1], [0, 0, 1, 1], 0, pattern);
    // tiled without a tile description is damaged; deep data is not read
    let mut tiled = f.clone();
    tiled[5] |= 0x02;
    assert!(matches!(decode("t.exr", &tiled), Err(MediaError::Decode(m)) if m.contains("tile description")));
    let mut deep = f.clone();
    deep[5] |= 0x08;
    assert!(matches!(decode("d.exr", &deep), Err(MediaError::Unsupported(m)) if m.contains("deep")));
    let piz = exr_file(RGBA_HALF, [0, 0, 1, 1], [0, 0, 1, 1], 4, pattern);
    assert!(matches!(decode("p.exr", &piz), Err(MediaError::Unsupported(m)) if m.contains("PIZ")));
    let b44 = exr_file(RGBA_HALF, [0, 0, 1, 1], [0, 0, 1, 1], 6, pattern);
    assert!(matches!(decode("b.exr", &b44), Err(MediaError::Unsupported(m)) if m.contains("B44")));
    let dwab = exr_file(RGBA_HALF, [0, 0, 1, 1], [0, 0, 1, 1], 9, pattern);
    assert!(matches!(decode("d.exr", &dwab), Err(MediaError::Unsupported(m)) if m.contains("DWAB")));
    let ry = exr_file(&[("BY", 1), ("RY", 1), ("Y", 1)], [0, 0, 1, 1], [0, 0, 1, 1], 0, pattern);
    assert!(matches!(decode("c.exr", &ry), Err(MediaError::Unsupported(_))));
    // alpha and depth make no picture (a lone depth channel does: see below)
    let data = exr_file(&[("A", 1), ("Z", 2)], [0, 0, 1, 1], [0, 0, 1, 1], 0, pattern);
    assert!(matches!(decode("z.exr", &data), Err(MediaError::Unsupported(m)) if m.contains("no R, G, B or Y")));
}

/// A renderer's multilayer file names every channel by its layer (`ViewLayer.Combined.R`): it has
/// no main picture and was refused ("no R, G, B or Y channel"); its richest colour layer shows.
#[test]
fn a_file_of_named_layers_shows_its_richest_colour_layer() {
    let chans: &[Chan] = &[
        ("ViewLayer.AO.B", 1),
        ("ViewLayer.AO.G", 1),
        ("ViewLayer.AO.R", 1),
        ("ViewLayer.Combined.A", 1),
        ("ViewLayer.Combined.B", 1),
        ("ViewLayer.Combined.G", 1),
        ("ViewLayer.Combined.R", 1),
        ("ViewLayer.Depth.Z", 2),
    ];
    // AO is 9 everywhere; Combined is (0.25, 0.5, 2, alpha 0.5)
    let value = |c: usize, _: i32, _: i32| [9.0, 9.0, 9.0, 0.5, 2.0, 0.5, 0.25, 100.0][c];
    let f = exr_file(chans, [0, 0, 2, 1], [0, 0, 2, 1], 3, value);
    let d = decode("multilayer.exr", &f).unwrap();
    assert_eq!(d.pixel_format, "RGBA 16-bit float, layer ViewLayer.Combined");
    assert!(d.has_alpha);
    assert!(floats(&d).as_chunks::<4>().0.iter().all(|p| p == &[0.25, 0.5, 2.0, 0.5]));
}

/// The unprefixed colour is the main picture even beside a richer named layer; alpha is never
/// taken from another layer.
#[test]
fn the_main_picture_wins_and_layers_are_not_mixed() {
    let chans: &[Chan] = &[("B", 1), ("G", 1), ("R", 1), ("beauty.A", 1), ("beauty.B", 1), ("beauty.G", 1), ("beauty.R", 1)];
    let f = exr_file(chans, [0, 0, 1, 0], [0, 0, 1, 0], 0, |c, _, _| [1.0, 2.0, 3.0, 0.5, 5.0, 6.0, 7.0][c]);
    let d = decode("main.exr", &f).unwrap();
    assert_eq!((d.pixel_format.as_str(), d.has_alpha), ("RGB 16-bit float", false));
    assert_eq!(&floats(&d)[..4], &[3.0, 2.0, 1.0, 1.0]);
    // bare alpha beside a named colour layer: the layer's colour, opaque
    let chans: &[Chan] = &[("A", 1), ("diffuse.B", 1), ("diffuse.G", 1), ("diffuse.R", 1)];
    let f = exr_file(chans, [0, 0, 1, 0], [0, 0, 1, 0], 0, |c, _, _| [0.5, 1.0, 2.0, 4.0][c]);
    let d = decode("diffuse.exr", &f).unwrap();
    assert_eq!((d.pixel_format.as_str(), d.has_alpha), ("RGB 16-bit float, layer diffuse", false));
    assert_eq!(&floats(&d)[..4], &[4.0, 2.0, 1.0, 1.0]);
}

/// A file of one channel (a depth or mask pass) was refused; it shows grey, and Y with alpha is
/// luminance with alpha.
#[test]
fn a_lone_channel_is_grey_and_luminance_keeps_its_alpha() {
    let f = exr_file(&[("Z", 2)], [0, 0, 1, 0], [0, 0, 1, 0], 0, |_, x, _| 10.0 + x as f32);
    let d = decode("z.exr", &f).unwrap();
    assert_eq!(d.pixel_format, "Y 32-bit float");
    assert_eq!(floats(&d), &[10.0, 10.0, 10.0, 1.0, 11.0, 11.0, 11.0, 1.0]);
    let f = exr_file(&[("A", 1), ("Y", 1)], [0, 0, 0, 0], [0, 0, 0, 0], 0, |c, _, _| [0.5, 2.0][c]);
    let d = decode("ya.exr", &f).unwrap();
    assert_eq!((d.pixel_format.as_str(), d.has_alpha), ("YA 16-bit float", true));
    assert_eq!(floats(&d), &[2.0, 2.0, 2.0, 0.5]);
}

#[test]
fn hostile_files_are_errors_not_panics() {
    // windows too large to allocate, or inverted
    for data in [[0, 0, i32::MAX, i32::MAX], [i32::MIN, 0, i32::MAX, 0], [5, 5, 0, 0], [0, 0, 1 << 14, 1 << 14]] {
        let f = exr_file(&[("R", 1)], [0, 0, 0, 0], [0, 0, 0, 0], 0, pattern);
        // patch the data window in the header
        let at = f.windows(10).position(|w| w == b"dataWindow").unwrap() + 10 + 1 + 6 + 4;
        let mut g = f.clone();
        g[at..at + 16].copy_from_slice(&box2i(data));
        assert!(decode("h.exr", &g).is_err(), "{data:?}");
    }
    // every truncation and many single-byte corruptions of real files
    for comp in [0, 1, 2, 3] {
        let f = exr_file(RGBA_HALF, [0, 0, 5, 17], [0, 0, 5, 17], comp, pattern);
        for n in 0..f.len() {
            assert!(decode("h.exr", &f[..n]).is_err(), "compression {comp}, {n} bytes");
        }
        for n in 0..f.len() {
            for x in [0x01, 0x80, 0xFF] {
                let mut g = f.clone();
                g[n] ^= x;
                let _ = decode("h.exr", &g);
            }
        }
    }
}

/// A tiny header naming many channels on a very wide line used to size each chunk buffer at
/// hundreds of GB (`Vec::with_capacity` aborts the process); it is refused instead.
#[test]
fn huge_scan_lines_are_refused_before_allocating() {
    let names: Vec<String> = (0..MAX_CHANNELS - 1).map(|i| format!("A{i:04}")).collect();
    let mut chans: Vec<Chan> = names.iter().map(|n| (&*n.clone().leak(), 2)).collect();
    chans.push(("R", 2));
    let f = exr_file(&chans, [0, 0, 0, 0], [0, 0, 0, 0], 3, pattern);
    let at = f.windows(10).position(|w| w == b"dataWindow").unwrap() + 10 + 1 + 6 + 4;
    let mut g = f.clone();
    g[at..at + 16].copy_from_slice(&box2i([0, 0, (1 << 26) - 1, 0]));
    assert!(decode("h.exr", &g).is_err());
}

#[test]
fn uncompressed_chunk_of_the_wrong_size_fails_before_the_picture_is_allocated() {
    let (w, h) = (7, 3);
    let mut f = exr_file(RGBA_HALF, [0, 0, w - 1, h - 1], [0, 0, w - 1, h - 1], 0, pattern);
    // the offset table follows the header; shorten chunk 0's stored length by one byte
    let table = f.len() - (h as usize) * (4 + 4 + 7 * 4 * 2) - (h as usize) * 8;
    let off = u64::from_le_bytes(f[table..table + 8].try_into().unwrap()) as usize;
    let size = u32::from_le_bytes(f[off + 4..off + 8].try_into().unwrap());
    f[off + 4..off + 8].copy_from_slice(&(size - 1).to_le_bytes());
    let e = decode("a.exr", &f).err().expect("a wrong-size chunk is an error").to_string();
    assert!(e.contains("chunk 0 has the wrong size"), "{e}");
}

/// A header can describe a large picture over compressed chunks of a few bytes. The picture used
/// to be allocated, and without an alpha channel written opaque, before the first chunk failed to
/// expand: a file of about 200 bytes (one half channel, 2^22 × 16 pixels, ZIP) committed 1 GiB. A
/// chunk too short to expand to its lines is now refused before the picture exists.
#[test]
fn compressed_chunks_too_short_for_their_lines_fail_before_the_picture_is_allocated() {
    for comp in [1, 3, 5] {
        let mut f = exr_file(&[("R", 1)], [0, 0, 0, 15], [0, 0, 0, 15], comp, pattern);
        for name in [&b"dataWindow"[..], b"displayWindow"] {
            let at = f.windows(name.len()).position(|w| w == name).unwrap() + name.len() + 1 + 6 + 4;
            f[at..at + 16].copy_from_slice(&box2i([0, 0, (1 << 16) - 1, 15]));
        }
        let e = decode("a.exr", &f).err().expect("a chunk too short for its lines is an error").to_string();
        assert!(e.contains("chunk 0 is too short for its"), "compression {comp}: {e}");
    }
}

/// Tiled files were refused. Every compression, edge tiles cut by the data window, a data window
/// off the origin: the same pixels as the scan-line file of the same values.
#[test]
fn tiled_files_decode_like_scan_lines() {
    for (data, tile) in [([0, 0, 12, 9], (4, 3)), ([3, -2, 9, 4], (2, 2)), ([0, 0, 4, 4], (64, 64)), ([0, 0, 6, 0], (1, 1))] {
        let base = decode("s.exr", &exr_file(RGBA_HALF, data, data, 0, pattern)).unwrap();
        for comp in [0, 1, 2, 3, 5] {
            let f = tiled(RGBA_HALF, data, comp, (tile.0, tile.1, 0), &pattern);
            let d = decode("t.exr", &f).unwrap_or_else(|e| panic!("{data:?} {tile:?} compression {comp}: {e}"));
            assert_eq!((d.frame.width, d.frame.height), (base.frame.width, base.frame.height));
            assert_eq!(floats(&d), floats(&base), "{data:?} {tile:?} compression {comp}");
            assert_eq!(d.pixel_format, "RGBA 16-bit float");
        }
    }
}

/// A mip-mapped file shows its full-resolution level; the smaller levels are not read.
#[test]
fn mip_mapped_tiles_show_the_full_resolution_level() {
    let data = [0, 0, 9, 6];
    let base = decode("s.exr", &exr_file(RGBA_HALF, data, data, 0, pattern)).unwrap();
    let f = tiled(RGBA_HALF, data, 3, (4, 4, 1), &pattern);
    assert_eq!(floats(&decode("m.exr", &f).unwrap()), floats(&base));
}

#[test]
fn damaged_tiled_files_are_errors() {
    let f = tiled(RGBA_HALF, [0, 0, 7, 7], 0, (4, 4, 0), &pattern);
    assert!(decode("t.exr", &f).is_ok());
    let header_end = f.windows(9).position(|w| w == b"tiledesc\0").unwrap() + 9 + 4 + 9 + 1;
    let entry = |f: &[u8], i: usize| u64::from_le_bytes(f[header_end + i * 8..header_end + i * 8 + 8].try_into().unwrap()) as usize;
    // tile 1 stored twice (tile 0's table entry pointing at it), or of another level
    let mut twice = f.clone();
    let one = entry(&f, 1) as u64;
    twice[header_end..header_end + 8].copy_from_slice(&one.to_le_bytes());
    assert!(decode("t.exr", &twice).err().unwrap().to_string().contains("stored twice"));
    let mut level = f.clone();
    let at = entry(&f, 2) + 8;
    level[at..at + 4].copy_from_slice(&1i32.to_le_bytes());
    assert!(decode("t.exr", &level).err().unwrap().to_string().contains("level (1, 0)"));
    // a tile outside the grid, and a tile size of 0
    let mut outside = f.clone();
    let at = entry(&f, 3);
    outside[at..at + 4].copy_from_slice(&9i32.to_le_bytes());
    assert!(decode("t.exr", &outside).err().unwrap().to_string().contains("of 2 × 2"));
    let mut zero = f.clone();
    let at = header_end - 10;
    zero[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
    assert!(decode("t.exr", &zero).is_err());
    // every truncation is an error, and corruptions never panic
    for n in 0..f.len() {
        assert!(decode("t.exr", &f[..n]).is_err(), "{n} bytes");
    }
    for comp in [0, 1, 3, 5] {
        let f = tiled(RGBA_HALF, [0, 0, 5, 6], comp, (3, 2, 1), &pattern);
        for n in 0..f.len() {
            for x in [0x01, 0x80, 0xFF] {
                let mut g = f.clone();
                g[n] ^= x;
                let _ = decode("t.exr", &g);
            }
        }
    }
}

/// A huge tile of a small picture is cut to the data window; a picture of many channels in one
/// tile larger than the chunk limit is refused before anything is allocated.
#[test]
fn tile_sizes_are_bounded_by_the_data_window() {
    let f = tiled(RGBA_HALF, [0, 0, 2, 2], 0, (1 << 30, 1 << 30, 0), &pattern);
    assert_eq!(floats(&decode("t.exr", &f).unwrap()), floats(&decode("s.exr", &exr_file(RGBA_HALF, [0, 0, 2, 2], [0, 0, 2, 2], 0, pattern)).unwrap()));
    let names: Vec<String> = (0..MAX_CHANNELS - 3).map(|i| format!("C{i:04}")).collect();
    let mut chans: Vec<Chan> = vec![("B", 2), ("G", 2), ("R", 2)];
    chans.extend(names.iter().map(|n| -> Chan { (n.clone().leak(), 2) }));
    let chans: &'static [Chan] = chans.leak();
    let mut f = tiled(chans, [0, 0, 0, 0], 3, (4096, 4096, 0), &pattern);
    let at = f.windows(10).position(|w| w == b"dataWindow").unwrap() + 10 + 1 + 6 + 4;
    f[at..at + 16].copy_from_slice(&box2i([0, 0, 4095, 4095]));
    assert!(matches!(decode("t.exr", &f), Err(MediaError::Unsupported(m)) if m.contains("tiles too large")));
}

/// Multi-part files were refused. The main picture is the first part with bare R, G and B (here a
/// tiled part after a depth part); the other parts' chunks are skipped.
#[test]
fn multi_part_files_show_the_part_with_the_main_picture() {
    let depth: &dyn Fn(usize, i32, i32) -> f32 = &|_, x, y| (x + y) as f32;
    let beauty: &dyn Fn(usize, i32, i32) -> f32 = &pattern;
    let other: &dyn Fn(usize, i32, i32) -> f32 = &|_, _, _| 7.0;
    let data = [0, 0, 8, 5];
    let parts = [
        TestPart { name: "depth", kind: "scanlineimage", chans: &[("Z", 2)], data, comp: 3, tile: (0, 0, 0), value: depth },
        TestPart { name: "beauty", kind: "tiledimage", chans: RGBA_HALF, data, comp: 2, tile: (4, 4, 0), value: beauty },
        TestPart { name: "other", kind: "scanlineimage", chans: &[("B", 1), ("G", 1), ("R", 1)], data, comp: 0, tile: (0, 0, 0), value: other },
    ];
    let f = exr_parts(&parts);
    let d = decode("multi.exr", &f).unwrap();
    assert_eq!(d.pixel_format, "RGBA 16-bit float, part beauty");
    assert_eq!(floats(&d), floats(&decode("s.exr", &exr_file(RGBA_HALF, data, data, 0, pattern)).unwrap()));
    // a scan-line main picture after a deep part, which is never read
    let parts = [
        TestPart { name: "deep", kind: "deepscanline", chans: &[("A", 1), ("Z", 2)], data, comp: 0, tile: (0, 0, 0), value: depth },
        TestPart { name: "rgb", kind: "scanlineimage", chans: &[("B", 1), ("G", 1), ("R", 1)], data, comp: 1, tile: (0, 0, 0), value: other },
    ];
    let d = decode("deep.exr", &exr_parts(&parts)).unwrap();
    assert_eq!(d.pixel_format, "RGB 16-bit float, part rgb");
    assert!(floats(&d).as_chunks::<4>().0.iter().all(|p| p == &[7.0, 7.0, 7.0, 1.0]));
    // named layers in separate parts: the richest
    let parts = [
        TestPart { name: "ViewLayer.Depth", kind: "scanlineimage", chans: &[("ViewLayer.Depth.Z", 2)], data, comp: 0, tile: (0, 0, 0), value: depth },
        TestPart {
            name: "ViewLayer.Combined",
            kind: "scanlineimage",
            chans: &[("ViewLayer.Combined.B", 1), ("ViewLayer.Combined.G", 1), ("ViewLayer.Combined.R", 1)],
            data,
            comp: 0,
            tile: (0, 0, 0),
            value: other,
        },
    ];
    let d = decode("layers.exr", &exr_parts(&parts)).unwrap();
    assert_eq!(d.pixel_format, "RGB 16-bit float, part ViewLayer.Combined, layer ViewLayer.Combined");
    // deep parts only: refused, saying why
    let parts = [
        TestPart { name: "a", kind: "deepscanline", chans: &[("B", 1), ("G", 1), ("R", 1)], data, comp: 0, tile: (0, 0, 0), value: depth },
        TestPart { name: "b", kind: "deeptile", chans: &[("Y", 1)], data, comp: 0, tile: (0, 0, 0), value: depth },
    ];
    assert!(matches!(decode("d.exr", &exr_parts(&parts)), Err(MediaError::Unsupported(m)) if m.contains("deep")));
}

#[test]
fn damaged_multi_part_files_are_errors() {
    let depth: &dyn Fn(usize, i32, i32) -> f32 = &|_, x, y| (x + y) as f32;
    let data = [0, 0, 3, 3];
    let parts = [
        TestPart { name: "z", kind: "scanlineimage", chans: &[("Z", 2)], data, comp: 0, tile: (0, 0, 0), value: depth },
        TestPart { name: "rgba", kind: "tiledimage", chans: RGBA_HALF, data, comp: 1, tile: (2, 2, 0), value: &pattern },
    ];
    let f = exr_parts(&parts);
    assert!(decode("m.exr", &f).is_ok());
    // the part before the picture without a chunk count: its table cannot be skipped
    let mut g = f.clone();
    let at = g.windows(10).position(|w| w == b"chunkCount").unwrap();
    g[at] = b'x';
    assert!(decode("m.exr", &g).err().unwrap().to_string().contains("part 0 has no chunkCount"));
    // a chunk count past the end of the file
    let mut g = f.clone();
    let at = f.windows(10).position(|w| w == b"chunkCount").unwrap() + 10 + 4 + 4;
    g[at..at + 4].copy_from_slice(&i32::MAX.to_le_bytes());
    assert!(decode("m.exr", &g).is_err());
    // a chunk of the other part where the picture's should be: part 0 has four scan lines, so
    // part 1's table starts at entry 4
    let tables = 8 + part_header(&parts[0], true, 4).len() + 1 + part_header(&parts[1], true, 4).len() + 1 + 1;
    let mut g = f.clone();
    let first_z = g[tables..tables + 8].to_vec();
    g[tables + 4 * 8..tables + 5 * 8].copy_from_slice(&first_z);
    assert!(decode("m.exr", &g).err().unwrap().to_string().contains("not part 1's"));
    // too many parts
    let mut many = MAGIC.to_vec();
    many.extend_from_slice(&(2u32 | 0x1000).to_le_bytes());
    for _ in 0..=MAX_PARTS {
        many.extend(part_header(&parts[0], true, 1));
        many.push(0);
    }
    many.push(0);
    assert!(matches!(decode("many.exr", &many), Err(MediaError::Unsupported(m)) if m.contains("parts")));
    // every truncation is an error, and corruptions never panic
    for n in 0..f.len() {
        assert!(decode("m.exr", &f[..n]).is_err(), "{n} bytes");
    }
    for n in 0..f.len() {
        for x in [0x01, 0x80, 0xFF] {
            let mut g = f.clone();
            g[n] ^= x;
            let _ = decode("m.exr", &g);
        }
    }
}

/// ffmpeg (an external test oracle) reads the tiled and multi-part files these tests write as we
/// do: the test writer follows the file layout as another reader understands it. The test writer
/// packs ZIP chunks as stored deflate blocks, larger than the raw data, which other readers take
/// to be raw (the format stores a chunk raw when compression would not shrink it): only
/// uncompressed and RLE chunks of data that shrinks are compared. ffmpeg reads a multi-part file's
/// first part only, and only when it is scan lines (asked for a later part or a tiled one it gives
/// black), so multi-part files are compared on a first scan-line part; later and tiled parts are
/// checked by [`multi_part_files_show_the_part_with_the_main_picture`].
#[test]
fn tiled_and_multi_part_files_match_ffmpeg() {
    let Some(ff) = filmcraft_testkit::oracle::ffmpeg_or_skip("tiled and multi-part OpenEXR") else { return };
    let dir = filmcraft_testkit::fixtures::fixtures_dir("media/exr_parts");
    std::fs::create_dir_all(&dir).unwrap();
    let rgba: &'static [Chan] = &[("A", 2), ("B", 2), ("G", 2), ("R", 2)];
    let value: &dyn Fn(usize, i32, i32) -> f32 =
        &|c, x, y| if c == 0 { ((x + 2 * y) % 5) as f32 / 4.0 } else { c as f32 * 0.3 + x as f32 * 0.07 - y as f32 * 0.11 };
    // runs of equal samples, so RLE shrinks every block
    let flat: &dyn Fn(usize, i32, i32) -> f32 = &|c, x, y| if c == 0 { 1.0 } else { c as f32 * 0.25 + (x / 6 + y / 4) as f32 };
    let data = [0, 0, 18, 12];
    let depth: &dyn Fn(usize, i32, i32) -> f32 = &|_, x, _| x as f32;
    let one = |kind, comp, tile, value| exr_parts(&[TestPart { name: "", kind, chans: rgba, data, comp, tile, value }]);
    let cases: Vec<(&str, Vec<u8>, Option<u32>)> = vec![
        ("lines_none.exr", one("scanlineimage", 0, (0, 0, 0), value), None),
        ("tiled_none.exr", one("tiledimage", 0, (8, 5, 0), value), None),
        ("tiled_big_tile.exr", one("tiledimage", 0, (32, 32, 0), value), None),
        ("tiled_rle.exr", one("tiledimage", 1, (6, 4, 0), flat), None),
        (
            "two_parts_rle.exr",
            exr_parts(&[
                TestPart { name: "rgba", kind: "scanlineimage", chans: rgba, data, comp: 1, tile: (0, 0, 0), value: flat },
                TestPart { name: "depth", kind: "tiledimage", chans: &[("Z", 2)], data, comp: 0, tile: (4, 4, 0), value: depth },
            ]),
            Some(0),
        ),
    ];
    // the RLE chunks are smaller than their raw data, so every reader decompresses them
    let raw = block_raw(rgba, 0, 6, 0, 4, flat);
    assert!(pack(1, raw.clone(), rgba, 6, 4).len() < raw.len());
    for (file, bytes, part) in cases {
        let path = dir.join(file);
        std::fs::write(&path, &bytes).unwrap();
        let mut cmd = std::process::Command::new(&ff);
        cmd.args(["-v", "error"]);
        if let Some(p) = part {
            cmd.args(["-part", &p.to_string()]);
        }
        let out = cmd.arg("-i").arg(&path).args(["-f", "rawvideo", "-pix_fmt", "gbrapf32le", "-"]).output().unwrap();
        if !out.status.success() {
            eprintln!("SKIPPED: ffmpeg cannot read {file}: {}", String::from_utf8_lossy(&out.stderr));
            continue;
        }
        let v: Vec<f32> = out.stdout.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect();
        let n = 19 * 13;
        assert_eq!(v.len(), n * 4, "{file}");
        let theirs: Vec<f32> = (0..n).flat_map(|i| [v[2 * n + i], v[i], v[n + i], v[3 * n + i]]).collect();
        let d = decode(file, &bytes).unwrap_or_else(|e| panic!("{file}: {e}"));
        assert_eq!(floats(&d), theirs.as_slice(), "{file}");
    }
}
