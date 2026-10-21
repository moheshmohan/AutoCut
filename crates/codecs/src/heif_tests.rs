use super::*;

fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = ((body.len() + 8) as u32).to_be_bytes().to_vec();
    v.extend(kind);
    v.extend(body);
    v
}

fn full(kind: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = ((u32::from(version) << 24) | flags).to_be_bytes().to_vec();
    b.extend(body);
    bx(kind, &b)
}

fn infe(id: u16, kind: &[u8; 4]) -> Vec<u8> {
    let mut b = id.to_be_bytes().to_vec();
    b.extend(0u16.to_be_bytes());
    b.extend(kind);
    b.push(0);
    full(b"infe", 2, 0, &b)
}

/// A HEIF file shaped like a phone's: a 2 × 2 grid (16 × 12 out of four 8 × 8 tiles, in `idat`)
/// turned a quarter anticlockwise, its tiles `hvc1` items in `mdat`, and an `Exif` item describing
/// it. The tiles' data is not HEVC (it cannot be decoded); everything around it is real.
fn phone_like(exif: &[u8]) -> Vec<u8> {
    let tile_data: Vec<Vec<u8>> = (0..4u8).map(|k| vec![k; 10]).collect();
    let mut exif_item = 0u32.to_be_bytes().to_vec();
    exif_item.extend(exif);
    let grid = [0u8, 0, 1, 1, 0, 16, 0, 12];
    let build = |mdat_at: u32| {
        let ftyp = bx(b"ftyp", &[b"heic".as_slice(), &0u32.to_be_bytes(), b"mif1", b"heic"].concat());
        let mut hdlr = vec![0u8; 4];
        hdlr.extend(b"pict");
        hdlr.extend([0u8; 13]);
        let mut iinf_body = 6u16.to_be_bytes().to_vec();
        for (id, k) in [(1u16, b"hvc1"), (2, b"hvc1"), (3, b"hvc1"), (4, b"hvc1"), (10, b"grid"), (20, b"Exif")] {
            iinf_body.extend(infe(id, k));
        }
        let mut dimg = 10u16.to_be_bytes().to_vec();
        dimg.extend(4u16.to_be_bytes());
        for t in 1..=4u16 {
            dimg.extend(t.to_be_bytes());
        }
        let mut cdsc = 20u16.to_be_bytes().to_vec();
        cdsc.extend(1u16.to_be_bytes());
        cdsc.extend(10u16.to_be_bytes());
        let iref = full(b"iref", 0, 0, &[bx(b"dimg", &dimg), bx(b"cdsc", &cdsc)].concat());
        let hvcc = bx(b"hvcC", &[1u8; 23]);
        let ispe_tile = full(b"ispe", 0, 0, &[8u32.to_be_bytes(), 8u32.to_be_bytes()].concat());
        let ispe_grid = full(b"ispe", 0, 0, &[16u32.to_be_bytes(), 12u32.to_be_bytes()].concat());
        let irot = bx(b"irot", &[1]);
        let mut colr = b"nclx".to_vec();
        colr.extend([0, 1, 0, 13, 0, 6, 0x80]);
        let ipco = bx(b"ipco", &[hvcc, ispe_tile, ispe_grid, irot, bx(b"colr", &colr)].concat());
        let mut ipma = 6u32.to_be_bytes().to_vec();
        for t in 1..=4u16 {
            ipma.extend(t.to_be_bytes());
            ipma.extend([2, 0x81, 0x82]);
        }
        ipma.extend(10u16.to_be_bytes());
        ipma.extend([3, 0x83, 0x84, 0x05]);
        ipma.extend(20u16.to_be_bytes());
        ipma.push(0);
        let iprp = bx(b"iprp", &[ipco, full(b"ipma", 0, 0, &ipma)].concat());
        // iloc v1: 4-byte offsets and lengths, no base, no index
        let mut iloc = vec![0x44, 0x00];
        iloc.extend(6u16.to_be_bytes());
        let mut at = mdat_at + 8;
        for (id, len) in (1..=4u16).map(|t| (t, 10u32)) {
            iloc.extend(id.to_be_bytes());
            iloc.extend(0u16.to_be_bytes());
            iloc.extend(0u16.to_be_bytes());
            iloc.extend(1u16.to_be_bytes());
            iloc.extend(at.to_be_bytes());
            iloc.extend(len.to_be_bytes());
            at += len;
        }
        iloc.extend(10u16.to_be_bytes());
        iloc.extend(1u16.to_be_bytes()); // construction method 1: idat
        iloc.extend(0u16.to_be_bytes());
        iloc.extend(1u16.to_be_bytes());
        iloc.extend(0u32.to_be_bytes());
        iloc.extend((grid.len() as u32).to_be_bytes());
        iloc.extend(20u16.to_be_bytes());
        iloc.extend(0u16.to_be_bytes());
        iloc.extend(0u16.to_be_bytes());
        iloc.extend(1u16.to_be_bytes());
        iloc.extend(at.to_be_bytes());
        iloc.extend((exif_item.len() as u32).to_be_bytes());
        let meta = full(
            b"meta",
            0,
            0,
            &[
                full(b"hdlr", 0, 0, &hdlr),
                full(b"pitm", 0, 0, &10u16.to_be_bytes()),
                full(b"iinf", 0, 0, &iinf_body),
                iref,
                iprp,
                bx(b"idat", &grid),
                full(b"iloc", 1, 0, &iloc),
            ]
            .concat(),
        );
        let mut file = ftyp;
        file.extend(meta);
        let mdat_body: Vec<u8> = tile_data.iter().flatten().copied().chain(exif_item.iter().copied()).collect();
        (file.len() as u32, [file, bx(b"mdat", &mdat_body)].concat())
    };
    let (mdat_at, _) = build(0);
    build(mdat_at).1
}

fn pic(w: u32, h: u32) -> Pic {
    // each pixel says where it is: (x, y, 0, 255)
    let rgba = (0..h).flat_map(|y| (0..w).flat_map(move |x| [x as u8, y as u8, 0, 255])).collect();
    Pic { w, h, rgba }
}

fn at(p: &Pic, x: u32, y: u32) -> (u8, u8) {
    let i = ((y * p.w + x) * 4) as usize;
    (p.rgba[i], p.rgba[i + 1])
}

#[test]
fn brands_tell_a_photo_from_a_movie() {
    let ftyp = |major: &[u8; 4], compat: &[&[u8; 4]]| {
        let mut b = major.to_vec();
        b.extend(0u32.to_be_bytes());
        for c in compat {
            b.extend(*c);
        }
        let mut v = bx(b"ftyp", &b);
        v.extend([0u8; 8]);
        v
    };
    assert!(sniff(&ftyp(b"heic", &[b"mif1", b"heic"])));
    assert!(sniff(&ftyp(b"mif1", &[b"heic"])));
    assert!(sniff(&ftyp(b"MiHE", &[b"mif1", b"MiHB"])), "an unknown major brand with a HEIF compatible one");
    assert!(!sniff(&ftyp(b"isom", &[b"mif1"])), "a movie stays the MP4 reader's");
    assert!(!sniff(&ftyp(b"msf1", &[b"heic"])), "an image sequence plays as a movie");
    assert!(!sniff(&ftyp(b"qt  ", &[])));
    assert!(!sniff(b"\x89PNG\r\n\x1a\n........"));
    assert!(!sniff(&[]));
}

#[test]
fn a_phone_like_file_is_read_whole() {
    let tiff = filmcraft_media::exif::build::block(Some(6), Some("2023:10:12 14:03:09"), Some("+04:00"), Some((25.1, 55.2)));
    let f = phone_like(&tiff);
    let h = Heif::parse("p.heic", &f).unwrap();
    assert_eq!(h.primary, 10);
    assert_eq!(h.kind_of(10), Some(*b"grid"));
    assert_eq!(h.inputs(10), vec![1, 2, 3, 4]);
    assert_eq!(h.data(10).unwrap(), vec![0, 0, 1, 1, 0, 16, 0, 12], "the grid's descriptor, from idat");
    assert_eq!(h.data(3).unwrap(), vec![2; 10], "a tile's data, from mdat");
    assert_eq!(h.size(10), Some((16, 12)));
    assert_eq!(h.props_of(10).filter(|p| matches!(p, Prop::Irot(1))).count(), 1);
    assert_eq!((h.nclx(10, None), h.nclx(1, None)), (Some((1, 13, 6, true)), None), "the grid's colour, not its tiles'");
    assert_eq!(h.nclx(1, h.nclx(10, None)), Some((1, 13, 6, true)), "…which they inherit");
    assert!(matches!(h.props_of(1).next(), Some(Prop::Hvcc(_))));
    // when and where, from the Exif item that describes the picture
    let e = exif(&f).unwrap();
    assert_eq!(e.taken_iso().as_deref(), Some("2023-10-12T14:03:09+04:00"));
    assert!(e.gps.is_some_and(|(la, lo)| (la - 25.1).abs() < 1e-4 && (lo - 55.2).abs() < 1e-4));
    // the tiles are not HEVC: an error saying so, not a panic
    let e = decode("p.heic", &f).unwrap_err().to_string();
    assert!(e.contains("p.heic") && e.contains("HEVC"), "{e}");
}

#[test]
fn transformations_turn_mirror_and_crop() {
    // a quarter turn anticlockwise: the top-right corner comes to the top left
    let r = rotate(pic(4, 2), 1);
    assert_eq!((r.w, r.h), (2, 4));
    assert_eq!(at(&r, 0, 0), (3, 0));
    assert_eq!(at(&r, 1, 3), (0, 1));
    let r = rotate(pic(4, 2), 2);
    assert_eq!(at(&r, 0, 0), (3, 1));
    let r = rotate(pic(4, 2), 3);
    assert_eq!((r.w, r.h, at(&r, 0, 0)), (2, 4, (0, 1)));
    assert_eq!(rotate(pic(4, 2), 0).rgba, pic(4, 2).rgba);
    // mirrors: about a vertical axis (left and right swap), about a horizontal one
    assert_eq!(at(&mirror(pic(4, 2), 0), 0, 0), (3, 0));
    assert_eq!(at(&mirror(pic(4, 2), 1), 0, 0), (0, 1));
    // the clean aperture: a 4 × 2 window centred in an 8 × 6 picture
    let c = clap(pic(8, 6), [4.0, 2.0, 0.0, 0.0]);
    assert_eq!((c.w, c.h, at(&c, 0, 0)), (4, 2, (2, 2)));
    // …moved right by 2: still inside
    let c = clap(pic(8, 6), [4.0, 2.0, 2.0, 0.0]);
    assert_eq!(at(&c, 0, 0), (4, 2));
    // hostile apertures leave the picture be, or crop within it
    for bad in [[f64::NAN, 2.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [1e9, 1e9, -1e9, 1e9], [3.0, 3.0, 1e9, -1e9]] {
        let c = clap(pic(8, 6), bad);
        assert!(c.w >= 1 && c.h >= 1 && c.rgba.len() == (c.w * c.h * 4) as usize, "{bad:?}");
    }
    let empty = clap(Pic { w: 0, h: 0, rgba: Vec::new() }, [1.0, 1.0, 0.0, 0.0]);
    assert_eq!((empty.w, empty.h), (0, 0));
}

#[test]
fn damaged_files_are_errors_not_crashes() {
    let good = phone_like(&filmcraft_media::exif::build::block(None, Some("2023:10:12 14:03:09"), None, None));
    for n in 0..good.len() {
        let r = std::panic::catch_unwind(|| decode("cut.heic", &good[..n]).map(|p| p.0));
        assert!(r.expect("no panic on truncation").is_err());
        let _ = std::panic::catch_unwind(|| exif(&good[..n])).expect("no panic reading EXIF");
    }
    for i in 0..good.len() {
        for bit in [0x01u8, 0x08, 0x80] {
            let mut b = good.clone();
            b[i] ^= bit;
            let _ = std::panic::catch_unwind(|| decode("flip.heic", &b).map(|p| p.0)).expect("no panic on a bit flip");
            let _ = std::panic::catch_unwind(|| exif(&b)).expect("no panic reading EXIF");
        }
    }
    // a grid that claims to be enormous is refused before anything is decoded
    let mut huge = good.clone();
    let at = huge.windows(8).position(|w| w == [0, 0, 1, 1, 0, 16, 0, 12]).unwrap();
    huge[at + 4..at + 8].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
    let e = decode("huge.heic", &huge).unwrap_err().to_string();
    assert!(e.contains("megapixels"), "{e}");
    // …and so is a tile far larger than its share of the grid (8192 × 8 in a 16 × 12 grid of 2 × 2)
    let mut wide = good.clone();
    let at = wide.windows(12).position(|w| w == [b'i', b's', b'p', b'e', 0, 0, 0, 0, 0, 0, 0, 8]).unwrap();
    wide[at + 8..at + 12].copy_from_slice(&8192u32.to_be_bytes());
    let e = decode("wide-tile.heic", &wide).unwrap_err().to_string();
    assert!(e.contains("8192 × 8 tile"), "{e}");
}

/// A real HEIC made by macOS's `sips` (an external fixture generator, like ffmpeg elsewhere; the
/// test is skipped where it is not installed) decodes to the picture it was made from.
#[test]
fn a_heic_from_sips_decodes_to_its_picture() {
    let Ok(sips) = std::process::Command::new("sips").arg("--help").output() else {
        eprintln!("sips not installed: skipped");
        return;
    };
    if !sips.status.success() && sips.stdout.is_empty() {
        eprintln!("sips not usable: skipped");
        return;
    }
    let dir = std::env::temp_dir().join(format!("filmcraft-heif-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    // 1300 × 700: three tiles across and two down, cut at the edges
    let (w, h) = (1300u32, 700u32);
    let src =
        image::RgbImage::from_fn(w, h, |x, y| image::Rgb([(x * 255 / w) as u8, (y * 255 / h) as u8, if (x / 100 + y / 100) % 2 == 0 { 40 } else { 200 }]));
    let jpg = dir.join("src.jpg");
    image::DynamicImage::ImageRgb8(src.clone()).save_with_format(&jpg, image::ImageFormat::Jpeg).unwrap();
    let heic = dir.join("src.heic");
    let made = std::process::Command::new("sips").args(["-s", "format", "heic"]).arg(&jpg).arg("--out").arg(&heic).output().unwrap();
    if !made.status.success() || !heic.exists() {
        eprintln!("sips made no HEIC: skipped ({})", String::from_utf8_lossy(&made.stderr));
        return;
    }
    let bytes = std::fs::read(&heic).unwrap();
    assert!(sniff(&bytes));
    let (dw, dh, rgba) = decode("src.heic", &bytes).unwrap();
    assert_eq!((dw, dh), (w, h));
    let mut diff = 0u64;
    for (k, p) in src.pixels().enumerate() {
        for c in 0..3 {
            diff += u64::from(p.0[c].abs_diff(rgba[k * 4 + c]));
        }
    }
    let mean = diff as f64 / (f64::from(w * h) * 3.0);
    assert!(mean < 8.0, "mean difference {mean:.2} levels");
    // and as a media source, through the opener the engine registers
    let s = crate::open_bytes("src.heic", bytes.into()).unwrap();
    assert_eq!(s.info().kind, filmcraft_media::MediaKind::Still);
    assert_eq!(s.info().video.as_ref().map(|v| (v.width, v.height)), Some((w, h)));
    std::fs::remove_dir_all(&dir).ok();
}
