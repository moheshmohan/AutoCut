//! HEIF / HEIC stills: the photos iPhones (and many Android phones) take, as media sources.
//!
//! Clean-room from ISO/IEC 23008-12 (HEIF: the `meta` box's items, their locations, properties
//! and references, the image grid) and ISO/IEC 14496-12 (the box structure), decoding each HEVC
//! picture with our own decoder (`filmcraft-hevc`). The primary item is shown:
//!
//! - an `hvc1` item: one HEVC picture (its `hvcC` property configures the decoder);
//! - a `grid` item: `rows × columns` HEVC tiles (its `dimg` references, in raster order) laid
//!   edge to edge and cut to the grid's output size — an iPhone photo is 48 tiles of 512 × 512;
//! - an `iden` item: its one input, with its own transformations;
//!
//! then its transformative properties in the order they are associated: `clap` (clean aperture,
//! a crop), `irot` (anticlockwise quarter turns) and `imir` (a mirror about a vertical or
//! horizontal axis). Colour comes from the item's `colr nclx` (the HEVC stream's own VUI
//! otherwise). The picture is scaled down to [`filmcraft_media::still::MAX_STILL_SIDE`] like any
//! still. Auxiliary images (alpha, depth), thumbnails and AVIF (`av01`) items are not decoded.
//!
//! Every number in the file is hostile: boxes are walked with bounds checks and depth and count
//! limits, the file is read whole only up to [`MAX_FILE`], a picture larger than [`MAX_PIXELS`] is
//! refused before anything is decoded, and a tile that does not decode is an error, never a panic.
//!
//! [`exif`] reads the file's `Exif` item (when it was taken and where; the picture is already
//! upright by `irot`/`imir`, which HEIF says to use instead of the EXIF orientation).

use std::collections::HashMap;
use std::sync::Arc;

use filmcraft_media::{MediaError, SharedSource};
use rayon::prelude::*;

/// Largest HEIF file read (a 48-megapixel HEIC is about 5–15 MB).
pub const MAX_FILE: u64 = 256 << 20;
/// Most pixels a picture may have as decoded (before it is scaled to the still cap): 128 MP, as
/// RGBA 512 MiB.
pub const MAX_PIXELS: u64 = 128 << 20;
/// Most items, properties, references and boxes read, and tiles in a grid.
const MAX_ITEMS: usize = 65_536;
const MAX_BOXES: usize = 65_536;
const MAX_TILES: usize = 16_384;
/// Deepest nesting of derived images (a grid of identities of …).
const MAX_DEPTH: usize = 4;

type Result<T> = std::result::Result<T, MediaError>;

fn bad(name: &str, why: impl std::fmt::Display) -> MediaError {
    MediaError::Decode(format!("{name}: HEIF: {why}"))
}

fn unsupported(name: &str, why: impl std::fmt::Display) -> MediaError {
    MediaError::Unsupported(format!("{name}: HEIF: {why}"))
}

/// Brands of HEIF still images (ISO/IEC 23008-12 Annex B).
const STILL_BRANDS: [&[u8; 4]; 7] = [b"heic", b"heix", b"heim", b"heis", b"mif1", b"mif2", b"avif"];
/// Brands of movies (and image sequences), which stay the MP4 reader's.
const MOVIE_BRANDS: [&[u8; 4]; 10] = [b"isom", b"iso2", b"mp41", b"mp42", b"qt  ", b"msf1", b"hevc", b"avc1", b"M4V ", b"avis"];

/// Whether `head` (a file's first bytes) is a HEIF still: an `ftyp` whose major brand is a still
/// image's, or whose compatible brands are while its major brand is not a movie's.
pub fn sniff(head: &[u8]) -> bool {
    if head.len() < 16 || head.get(4..8) != Some(b"ftyp") {
        return false;
    }
    let size = u32::from_be_bytes([head[0], head[1], head[2], head[3]]) as usize;
    let Some(major) = head.get(8..12) else { return false };
    let is = |list: &[&[u8; 4]], b: &[u8]| list.iter().any(|x| x.as_slice() == b);
    if is(&STILL_BRANDS, major) {
        return true;
    }
    if is(&MOVIE_BRANDS, major) {
        return false;
    }
    let end = size.clamp(16, head.len());
    head.get(16..end).is_some_and(|c| c.as_chunks::<4>().0.iter().any(|b| is(&[b"heic", b"heix", b"mif1"], b)))
}

/// The byte opener (`filmcraft_media::open_bytes`).
pub fn opener(name: &str, bytes: Arc<[u8]>) -> Option<std::result::Result<SharedSource, MediaError>> {
    if !sniff(&bytes) {
        return None;
    }
    Some(still(name, &bytes))
}

/// The reader opener: a HEIF file is read whole (it is one picture), up to [`MAX_FILE`].
pub fn reader_opener(name: &str, head: &[u8], reader: &filmcraft_media::SharedReader) -> Option<std::result::Result<SharedSource, MediaError>> {
    if !sniff(head) {
        return None;
    }
    let len = reader.len();
    if len > MAX_FILE {
        return Some(Err(unsupported(name, format!("the file is larger than {} MiB", MAX_FILE >> 20))));
    }
    let all = match filmcraft_media::reader::read_range(&**reader, 0, usize::try_from(len).unwrap_or(usize::MAX)) {
        Ok(b) => b,
        Err(e) => return Some(Err(MediaError::Io(format!("{name}: {e}")))),
    };
    Some(still(name, &all))
}

fn still(name: &str, bytes: &[u8]) -> std::result::Result<SharedSource, MediaError> {
    let (w, h, rgba) = decode(name, bytes)?;
    Ok(Arc::new(filmcraft_media::still::StillSource::from_rgba(name, w, h, rgba, false, bytes.len() as u64)))
}

/// The primary picture of a HEIF file as straight RGBA8 `(width, height, pixels)`, upright and
/// within the still cap.
pub fn decode(name: &str, bytes: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let file = Heif::parse(name, bytes)?;
    let pic = file.picture(file.primary, None, 0)?;
    let (w, h, rgba) = filmcraft_media::still::cap_rgba(pic.w, pic.h, pic.rgba);
    if rgba.is_empty() {
        return Err(bad(name, "the picture could not be scaled"));
    }
    Ok((w, h, rgba))
}

/// The EXIF of a HEIF file's `Exif` item (the one describing the primary item, else the first).
pub fn exif(bytes: &[u8]) -> Option<filmcraft_media::exif::Exif> {
    let file = Heif::parse("", bytes).ok()?;
    let describes = |id: u32| file.refs.iter().any(|r| &r.kind == b"cdsc" && r.from == id && r.to.contains(&file.primary));
    let item = file.items.iter().filter(|i| &i.kind == b"Exif").max_by_key(|i| describes(i.id))?;
    let data = file.data(item.id).ok()?;
    // exif_tiff_header_offset, then the block (some writers put "Exif\0\0" before the header)
    let skip = u32::from_be_bytes(data.get(0..4)?.try_into().ok()?) as usize;
    let rest = data.get(4..)?;
    let tiff = rest.get(skip..).filter(|t| t.starts_with(b"II*\0") || t.starts_with(b"MM\0*")).or_else(|| {
        let at = rest.windows(4).take(64).position(|w| w == b"II*\0" || w == b"MM\0*")?;
        rest.get(at..)
    })?;
    filmcraft_media::exif::parse(tiff)
}

/// A bounds-checked big-endian reader.
struct R<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> R<'a> {
    fn new(b: &'a [u8]) -> Self {
        Self { b, at: 0 }
    }
    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.b.get(self.at..self.at.checked_add(n)?)?;
        self.at += n;
        Some(s)
    }
    fn u8(&mut self) -> Option<u8> {
        self.take(1).map(|s| s[0])
    }
    fn u16(&mut self) -> Option<u16> {
        self.take(2).map(|s| u16::from_be_bytes([s[0], s[1]]))
    }
    fn u32(&mut self) -> Option<u32> {
        self.take(4).map(|s| u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn u64(&mut self) -> Option<u64> {
        let s = self.take(8)?;
        Some(u64::from_be_bytes([s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7]]))
    }
    /// An unsigned integer of `n` bytes (0, 4 or 8; 0 reads as 0).
    fn uint(&mut self, n: u8) -> Option<u64> {
        match n {
            0 => Some(0),
            4 => self.u32().map(u64::from),
            8 => self.u64(),
            _ => None,
        }
    }
    fn fourcc(&mut self) -> Option<[u8; 4]> {
        self.take(4).map(|s| [s[0], s[1], s[2], s[3]])
    }
    /// A full box's version and flags.
    fn full(&mut self) -> Option<(u8, u32)> {
        let v = self.u32()?;
        Some(((v >> 24) as u8, v & 0x00FF_FFFF))
    }
    /// A null-terminated string (skipped).
    fn skip_cstr(&mut self) {
        let rest = self.b.get(self.at..).unwrap_or_default();
        self.at += rest.iter().position(|c| *c == 0).map_or(rest.len(), |p| p + 1);
    }
}

/// The boxes in `b`, in order: `(type, body)`. Stops at the first box whose size does not fit.
fn boxes(b: &[u8]) -> Vec<([u8; 4], &[u8])> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at + 8 <= b.len() && out.len() < MAX_BOXES {
        let mut r = R { b, at };
        let (Some(size), Some(kind)) = (r.u32(), r.fourcc()) else { break };
        let (header, size) = match size {
            0 => (8, (b.len() - at) as u64),
            1 => match r.u64() {
                Some(s) => (16, s),
                None => break,
            },
            s => (8, u64::from(s)),
        };
        let Ok(size) = usize::try_from(size) else { break };
        if size < header {
            break;
        }
        let Some(end) = at.checked_add(size).filter(|e| *e <= b.len()) else { break };
        out.push((kind, &b[at + header..end]));
        at = end;
    }
    out
}

/// An item of the file (`infe`).
struct Item {
    id: u32,
    kind: [u8; 4],
}

/// Where an item's data is (`iloc`).
#[derive(Default)]
struct Loc {
    method: u8,
    base: u64,
    /// `(offset, length)`; a length of 0 runs to the end.
    extents: Vec<(u64, u64)>,
}

/// An item reference (`iref`): `from` refers to `to` (in order) as `kind`.
struct Ref {
    kind: [u8; 4],
    from: u32,
    to: Vec<u32>,
}

/// The properties FilmCraft uses (`ipco`).
#[derive(Clone, Debug, PartialEq)]
enum Prop<'a> {
    Hvcc(&'a [u8]),
    Ispe(u32, u32),
    /// Anticlockwise quarter turns.
    Irot(u8),
    /// 0: about a vertical axis (left and right swap), 1: about a horizontal axis.
    Imir(u8),
    /// Clean aperture: width, height, horizontal and vertical offset of its centre from the
    /// picture's, as fractions.
    Clap([f64; 4]),
    /// `colr nclx`: primaries, transfer, matrix, full range.
    Nclx(u8, u8, u8, bool),
    Other,
}

struct Heif<'a> {
    name: String,
    file: &'a [u8],
    primary: u32,
    items: Vec<Item>,
    locs: HashMap<u32, Loc>,
    idat: &'a [u8],
    refs: Vec<Ref>,
    props: Vec<Prop<'a>>,
    /// Each item's properties (indices into `props`), in association order.
    assoc: HashMap<u32, Vec<usize>>,
}

/// A decoded picture.
struct Pic {
    w: u32,
    h: u32,
    rgba: Vec<u8>,
}

impl<'a> Heif<'a> {
    fn parse(name: &str, file: &'a [u8]) -> Result<Self> {
        let top = boxes(file);
        let meta = top.iter().find(|(k, _)| k == b"meta").map(|(_, b)| *b).ok_or_else(|| bad(name, "no meta box"))?;
        let mut r = R::new(meta);
        r.full().ok_or_else(|| bad(name, "meta box too short"))?;
        let mut h = Heif {
            name: name.to_string(),
            file,
            primary: 0,
            items: Vec::new(),
            locs: HashMap::new(),
            idat: &[],
            refs: Vec::new(),
            props: Vec::new(),
            assoc: HashMap::new(),
        };
        let mut primary = None;
        for (kind, body) in boxes(meta.get(4..).unwrap_or_default()) {
            match &kind {
                b"pitm" => {
                    let mut r = R::new(body);
                    primary = match r.full() {
                        Some((0, _)) => r.u16().map(u32::from),
                        Some(_) => r.u32(),
                        None => None,
                    };
                }
                b"iinf" => h.items = infos(body),
                b"iloc" => h.locs = locations(body).ok_or_else(|| bad(name, "damaged iloc box"))?,
                b"iref" => h.refs = references(body),
                b"iprp" => {
                    for (k, b) in boxes(body) {
                        match &k {
                            b"ipco" => h.props = boxes(b).into_iter().map(|(k, b)| property(&k, b)).collect(),
                            b"ipma" => associations(b, &mut h.assoc),
                            _ => {}
                        }
                    }
                }
                b"idat" => h.idat = body,
                _ => {}
            }
        }
        h.primary = primary.ok_or_else(|| bad(name, "no primary item"))?;
        Ok(h)
    }

    fn kind_of(&self, id: u32) -> Option<[u8; 4]> {
        self.items.iter().find(|i| i.id == id).map(|i| i.kind)
    }

    fn props_of(&self, id: u32) -> impl Iterator<Item = &Prop<'a>> {
        self.assoc.get(&id).into_iter().flatten().filter_map(|i| self.props.get(*i))
    }

    /// An item's data, its extents joined.
    fn data(&self, id: u32) -> Result<Vec<u8>> {
        let name = &self.name;
        let loc = self.locs.get(&id).ok_or_else(|| bad(name, format!("item {id} has no location")))?;
        let src: &[u8] = match loc.method {
            0 => self.file,
            1 => self.idat,
            m => return Err(unsupported(name, format!("item data construction method {m}"))),
        };
        let mut out = Vec::new();
        for (off, len) in &loc.extents {
            let start = loc.base.checked_add(*off).and_then(|s| usize::try_from(s).ok()).ok_or_else(|| bad(name, "item data offset"))?;
            let end = if *len == 0 {
                src.len()
            } else {
                start.checked_add(usize::try_from(*len).map_err(|_| bad(name, "item data length"))?).ok_or_else(|| bad(name, "item data length"))?
            };
            let part = src.get(start..end).ok_or_else(|| bad(name, format!("item {id}'s data runs past the end of the file")))?;
            if out.len().saturating_add(part.len()) > src.len() {
                return Err(bad(name, "item data longer than the file"));
            }
            out.extend_from_slice(part);
        }
        Ok(out)
    }

    /// The items `id` derives from (`dimg`), in order.
    fn inputs(&self, id: u32) -> Vec<u32> {
        self.refs.iter().filter(|r| &r.kind == b"dimg" && r.from == id).flat_map(|r| r.to.iter().copied()).collect()
    }

    /// The colour an item's own `colr nclx` says, else `inherited`.
    fn nclx(&self, id: u32, inherited: Option<(u8, u8, u8, bool)>) -> Option<(u8, u8, u8, bool)> {
        self.props_of(id)
            .find_map(|p| match p {
                Prop::Nclx(a, b, c, d) => Some((*a, *b, *c, *d)),
                _ => None,
            })
            .or(inherited)
    }

    /// Item `id` decoded, its own transformations applied.
    fn picture(&self, id: u32, colour: Option<(u8, u8, u8, bool)>, depth: usize) -> Result<Pic> {
        let name = &self.name;
        if depth > MAX_DEPTH {
            return Err(bad(name, "derived images nest too deep"));
        }
        let colour = self.nclx(id, colour);
        let kind = self.kind_of(id).ok_or_else(|| bad(name, format!("no item {id}")))?;
        let pic = match &kind {
            b"hvc1" => {
                if let Some((w, h)) = self.size(id) {
                    check_pixels(name, w, h)?;
                }
                let hvcc = self
                    .props_of(id)
                    .find_map(|p| match p {
                        Prop::Hvcc(b) => Some(*b),
                        _ => None,
                    })
                    .ok_or_else(|| bad(name, format!("item {id} has no hvcC")))?;
                hevc_picture(name, hvcc, &self.data(id)?, colour)?
            }
            b"grid" => self.grid(id, colour, depth)?,
            b"iden" => {
                let input = *self.inputs(id).first().ok_or_else(|| bad(name, "an identity item without its input"))?;
                self.picture(input, colour, depth + 1)?
            }
            b"av01" => return Err(unsupported(name, "AVIF (AV1) images")),
            other => return Err(unsupported(name, format!("`{}` images", String::from_utf8_lossy(other)))),
        };
        Ok(self.transform(id, pic))
    }

    /// An item's `ispe` size.
    fn size(&self, id: u32) -> Option<(u32, u32)> {
        self.props_of(id).find_map(|p| match p {
            Prop::Ispe(w, h) => Some((*w, *h)),
            _ => None,
        })
    }

    /// A grid: its tiles decoded (in parallel), laid out and cut to its output size.
    fn grid(&self, id: u32, colour: Option<(u8, u8, u8, bool)>, depth: usize) -> Result<Pic> {
        let name = &self.name;
        let d = self.data(id)?;
        let mut r = R::new(&d);
        let (_version, flags) = (r.u8(), r.u8().unwrap_or(0));
        let (rows, cols) = (usize::from(r.u8().ok_or_else(|| bad(name, "short grid"))?) + 1, usize::from(r.u8().ok_or_else(|| bad(name, "short grid"))?) + 1);
        let (ow, oh) = if flags & 1 != 0 { (r.u32(), r.u32()) } else { (r.u16().map(u32::from), r.u16().map(u32::from)) };
        let (ow, oh) = (ow.ok_or_else(|| bad(name, "short grid"))?, oh.ok_or_else(|| bad(name, "short grid"))?);
        check_pixels(name, ow, oh)?;
        let tiles = self.inputs(id);
        let n = rows * cols;
        if n > MAX_TILES || tiles.len() < n {
            return Err(bad(name, format!("a {rows} × {cols} grid with {} tiles", tiles.len())));
        }
        let (ow, oh) = (ow as usize, oh as usize);
        // a tile is at most what its share of the grid needs (ISO/IEC 23008-12: the tiles cover
        // the output, the last row and column overhanging it): one claiming more is refused
        // before anything is decoded, so a hostile grid cannot multiply its memory by its tiles
        let share = |len: usize, parts: usize, out: usize| {
            if parts == 1 { len <= out.saturating_mul(2).saturating_add(64) } else { len.saturating_mul(parts - 1) < out }
        };
        let fits = |w: usize, h: usize| w > 0 && h > 0 && share(w, cols, ow) && share(h, rows, oh);
        let tiles = tiles.get(..n).ok_or_else(|| bad(name, "a grid without its tiles"))?;
        for t in tiles {
            if let Some((w, h)) = self.size(*t)
                && !fits(w as usize, h as usize)
            {
                return Err(bad(name, format!("a {w} × {h} tile in a {ow} × {oh} grid of {rows} × {cols}")));
            }
        }
        let mut out = vec![0u8; ow * oh * 4];
        let mut size: Option<(usize, usize)> = None;
        // a few tiles at a time, in parallel, each laid in as soon as it is decoded
        let per = rayon::current_num_threads().clamp(1, 16);
        for (c, chunk) in tiles.chunks(per).enumerate() {
            let decoded: Vec<Result<Pic>> = chunk.par_iter().map(|t| self.picture(*t, colour, depth + 1)).collect();
            for (j, tile) in decoded.into_iter().enumerate() {
                let tile = tile?;
                let (w, h) = (tile.w as usize, tile.h as usize);
                // every tile the first one's size, within its share
                let (tw, th) = *size.get_or_insert((w, h));
                if (w, h) != (tw, th) || !fits(w, h) {
                    return Err(bad(name, format!("a {w} × {h} tile in a {ow} × {oh} grid of {tw} × {th} tiles")));
                }
                let k = c * per + j;
                let (x0, y0) = ((k % cols) * tw, (k / cols) * th);
                for y in 0..h.min(oh.saturating_sub(y0)) {
                    let n = w.min(ow.saturating_sub(x0)) * 4;
                    let src = tile.rgba.get(y * w * 4..y * w * 4 + n);
                    let at = ((y0 + y) * ow + x0) * 4;
                    if let (Some(src), Some(dst)) = (src, out.get_mut(at..at + n)) {
                        dst.copy_from_slice(src);
                    }
                }
            }
        }
        Ok(Pic { w: ow as u32, h: oh as u32, rgba: out })
    }

    /// `pic` with item `id`'s transformations, in their order.
    fn transform(&self, id: u32, mut pic: Pic) -> Pic {
        for p in self.props_of(id) {
            pic = match p {
                Prop::Clap(c) => clap(pic, *c),
                Prop::Irot(a) => rotate(pic, *a),
                Prop::Imir(axis) => mirror(pic, *axis),
                _ => pic,
            };
        }
        pic
    }
}

fn check_pixels(name: &str, w: u32, h: u32) -> Result<()> {
    if w == 0 || h == 0 {
        return Err(bad(name, "a picture with no pixels"));
    }
    if u64::from(w) * u64::from(h) > MAX_PIXELS {
        return Err(unsupported(name, format!("{w} × {h} is more than {} megapixels", MAX_PIXELS >> 20)));
    }
    Ok(())
}

fn infos(body: &[u8]) -> Vec<Item> {
    let mut r = R::new(body);
    let n = match r.full() {
        Some((0, _)) => r.u16().map(u32::from),
        Some(_) => r.u32(),
        None => None,
    };
    if n.is_none() {
        return Vec::new();
    }
    boxes(body.get(r.at..).unwrap_or_default())
        .into_iter()
        .filter(|(k, _)| k == b"infe")
        .take(MAX_ITEMS)
        .filter_map(|(_, b)| {
            let mut r = R::new(b);
            let (version, _flags) = r.full()?;
            match version {
                2 | 3 => {
                    let id = if version == 2 { u32::from(r.u16()?) } else { r.u32()? };
                    r.u16()?; // protection index
                    let kind = r.fourcc()?;
                    r.skip_cstr();
                    Some(Item { id, kind })
                }
                // (versions 0 and 1 carry no item type: nothing FilmCraft decodes)
                _ => None,
            }
        })
        .collect()
}

fn locations(body: &[u8]) -> Option<HashMap<u32, Loc>> {
    let mut r = R::new(body);
    let (version, _) = r.full()?;
    let a = r.u8()?;
    let b = r.u8()?;
    let (off_size, len_size, base_size) = (a >> 4, a & 15, b >> 4);
    let index_size = if version == 1 || version == 2 { b & 15 } else { 0 };
    let n = if version < 2 { u32::from(r.u16()?) } else { r.u32()? };
    let mut out = HashMap::new();
    for _ in 0..n.min(MAX_ITEMS as u32) {
        let id = if version < 2 { u32::from(r.u16()?) } else { r.u32()? };
        let method = if version == 1 || version == 2 { (r.u16()? & 15) as u8 } else { 0 };
        r.u16()?; // data reference index
        let base = r.uint(base_size)?;
        let count = r.u16()?;
        let mut extents = Vec::with_capacity(usize::from(count.min(64)));
        for _ in 0..count {
            if index_size > 0 {
                r.uint(index_size)?;
            }
            let off = r.uint(off_size)?;
            let len = r.uint(len_size)?;
            extents.push((off, len));
        }
        out.insert(id, Loc { method, base, extents });
    }
    Some(out)
}

fn references(body: &[u8]) -> Vec<Ref> {
    let mut r = R::new(body);
    let Some((version, _)) = r.full() else { return Vec::new() };
    boxes(body.get(4..).unwrap_or_default())
        .into_iter()
        .take(MAX_ITEMS)
        .filter_map(|(kind, b)| {
            let mut r = R::new(b);
            let id = |r: &mut R| if version == 0 { r.u16().map(u32::from) } else { r.u32() };
            let from = id(&mut r)?;
            let n = r.u16()?;
            let to = (0..n).map_while(|_| id(&mut r)).collect();
            Some(Ref { kind, from, to })
        })
        .collect()
}

fn property<'a>(kind: &[u8; 4], b: &'a [u8]) -> Prop<'a> {
    let mut r = R::new(b);
    let p = match kind {
        b"hvcC" => Some(Prop::Hvcc(b)),
        b"ispe" => r.full().and_then(|_| Some(Prop::Ispe(r.u32()?, r.u32()?))),
        b"irot" => r.u8().map(|a| Prop::Irot(a & 3)),
        b"imir" => r.u8().map(|a| Prop::Imir(a & 1)),
        b"clap" => (|| {
            let mut v = [0.0f64; 4];
            for x in &mut v {
                let (n, d) = (r.u32()?, r.u32()?);
                // the offsets are signed
                if d == 0 {
                    return None;
                }
                *x = f64::from(n as i32) / f64::from(d);
            }
            // (width and height are unsigned)
            Some(Prop::Clap(v))
        })(),
        b"colr" => match r.fourcc() {
            Some(t) if &t == b"nclx" => (|| {
                let (p, t, m) = (r.u16()?, r.u16()?, r.u16()?);
                let full = r.u8()? & 0x80 != 0;
                Some(Prop::Nclx(u8::try_from(p).ok()?, u8::try_from(t).ok()?, u8::try_from(m).ok()?, full))
            })(),
            _ => None,
        },
        _ => None,
    };
    p.unwrap_or(Prop::Other)
}

fn associations(b: &[u8], out: &mut HashMap<u32, Vec<usize>>) {
    let mut r = R::new(b);
    let Some((version, flags)) = r.full() else { return };
    let Some(n) = r.u32() else { return };
    for _ in 0..n.min(MAX_ITEMS as u32) {
        let Some(id) = (if version < 1 { r.u16().map(u32::from) } else { r.u32() }) else { return };
        let Some(count) = r.u8() else { return };
        let list = out.entry(id).or_default();
        for _ in 0..count {
            let index = if flags & 1 != 0 { r.u16().map(|v| usize::from(v & 0x7FFF)) } else { r.u8().map(|v| usize::from(v & 0x7F)) };
            match index {
                // 1-based; 0 is no property
                Some(0) => {}
                Some(i) => list.push(i - 1),
                None => return,
            }
        }
    }
}

/// One HEVC picture as RGBA: the `hvcC` configures a single-threaded decoder (tiles are decoded
/// in parallel instead), the item's data is its access unit.
fn hevc_picture(name: &str, hvcc: &[u8], data: &[u8], colour: Option<(u8, u8, u8, bool)>) -> Result<Pic> {
    let err = |e: filmcraft_hevc::Error| match e {
        filmcraft_hevc::Error::Unsupported(what) => unsupported(name, format!("HEVC: {what}")),
        e => bad(name, format!("HEVC: {e}")),
    };
    let mut dec = filmcraft_hevc::Decoder::with_threads(1);
    dec.configure_hvcc(hvcc).map_err(err)?;
    let mut pics = dec.decode(data, 0).map_err(err)?;
    pics.extend(dec.flush());
    if let Some(e) = dec.take_error() {
        return Err(err(e));
    }
    let p = pics.into_iter().next().ok_or_else(|| bad(name, "a tile decoded to no picture"))?;
    check_pixels(name, p.width, p.height)?;
    let (w, h) = (p.width as usize, p.height as usize);
    let (cw, ch) = (p.chroma_width as usize, p.chroma_height as usize);
    let (ys, uvs) = (p.y_stride, p.uv_stride);
    let (primaries, transfer, matrix, full) = colour.unwrap_or((p.color.primaries, p.color.transfer, p.color.matrix, p.color.full_range));
    let color = crate::video::vui_color(p.width, p.height, primaries, matrix, transfer, full);
    let planes_ok = |len: usize, stride: usize, w: usize, h: usize| {
        h == 0 || (stride >= w && stride.checked_mul(h - 1).and_then(|n| n.checked_add(w)).is_some_and(|n| n <= len))
    };
    if !(planes_ok(p.y.len(), ys, w, h) && planes_ok(p.u.len(), uvs, cw, ch) && planes_ok(p.v.len(), uvs, cw, ch)) {
        return Err(bad(name, "a tile's planes are smaller than its size"));
    }
    let data = crate::video::hevc_pixel_data([p.y, p.u, p.v], (p.bit_depth, p.bit_depth_chroma), [w, h, cw, ch, ys, uvs]);
    let frame = filmcraft_frame::VideoFrame { width: p.width, height: p.height, data, color, par: (1, 1), pts: filmcraft_time::Tick::ZERO };
    Ok(Pic { w: p.width, h: p.height, rgba: frame.to_rgba8() })
}

/// The clean aperture `[width, height, horizontal offset, vertical offset]` of `pic` (ISO/IEC
/// 14496-12 12.1.4: a window centred `offset` from the picture's centre), within the picture.
fn clap(pic: Pic, c: [f64; 4]) -> Pic {
    if pic.w == 0 || pic.h == 0 {
        return pic;
    }
    let (w, h) = (f64::from(pic.w), f64::from(pic.h));
    let [cw, ch, dx, dy] = c;
    if !(cw.is_finite() && ch.is_finite() && dx.is_finite() && dy.is_finite()) || cw < 1.0 || ch < 1.0 {
        return pic;
    }
    let left = ((w - 1.0) / 2.0 + dx - (cw - 1.0) / 2.0).round().clamp(0.0, w - 1.0) as usize;
    let top = ((h - 1.0) / 2.0 + dy - (ch - 1.0) / 2.0).round().clamp(0.0, h - 1.0) as usize;
    let nw = (cw.round() as usize).min(pic.w as usize - left).max(1);
    let nh = (ch.round() as usize).min(pic.h as usize - top).max(1);
    let mut out = Vec::with_capacity(nw * nh * 4);
    for y in top..top + nh {
        let at = (y * pic.w as usize + left) * 4;
        match pic.rgba.get(at..at + nw * 4) {
            Some(row) => out.extend_from_slice(row),
            None => return pic,
        }
    }
    Pic { w: nw as u32, h: nh as u32, rgba: out }
}

fn as_image(pic: Pic) -> std::result::Result<image::RgbaImage, Pic> {
    let (w, h) = (pic.w, pic.h);
    if pic.rgba.len() != (w as usize) * (h as usize) * 4 {
        return Err(pic);
    }
    image::RgbaImage::from_raw(w, h, pic.rgba).ok_or(Pic { w, h, rgba: Vec::new() })
}

fn of_image(img: image::RgbaImage) -> Pic {
    let (w, h) = img.dimensions();
    Pic { w, h, rgba: img.into_raw() }
}

/// `pic` turned `quarters` quarter turns anticlockwise.
fn rotate(pic: Pic, quarters: u8) -> Pic {
    if quarters.is_multiple_of(4) {
        return pic;
    }
    match as_image(pic) {
        Ok(img) => of_image(match quarters % 4 {
            1 => image::imageops::rotate270(&img),
            2 => image::imageops::rotate180(&img),
            _ => image::imageops::rotate90(&img),
        }),
        Err(pic) => pic,
    }
}

/// `pic` mirrored about a vertical (`axis` 0: left and right swap) or horizontal axis (1).
fn mirror(pic: Pic, axis: u8) -> Pic {
    match as_image(pic) {
        Ok(img) => of_image(if axis == 0 { image::imageops::flip_horizontal(&img) } else { image::imageops::flip_vertical(&img) }),
        Err(pic) => pic,
    }
}

#[cfg(test)]
#[path = "heif_tests.rs"]
mod tests;
