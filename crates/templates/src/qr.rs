//! A QR code encoder (ISO/IEC 18004:2015, Model 2): byte mode, error correction levels L, M, Q
//! and H, versions 1–40, the eight data masks scored by the standard's penalty rules. Written from
//! the standard (the tables are its Table 9 and Annex E); nothing here is taken from another
//! implementation. The Templates panel shows a template's install link with it, and
//! `template.qr` writes one as a PNG.
//!
//! Untrusted input: the data is capped at what version 40 holds, every index is checked and
//! nothing panics.

/// Error correction level: how much of the symbol can be damaged and still read (about 7, 15,
/// 25 and 30 percent).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EcLevel {
    L,
    #[default]
    M,
    Q,
    H,
}

impl EcLevel {
    fn row(self) -> usize {
        match self {
            EcLevel::L => 0,
            EcLevel::M => 1,
            EcLevel::Q => 2,
            EcLevel::H => 3,
        }
    }
    /// The two format-information bits (Table 12).
    fn format_bits(self) -> u32 {
        match self {
            EcLevel::L => 1,
            EcLevel::M => 0,
            EcLevel::Q => 3,
            EcLevel::H => 2,
        }
    }
    pub fn from_id(s: &str) -> Option<EcLevel> {
        match s.to_ascii_uppercase().as_str() {
            "L" => Some(EcLevel::L),
            "M" => Some(EcLevel::M),
            "Q" => Some(EcLevel::Q),
            "H" => Some(EcLevel::H),
            _ => None,
        }
    }
}

/// Why data cannot be encoded.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum QrError {
    #[error("{0} bytes do not fit in a QR code at this error correction level (at most {1})")]
    TooLong(usize, usize),
    #[error("nothing to encode")]
    Empty,
}

/// Error correction codewords per block, by level (L, M, Q, H) and version 1–40 (Table 9).
const ECC_PER_BLOCK: [[u8; 40]; 4] = [
    [
        7, 10, 15, 20, 26, 18, 20, 24, 30, 18, 20, 24, 26, 30, 22, 24, 28, 30, 28, 28, 28, 28, 30, 30, 26, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30,
    ],
    [
        10, 16, 26, 18, 24, 16, 18, 22, 22, 26, 30, 22, 22, 24, 24, 28, 28, 26, 26, 26, 26, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
        28, 28,
    ],
    [
        13, 22, 18, 26, 18, 24, 18, 22, 20, 24, 28, 26, 24, 20, 30, 24, 28, 28, 26, 30, 28, 30, 30, 30, 30, 28, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30,
    ],
    [
        17, 28, 22, 16, 22, 28, 26, 26, 24, 28, 24, 28, 22, 24, 24, 30, 28, 28, 26, 28, 30, 24, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
        30, 30,
    ],
];

/// Error correction blocks, by level and version (Table 9).
const BLOCKS: [[u8; 40]; 4] = [
    [1, 1, 1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 4, 6, 6, 6, 6, 7, 8, 8, 9, 9, 10, 12, 12, 12, 13, 14, 15, 16, 17, 18, 19, 19, 20, 21, 22, 24, 25],
    [1, 1, 1, 2, 2, 4, 4, 4, 5, 5, 5, 8, 9, 9, 10, 10, 11, 13, 14, 16, 17, 17, 18, 20, 21, 23, 25, 26, 28, 29, 31, 33, 35, 37, 38, 40, 43, 45, 47, 49],
    [1, 1, 2, 2, 4, 4, 6, 6, 8, 8, 8, 10, 12, 16, 12, 17, 16, 18, 21, 20, 23, 23, 25, 27, 29, 34, 34, 35, 38, 40, 43, 45, 48, 51, 53, 56, 59, 62, 65, 68],
    [1, 1, 2, 4, 4, 4, 5, 6, 8, 8, 11, 11, 16, 16, 18, 16, 19, 21, 25, 25, 25, 34, 30, 32, 35, 37, 40, 42, 45, 48, 51, 54, 57, 60, 63, 66, 70, 74, 77, 81],
];

/// Row / column coordinates of the alignment pattern centres, versions 2–40 (Annex E, Table E.1).
const ALIGNMENT: [&[u8]; 39] = [
    &[6, 18],
    &[6, 22],
    &[6, 26],
    &[6, 30],
    &[6, 34],
    &[6, 22, 38],
    &[6, 24, 42],
    &[6, 26, 46],
    &[6, 28, 50],
    &[6, 30, 54],
    &[6, 32, 58],
    &[6, 34, 62],
    &[6, 26, 46, 66],
    &[6, 26, 48, 70],
    &[6, 26, 50, 74],
    &[6, 30, 54, 78],
    &[6, 30, 56, 82],
    &[6, 30, 58, 86],
    &[6, 34, 62, 90],
    &[6, 28, 50, 72, 94],
    &[6, 26, 50, 74, 98],
    &[6, 30, 54, 78, 102],
    &[6, 28, 54, 80, 106],
    &[6, 32, 58, 84, 110],
    &[6, 30, 58, 86, 114],
    &[6, 34, 62, 90, 118],
    &[6, 26, 50, 74, 98, 122],
    &[6, 30, 54, 78, 102, 126],
    &[6, 26, 52, 78, 104, 130],
    &[6, 30, 56, 82, 108, 134],
    &[6, 34, 60, 86, 112, 138],
    &[6, 30, 58, 86, 114, 142],
    &[6, 34, 62, 90, 118, 146],
    &[6, 30, 54, 78, 102, 126, 150],
    &[6, 24, 50, 76, 102, 128, 154],
    &[6, 28, 54, 80, 106, 132, 158],
    &[6, 32, 58, 84, 110, 136, 162],
    &[6, 26, 54, 82, 110, 138, 166],
    &[6, 30, 58, 86, 114, 142, 170],
];

/// A QR code symbol: `size` × `size` modules, `true` = dark.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QrCode {
    pub version: u8,
    pub size: usize,
    pub ec: EcLevel,
    pub mask: u8,
    modules: Vec<bool>,
}

/// The matrix while it is built: modules and which of them are function patterns.
struct Grid {
    size: usize,
    dark: Vec<bool>,
    function: Vec<bool>,
}

impl Grid {
    fn new(size: usize) -> Grid {
        Grid { size, dark: vec![false; size * size], function: vec![false; size * size] }
    }
    fn at(&self, x: usize, y: usize) -> Option<usize> {
        (x < self.size && y < self.size).then_some(y * self.size + x)
    }
    fn set_function(&mut self, x: usize, y: usize, dark: bool) {
        if let Some(i) = self.at(x, y) {
            if let Some(d) = self.dark.get_mut(i) {
                *d = dark;
            }
            if let Some(f) = self.function.get_mut(i) {
                *f = true;
            }
        }
    }
    fn is_function(&self, x: usize, y: usize) -> bool {
        self.at(x, y).and_then(|i| self.function.get(i).copied()).unwrap_or(true)
    }
    fn get(&self, x: usize, y: usize) -> bool {
        self.at(x, y).and_then(|i| self.dark.get(i).copied()).unwrap_or(false)
    }
    fn flip(&mut self, x: usize, y: usize) {
        if let Some(d) = self.at(x, y).and_then(|i| self.dark.get_mut(i)) {
            *d = !*d;
        }
    }
}

/// Side of a version's symbol.
fn side(version: u8) -> usize {
    17 + 4 * usize::from(version)
}

fn table(t: &[[u8; 40]; 4], ec: EcLevel, version: u8) -> usize {
    t.get(ec.row()).and_then(|r| r.get(usize::from(version).wrapping_sub(1))).map_or(0, |n| usize::from(*n))
}

/// The function patterns of `version`: finders and their separators, timing, alignment, and the
/// format and version areas (reserved; written once the mask is known).
fn function_patterns(version: u8) -> Grid {
    let size = side(version);
    let mut g = Grid::new(size);
    for i in 0..size {
        g.set_function(6, i, i % 2 == 0);
        g.set_function(i, 6, i % 2 == 0);
    }
    for (cx, cy) in [(3usize, 3usize), (size - 4, 3), (3, size - 4)] {
        for dy in -4i32..=4 {
            for dx in -4i32..=4 {
                let d = dx.abs().max(dy.abs());
                let (x, y) = (cx as i32 + dx, cy as i32 + dy);
                if x >= 0 && y >= 0 {
                    g.set_function(x as usize, y as usize, d != 2 && d != 4);
                }
            }
        }
    }
    if version >= 2 {
        let pos = ALIGNMENT.get(usize::from(version) - 2).copied().unwrap_or(&[]);
        let last = pos.last().copied().unwrap_or(6);
        for &cy in pos {
            for &cx in pos {
                // the three corners are finders
                if (cx == 6 && cy == 6) || (cx == 6 && cy == last) || (cx == last && cy == 6) {
                    continue;
                }
                for dy in -2i32..=2 {
                    for dx in -2i32..=2 {
                        let d = dx.abs().max(dy.abs());
                        g.set_function((i32::from(cx) + dx) as usize, (i32::from(cy) + dy) as usize, d != 1);
                    }
                }
            }
        }
    }
    // reserve the format areas (and the dark module) and the version areas
    draw_format(&mut g, EcLevel::M, 0);
    if version >= 7 {
        draw_version(&mut g, version);
    }
    g
}

/// BCH(15, 5) format information of `ec` and `mask`, XOR-masked (§7.9).
fn format_word(ec: EcLevel, mask: u8) -> u32 {
    let data = (ec.format_bits() << 3) | u32::from(mask & 7);
    let mut rem = data;
    for _ in 0..10 {
        rem = (rem << 1) ^ ((rem >> 9) * 0x537);
    }
    ((data << 10) | (rem & 0x3ff)) ^ 0x5412
}

fn draw_format(g: &mut Grid, ec: EcLevel, mask: u8) {
    let bits = format_word(ec, mask);
    let bit = |i: u32| (bits >> i) & 1 == 1;
    let size = g.size;
    // around the top-left finder: bit 0 at the top of column 8, bit 14 at the left of row 8
    for i in 0..6 {
        g.set_function(8, i as usize, bit(i));
    }
    g.set_function(8, 7, bit(6));
    g.set_function(8, 8, bit(7));
    g.set_function(7, 8, bit(8));
    for i in 9..15 {
        g.set_function(14 - i as usize, 8, bit(i));
    }
    // the copy beside the top-right and bottom-left finders
    for i in 0..8 {
        g.set_function(size - 1 - i as usize, 8, bit(i));
    }
    for i in 8..15 {
        g.set_function(8, size - 15 + i as usize, bit(i));
    }
    g.set_function(8, size - 8, true);
}

/// BCH(18, 6) version information (§7.10), versions 7 and up.
fn version_word(version: u8) -> u32 {
    let data = u32::from(version);
    let mut rem = data;
    for _ in 0..12 {
        rem = (rem << 1) ^ ((rem >> 11) * 0x1f25);
    }
    (data << 12) | (rem & 0xfff)
}

fn draw_version(g: &mut Grid, version: u8) {
    let bits = version_word(version);
    let size = g.size;
    for i in 0..18usize {
        let dark = (bits >> i) & 1 == 1;
        let (a, b) = (size - 11 + i % 3, i / 3);
        g.set_function(a, b, dark);
        g.set_function(b, a, dark);
    }
}

/// How many codewords the data area of `version` holds (data and error correction).
fn total_codewords(version: u8) -> usize {
    let g = function_patterns(version);
    g.function.iter().filter(|f| !**f).count() / 8
}

/// Data codewords of `version` at `ec`.
fn data_codewords(version: u8, ec: EcLevel) -> usize {
    total_codewords(version).saturating_sub(table(&ECC_PER_BLOCK, ec, version) * table(&BLOCKS, ec, version))
}

/// Bytes of data byte mode holds in `version` at `ec`.
pub fn capacity(version: u8, ec: EcLevel) -> usize {
    let count_bits = if version <= 9 { 8 } else { 16 };
    (data_codewords(version, ec) * 8).saturating_sub(4 + count_bits) / 8
}

// ------------------------------------------------------------------------------------------------
// Reed–Solomon over GF(2^8) with the primitive polynomial x^8 + x^4 + x^3 + x^2 + 1 (0x11d)
// ------------------------------------------------------------------------------------------------

struct Gf {
    exp: [u8; 512],
    log: [u8; 256],
}

impl Gf {
    fn new() -> Gf {
        let mut exp = [0u8; 512];
        let mut log = [0u8; 256];
        let mut x: u32 = 1;
        for (i, e) in exp.iter_mut().enumerate().take(255) {
            *e = x as u8;
            if let Some(l) = log.get_mut(x as usize) {
                *l = i as u8;
            }
            x <<= 1;
            if x & 0x100 != 0 {
                x ^= 0x11d;
            }
        }
        for i in 255..512 {
            exp[i] = exp[i - 255];
        }
        Gf { exp, log }
    }
    fn mul(&self, a: u8, b: u8) -> u8 {
        if a == 0 || b == 0 {
            return 0;
        }
        let i = usize::from(self.log[usize::from(a)]) + usize::from(self.log[usize::from(b)]);
        self.exp[i % 255]
    }
    /// The generator polynomial of degree `n` (roots α^0 … α^(n-1)), highest power first,
    /// without its leading 1.
    fn generator(&self, n: usize) -> Vec<u8> {
        let mut g = vec![0u8; n];
        if let Some(last) = g.last_mut() {
            *last = 1;
        }
        let mut root: u8 = 1;
        for _ in 0..n {
            // multiply by (x - root)
            for j in 0..n {
                let v = self.mul(g[j], root);
                g[j] = v ^ g.get(j + 1).copied().unwrap_or(0);
            }
            root = self.mul(root, 2);
        }
        g
    }
    /// The `n` error correction codewords of `data`.
    fn remainder(&self, data: &[u8], generator: &[u8]) -> Vec<u8> {
        let n = generator.len();
        let mut r = vec![0u8; n];
        for &b in data {
            let factor = b ^ r.first().copied().unwrap_or(0);
            r.rotate_left(1);
            if let Some(last) = r.last_mut() {
                *last = 0;
            }
            for (x, g) in r.iter_mut().zip(generator) {
                *x ^= self.mul(*g, factor);
            }
        }
        r
    }
}

// ------------------------------------------------------------------------------------------------
// Encoding
// ------------------------------------------------------------------------------------------------

struct Bits(Vec<bool>);

impl Bits {
    fn push(&mut self, value: u32, n: u32) {
        for i in (0..n).rev() {
            self.0.push((value >> i) & 1 == 1);
        }
    }
}

/// The data codewords: byte mode, the count, the bytes, the terminator and the pad codewords.
fn data_bytes(data: &[u8], version: u8, ec: EcLevel) -> Vec<u8> {
    let cap = data_codewords(version, ec) * 8;
    let mut b = Bits(Vec::with_capacity(cap));
    b.push(0b0100, 4);
    b.push(data.len() as u32, if version <= 9 { 8 } else { 16 });
    for &x in data {
        b.push(u32::from(x), 8);
    }
    let term = cap.saturating_sub(b.0.len()).min(4);
    b.push(0, term as u32);
    while !b.0.len().is_multiple_of(8) {
        b.0.push(false);
    }
    let mut out: Vec<u8> = b.0.chunks(8).map(|c| c.iter().fold(0u8, |a, bit| (a << 1) | u8::from(*bit))).collect();
    let mut pad = [0xec, 0x11].into_iter().cycle();
    while out.len() < cap / 8 {
        out.push(pad.next().unwrap_or(0xec));
    }
    out
}

/// Split into blocks, add each block's error correction and interleave (§7.6).
fn interleave(data: &[u8], version: u8, ec: EcLevel) -> Vec<u8> {
    let blocks = table(&BLOCKS, ec, version).max(1);
    let ecc = table(&ECC_PER_BLOCK, ec, version);
    let total = total_codewords(version);
    let short_len = total / blocks; // codewords in a short block
    let long_blocks = total % blocks;
    let short_blocks = blocks - long_blocks;
    let gf = Gf::new();
    let generator = gf.generator(ecc);
    let mut split: Vec<(Vec<u8>, Vec<u8>)> = Vec::with_capacity(blocks);
    let mut at = 0usize;
    for i in 0..blocks {
        let len = short_len.saturating_sub(ecc) + usize::from(i >= short_blocks);
        let d = data.get(at..at + len).map(<[u8]>::to_vec).unwrap_or_default();
        at += len;
        let e = gf.remainder(&d, &generator);
        split.push((d, e));
    }
    let mut out = Vec::with_capacity(total);
    let longest = split.iter().map(|b| b.0.len()).max().unwrap_or(0);
    for k in 0..longest {
        for b in &split {
            if let Some(x) = b.0.get(k) {
                out.push(*x);
            }
        }
    }
    for k in 0..ecc {
        for b in &split {
            if let Some(x) = b.1.get(k) {
                out.push(*x);
            }
        }
    }
    out
}

/// Place the codewords in the two-module-wide columns, bottom right first, up and down in turn,
/// skipping the vertical timing pattern (§7.7.3).
fn place(g: &mut Grid, codewords: &[u8]) {
    let size = g.size as isize;
    let total = codewords.len() * 8;
    let mut i = 0usize;
    let mut right = size - 1;
    while right >= 1 {
        if right == 6 {
            right = 5;
        }
        for vert in 0..size {
            for j in 0..2 {
                let x = (right - j) as usize;
                let upward = (right + 1) & 2 == 0;
                let y = if upward { (size - 1 - vert) as usize } else { vert as usize };
                if !g.is_function(x, y) && i < total {
                    let byte = codewords.get(i / 8).copied().unwrap_or(0);
                    if (byte >> (7 - (i % 8))) & 1 == 1
                        && let Some(d) = g.at(x, y).and_then(|k| g.dark.get_mut(k))
                    {
                        *d = true;
                    }
                    i += 1;
                }
            }
        }
        right -= 2;
    }
}

/// Whether data mask `m` flips the module at row `i`, column `j` (Table 10).
fn mask_bit(m: u8, i: usize, j: usize) -> bool {
    match m {
        0 => (i + j).is_multiple_of(2),
        1 => i.is_multiple_of(2),
        2 => j.is_multiple_of(3),
        3 => (i + j).is_multiple_of(3),
        4 => (i / 2 + j / 3).is_multiple_of(2),
        5 => (i * j) % 2 + (i * j) % 3 == 0,
        6 => ((i * j) % 2 + (i * j) % 3).is_multiple_of(2),
        _ => ((i + j) % 2 + (i * j) % 3).is_multiple_of(2),
    }
}

fn apply_mask(g: &mut Grid, m: u8) {
    for y in 0..g.size {
        for x in 0..g.size {
            if !g.is_function(x, y) && mask_bit(m, y, x) {
                g.flip(x, y);
            }
        }
    }
}

/// The penalty score of a masked symbol (§7.8.3): runs, 2×2 blocks, finder-like patterns and
/// the balance of dark and light.
fn penalty(g: &Grid) -> u64 {
    let n = g.size;
    let mut score: u64 = 0;
    let line = |k: usize, i: usize, rows: bool| if rows { g.get(i, k) } else { g.get(k, i) };
    const FINDER_A: [bool; 11] = [true, false, true, true, true, false, true, false, false, false, false];
    const FINDER_B: [bool; 11] = [false, false, false, false, true, false, true, true, true, false, true];
    for rows in [true, false] {
        for k in 0..n {
            // N1: runs of five or more of one colour
            let mut run = 1usize;
            for i in 1..n {
                if line(k, i, rows) == line(k, i - 1, rows) {
                    run += 1;
                } else {
                    if run >= 5 {
                        score += 3 + (run as u64 - 5);
                    }
                    run = 1;
                }
            }
            if run >= 5 {
                score += 3 + (run as u64 - 5);
            }
            // N3: 1:1:3:1:1 with four light modules on one side
            if n >= 11 {
                for i in 0..=n - 11 {
                    let matches = |p: &[bool; 11]| p.iter().enumerate().all(|(o, want)| line(k, i + o, rows) == *want);
                    if matches(&FINDER_A) || matches(&FINDER_B) {
                        score += 40;
                    }
                }
            }
        }
    }
    // N2: 2×2 blocks of one colour
    for y in 0..n.saturating_sub(1) {
        for x in 0..n.saturating_sub(1) {
            let c = g.get(x, y);
            if c == g.get(x + 1, y) && c == g.get(x, y + 1) && c == g.get(x + 1, y + 1) {
                score += 3;
            }
        }
    }
    // N4: how far the dark share is from half, in 5% steps
    let dark = g.dark.iter().filter(|d| **d).count() as u64;
    let total = (n * n).max(1) as u64;
    let pct = dark * 100 / total;
    score += 10 * (pct.abs_diff(50) / 5);
    score
}

impl QrCode {
    /// Encode `data` in byte mode at error correction `ec`, in the smallest version that holds it.
    pub fn encode(data: &[u8], ec: EcLevel) -> Result<QrCode, QrError> {
        if data.is_empty() {
            return Err(QrError::Empty);
        }
        let max = capacity(40, ec);
        let version = (1..=40u8).find(|v| capacity(*v, ec) >= data.len()).ok_or(QrError::TooLong(data.len(), max))?;
        let words = interleave(&data_bytes(data, version, ec), version, ec);
        let mut base = function_patterns(version);
        place(&mut base, &words);
        let mut best: Option<(u64, u8, Grid)> = None;
        for m in 0..8u8 {
            let mut g = Grid { size: base.size, dark: base.dark.clone(), function: base.function.clone() };
            apply_mask(&mut g, m);
            draw_format(&mut g, ec, m);
            let p = penalty(&g);
            if best.as_ref().is_none_or(|(b, ..)| p < *b) {
                best = Some((p, m, g));
            }
        }
        let (_, mask, g) = best.ok_or(QrError::Empty)?;
        Ok(QrCode { version, size: g.size, ec, mask, modules: g.dark })
    }

    /// Encode text (UTF-8).
    pub fn encode_text(text: &str, ec: EcLevel) -> Result<QrCode, QrError> {
        QrCode::encode(text.as_bytes(), ec)
    }

    /// Whether module (`x`, `y`) is dark (outside the symbol: light).
    pub fn get(&self, x: usize, y: usize) -> bool {
        x < self.size && y < self.size && self.modules.get(y * self.size + x).copied().unwrap_or(false)
    }

    /// Rows of `1` (dark) and `0` (light).
    pub fn rows(&self) -> Vec<String> {
        (0..self.size).map(|y| (0..self.size).map(|x| if self.get(x, y) { '1' } else { '0' }).collect()).collect()
    }

    /// The symbol as RGBA8 pixels, black on white: `scale` pixels a module (1–64) and a quiet
    /// zone of `border` modules (0–16; the standard asks for 4). Returns (width, height, pixels).
    pub fn to_rgba(&self, scale: u32, border: u32) -> (u32, u32, Vec<u8>) {
        let scale = scale.clamp(1, 64) as usize;
        let border = border.min(16) as usize;
        let side = (self.size + 2 * border) * scale;
        let mut px = vec![255u8; side * side * 4];
        for y in 0..self.size {
            for x in 0..self.size {
                if !self.get(x, y) {
                    continue;
                }
                for dy in 0..scale {
                    let row = (y + border) * scale + dy;
                    for dx in 0..scale {
                        let col = (x + border) * scale + dx;
                        let i = (row * side + col) * 4;
                        if let Some(p) = px.get_mut(i..i + 3) {
                            p.fill(0);
                        }
                    }
                }
            }
        }
        (side as u32, side as u32, px)
    }

    /// The symbol as an SVG image (one path, `border` modules of quiet zone).
    pub fn to_svg(&self, border: u32) -> String {
        let b = border.min(16) as usize;
        let side = self.size + 2 * b;
        let mut path = String::new();
        for y in 0..self.size {
            for x in 0..self.size {
                if self.get(x, y) {
                    path.push_str(&format!("M{},{}h1v1h-1z", x + b, y + b));
                }
            }
        }
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {side} {side}\" shape-rendering=\"crispEdges\"><rect width=\"{side}\" height=\"{side}\" fill=\"#fff\"/><path d=\"{path}\" fill=\"#000\"/></svg>"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Read the symbol back: format information, unmask, collect the codewords, check every
    /// block's Reed–Solomon syndromes and decode the byte-mode segment. A reader of our own
    /// symbols (tested against the standard's numbers below), so encoding round-trips.
    fn read(q: &QrCode) -> Vec<u8> {
        let g = Grid { size: q.size, dark: q.modules.clone(), function: function_patterns(q.version).function };
        // the format information (first copy) names the level and mask
        let mut bits = 0u32;
        let size = q.size;
        let pos: Vec<(usize, usize)> = (0..6).map(|i| (8, i)).chain([(8, 7), (8, 8), (7, 8)]).chain((9..15).map(|i| (14 - i, 8))).collect();
        for (i, (x, y)) in pos.into_iter().enumerate() {
            if g.get(x, y) {
                bits |= 1 << i;
            }
        }
        let (ec, mask) = (0..4)
            .flat_map(|e| (0..8u8).map(move |m| (e, m)))
            .map(|(e, m)| ([EcLevel::L, EcLevel::M, EcLevel::Q, EcLevel::H][e], m))
            .find(|(e, m)| format_word(*e, *m) == bits)
            .expect("format information");
        assert_eq!((ec, mask), (q.ec, q.mask));
        let mut g = g;
        apply_mask(&mut g, mask);
        // collect bits in placement order
        let mut out = Vec::new();
        let mut right = size as isize - 1;
        while right >= 1 {
            if right == 6 {
                right = 5;
            }
            for vert in 0..size as isize {
                for j in 0..2 {
                    let x = (right - j) as usize;
                    let upward = (right + 1) & 2 == 0;
                    let y = if upward { (size as isize - 1 - vert) as usize } else { vert as usize };
                    if !g.is_function(x, y) {
                        out.push(g.get(x, y));
                    }
                }
            }
            right -= 2;
        }
        let words: Vec<u8> = out.chunks(8).filter(|c| c.len() == 8).map(|c| c.iter().fold(0u8, |a, b| (a << 1) | u8::from(*b))).collect();
        let total = total_codewords(q.version);
        let words = &words[..total];
        // de-interleave and check each block: its syndromes are all zero
        let blocks = table(&BLOCKS, ec, q.version);
        let ecc = table(&ECC_PER_BLOCK, ec, q.version);
        let short_len = total / blocks;
        let short_blocks = blocks - total % blocks;
        let lens: Vec<usize> = (0..blocks).map(|i| short_len - ecc + usize::from(i >= short_blocks)).collect();
        let mut data_blocks: Vec<Vec<u8>> = vec![Vec::new(); blocks];
        let mut k = 0;
        for col in 0..*lens.iter().max().unwrap() {
            for (b, len) in lens.iter().enumerate() {
                if col < *len {
                    data_blocks[b].push(words[k]);
                    k += 1;
                }
            }
        }
        let mut ecc_blocks: Vec<Vec<u8>> = vec![Vec::new(); blocks];
        for _ in 0..ecc {
            for b in ecc_blocks.iter_mut() {
                b.push(words[k]);
                k += 1;
            }
        }
        let gf = Gf::new();
        for (d, e) in data_blocks.iter().zip(&ecc_blocks) {
            let cw: Vec<u8> = d.iter().chain(e).copied().collect();
            let mut root = 1u8;
            for _ in 0..ecc {
                // evaluate at root (Horner)
                let s = cw.iter().fold(0u8, |acc, c| gf.mul(acc, root) ^ c);
                assert_eq!(s, 0, "a syndrome is not zero");
                root = gf.mul(root, 2);
            }
        }
        let data: Vec<u8> = data_blocks.concat();
        let bit = |i: usize| (data[i / 8] >> (7 - i % 8)) & 1 == 1;
        let num = |from: usize, n: usize| (from..from + n).fold(0usize, |a, i| (a << 1) | usize::from(bit(i)));
        assert_eq!(num(0, 4), 0b0100, "byte mode");
        let cb = if q.version <= 9 { 8 } else { 16 };
        let len = num(4, cb);
        (0..len).map(|i| num(4 + cb + i * 8, 8) as u8).collect()
    }

    #[test]
    fn capacities_match_the_standard() {
        // byte-mode capacities of Table 7
        for (v, l, m, q, h) in
            [(1, 17, 14, 11, 7), (2, 32, 26, 20, 14), (5, 106, 84, 60, 44), (7, 154, 122, 86, 64), (10, 271, 213, 151, 119), (40, 2953, 2331, 1663, 1273)]
        {
            assert_eq!([capacity(v, EcLevel::L), capacity(v, EcLevel::M), capacity(v, EcLevel::Q), capacity(v, EcLevel::H)], [l, m, q, h], "version {v}");
        }
        // total codewords of Table 9
        for (v, n) in [(1, 26), (2, 44), (6, 172), (7, 196), (14, 581), (21, 1156), (32, 2465), (40, 3706)] {
            assert_eq!(total_codewords(v), n, "version {v}");
        }
    }

    #[test]
    fn format_and_version_words_match_the_standard() {
        // Annex C: level M mask 5 is 100000011001110; version 7 is 000111110010010100
        assert_eq!(format_word(EcLevel::M, 5), 0b100000011001110);
        assert_eq!(format_word(EcLevel::L, 0), 0b111011111000100);
        assert_eq!(version_word(7), 0b000111110010010100);
        assert_eq!(version_word(40), 0b101000110001101001);
    }

    #[test]
    fn reed_solomon_matches_the_standard_example() {
        // Annex I: "01234567" at 1-M — data codewords and their ten error correction codewords
        let gf = Gf::new();
        let data = [0x10, 0x20, 0x0c, 0x56, 0x61, 0x80, 0xec, 0x11, 0xec, 0x11, 0xec, 0x11, 0xec, 0x11, 0xec, 0x11];
        assert_eq!(gf.remainder(&data, &gf.generator(10)), [0xa5, 0x24, 0xd4, 0xc1, 0xed, 0x36, 0xc7, 0x87, 0x2c, 0x55]);
    }

    #[test]
    fn encodes_and_reads_back_at_every_level_and_many_versions() {
        let url = "https://example.org/templates/Photo%20Beat%20Slideshow.zip";
        for ec in [EcLevel::L, EcLevel::M, EcLevel::Q, EcLevel::H] {
            let q = QrCode::encode_text(url, ec).unwrap();
            assert_eq!(q.size, 17 + 4 * q.version as usize);
            assert_eq!(read(&q), url.as_bytes(), "{ec:?}");
        }
        for len in [1, 14, 15, 100, 213, 214, 500, 1200, 2331] {
            let data: Vec<u8> = (0..len).map(|i| (i * 37 % 251) as u8).collect();
            let q = QrCode::encode(&data, EcLevel::M).unwrap();
            assert_eq!(read(&q), data, "{len} bytes, version {}", q.version);
        }
        assert_eq!(QrCode::encode(&[0; 2332], EcLevel::M), Err(QrError::TooLong(2332, 2331)));
        assert_eq!(QrCode::encode(&[], EcLevel::M), Err(QrError::Empty));
    }

    #[test]
    fn renders_with_a_quiet_zone() {
        let q = QrCode::encode_text("hi", EcLevel::M).unwrap();
        assert_eq!((q.version, q.size), (1, 21));
        let (w, h, px) = q.to_rgba(4, 4);
        assert_eq!((w, h), (116, 116));
        assert_eq!(px.len(), 116 * 116 * 4);
        // the quiet zone is white, the top-left finder's corner black
        assert_eq!(&px[0..4], &[255, 255, 255, 255]);
        let at = |x: usize, y: usize| px[(y * 116 + x) * 4];
        assert_eq!(at(16, 16), 0, "the finder's dark border");
        assert_eq!(at(16 + 4, 16 + 4), 255, "the finder's light ring");
        assert_eq!(at(16 + 8, 16 + 8), 0, "the finder's dark centre");
        assert!(q.to_svg(4).starts_with("<svg"));
        assert_eq!(q.rows().len(), 21);
        // absurd scales are capped
        let (w, ..) = q.to_rgba(10_000, 10_000);
        assert_eq!(w, (21 + 32) * 64);
    }
}
