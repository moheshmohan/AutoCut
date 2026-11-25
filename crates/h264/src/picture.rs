//! Picture buffers and per-macroblock state.

use crate::{ChromaFormat, Plane};
use std::sync::{Arc, OnceLock};

/// Sample format of a picture: chroma sampling and the luma / chroma bit depths (8..=10).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format {
    pub chroma: ChromaFormat,
    pub bit_depth_y: u32,
    pub bit_depth_c: u32,
}

impl Format {
    /// 8-bit 4:2:0 (Baseline / Main / High).
    pub const YUV420_8: Format = Format { chroma: ChromaFormat::C420, bit_depth_y: 8, bit_depth_c: 8 };

    /// Whether the planes are stored as `u8` (both depths 8), else as `u16`.
    #[inline]
    pub fn stores_u8(self) -> bool {
        self.bit_depth_y <= 8 && self.bit_depth_c <= 8
    }
    /// log2(SubHeightC): 1 for 4:2:0, 0 for 4:2:2.
    #[inline(always)]
    pub fn sy(self) -> u32 {
        self.chroma.shifts().1
    }
    /// MbHeightC: chroma lines per macroblock (8 or 16).
    #[inline(always)]
    pub fn mb_height_c(self) -> usize {
        16 >> self.sy()
    }
    /// log2(MbHeightC).
    #[inline(always)]
    pub fn log2_mb_height_c(self) -> u32 {
        4 - self.sy()
    }
    /// Bytes of one picture of `luma_samples` luma samples in this format (every plane).
    pub fn picture_bytes(self, luma_samples: u64) -> u64 {
        let per_sample = if self.stores_u8() { 1 } else { 2 };
        // chroma planes: half the luma (4:2:0) or the same (4:2:2), for Cb and Cr together
        let chroma = match self.chroma {
            ChromaFormat::C420 => luma_samples / 2,
            ChromaFormat::C422 => luma_samples,
        };
        luma_samples.saturating_add(chroma).saturating_mul(per_sample)
    }
}

/// Storage type of decoded samples: `u8` for 8-bit pictures (the original, fastest path), `u16`
/// when either bit depth is above 8. Values are always within `0..1 << bit_depth`.
pub trait Sample: Copy + Default + PartialEq + std::fmt::Debug + Send + Sync + 'static {
    /// Bound of the inverse transform input (8.5.12.1: −2^(7 + bitDepth)..2^(7 + bitDepth) − 1)
    /// the butterflies clamp to.
    const COEF_LIMIT: i32;
    /// The largest sample value at `bit_depth` (always 255 for `u8` storage).
    fn max_value(bit_depth: u32) -> i32;
    /// 1 << (bit_depth − 1), the value of unavailable intra neighbours (128 for `u8`).
    fn half(bit_depth: u32) -> i32;
    fn to_i32(self) -> i32;
    fn to_i16(self) -> i16;
    /// `v`, already within the sample range, in the storage type.
    fn from_i32(v: i32) -> Self;
    /// `v` clipped to 0..=max (Clip1).
    fn clip(v: i32, max: i32) -> Self;
    /// Rounded average (a + b + 1) >> 1, in the narrowest lanes that hold it.
    fn avg(a: Self, b: Self) -> Self;
    /// The rows of `f` if they are stored with this type.
    fn rows(f: &Frame) -> Option<&[OnceLock<FrameRow<Self>>]>;
    fn wrap_rows(rows: Box<[OnceLock<FrameRow<Self>>]>) -> Rows;
    fn into_plane(v: Vec<Self>) -> Plane;
    /// An empty output plane with room for `n` samples.
    fn alloc(n: usize) -> Vec<Self>;
    /// Chroma interpolation (8.4.2.2.2), see [`crate::inter::mc_chroma_win`].
    fn mc_chroma_win(src: &[Self], ss: usize, fx: u32, fy: u32, bw: usize, bh: usize, out: &mut [Self], os: usize);
    /// The eight samples p3..q3 of `N` lines across the vertical edge whose q0 column is `x`,
    /// transposed (`s[k][i]` = sample k of line i), or None outside the plane.
    fn load_vertical<const N: usize>(pix: &[Self], stride: usize, x: usize, y: usize) -> Option<[[i16; N]; 8]>;
    /// Write back what [`Sample::load_vertical`] read.
    fn store_vertical<const N: usize>(pix: &mut [Self], stride: usize, x: usize, y: usize, s: &[[i16; N]; 8]);
}

impl Sample for u8 {
    const COEF_LIMIT: i32 = 1 << 15;
    #[inline(always)]
    fn max_value(_bit_depth: u32) -> i32 {
        255
    }
    #[inline(always)]
    fn half(_bit_depth: u32) -> i32 {
        128
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u8
    }
    #[inline(always)]
    fn clip(v: i32, _max: i32) -> Self {
        v.clamp(0, 255) as u8
    }
    #[inline(always)]
    fn avg(a: u8, b: u8) -> u8 {
        ((a as u16 + b as u16 + 1) >> 1) as u8
    }
    #[inline(always)]
    fn rows(f: &Frame) -> Option<&[OnceLock<FrameRow<u8>>]> {
        match &f.rows {
            Rows::U8(r) => Some(r),
            Rows::U16(_) => None,
        }
    }
    fn wrap_rows(rows: Box<[OnceLock<FrameRow<u8>>]>) -> Rows {
        Rows::U8(rows)
    }
    fn into_plane(v: Vec<u8>) -> Plane {
        Plane::U8(v)
    }
    fn alloc(n: usize) -> Vec<u8> {
        plane_allocator()(n)
    }
    #[inline(always)]
    fn mc_chroma_win(src: &[u8], ss: usize, fx: u32, fy: u32, bw: usize, bh: usize, out: &mut [u8], os: usize) {
        crate::inter::mc_chroma_win(src, ss, fx, fy, bw, bh, out, os);
    }
    #[inline(always)]
    fn load_vertical<const N: usize>(pix: &[u8], stride: usize, x: usize, y: usize) -> Option<[[i16; N]; 8]> {
        // Each line's eight samples as one little-endian word: byte k = sample k (p3 .. q3).
        let mut w = [0u64; N];
        for (i, w) in w.iter_mut().enumerate() {
            let o = (y + i) * stride + x - 4;
            let &b = pix.get(o..).and_then(|p| p.first_chunk::<8>())?;
            *w = u64::from_le_bytes(b);
        }
        let mut s = [[0i16; N]; 8];
        for (k, row) in s.iter_mut().enumerate() {
            for i in 0..N {
                row[i] = ((w[i] >> (8 * k)) & 0xff) as i16;
            }
        }
        Some(s)
    }
    #[inline(always)]
    fn store_vertical<const N: usize>(pix: &mut [u8], stride: usize, x: usize, y: usize, s: &[[i16; N]; 8]) {
        let mut w = [0u64; N];
        for (i, w) in w.iter_mut().enumerate() {
            *w = (0..8).fold(0u64, |acc, k| acc | ((s[k][i] as u8 as u64) << (8 * k)));
        }
        for (i, w) in w.iter().enumerate() {
            let o = (y + i) * stride + x - 4;
            if let Some(dst) = pix.get_mut(o..).and_then(|p| p.first_chunk_mut::<8>()) {
                *dst = w.to_le_bytes();
            }
        }
    }
}

impl Sample for u16 {
    const COEF_LIMIT: i32 = 1 << 17;
    #[inline(always)]
    fn max_value(bit_depth: u32) -> i32 {
        (1 << bit_depth.clamp(1, 16)) - 1
    }
    #[inline(always)]
    fn half(bit_depth: u32) -> i32 {
        1 << (bit_depth.clamp(1, 16) - 1)
    }
    #[inline(always)]
    fn to_i32(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn to_i16(self) -> i16 {
        self as i16
    }
    #[inline(always)]
    fn from_i32(v: i32) -> Self {
        v as u16
    }
    #[inline(always)]
    fn clip(v: i32, max: i32) -> Self {
        v.max(0).min(max) as u16
    }
    #[inline(always)]
    fn avg(a: u16, b: u16) -> u16 {
        ((a as u32 + b as u32 + 1) >> 1) as u16
    }
    #[inline(always)]
    fn rows(f: &Frame) -> Option<&[OnceLock<FrameRow<u16>>]> {
        match &f.rows {
            Rows::U16(r) => Some(r),
            Rows::U8(_) => None,
        }
    }
    fn wrap_rows(rows: Box<[OnceLock<FrameRow<u16>>]>) -> Rows {
        Rows::U16(rows)
    }
    fn into_plane(v: Vec<u16>) -> Plane {
        Plane::U16(v)
    }
    fn alloc(n: usize) -> Vec<u16> {
        Vec::with_capacity(n)
    }
    #[inline(always)]
    fn mc_chroma_win(src: &[u16], ss: usize, fx: u32, fy: u32, bw: usize, bh: usize, out: &mut [u16], os: usize) {
        crate::inter::mc_chroma_win16(src, ss, fx, fy, bw, bh, out, os);
    }
    #[inline(always)]
    fn load_vertical<const N: usize>(pix: &[u16], stride: usize, x: usize, y: usize) -> Option<[[i16; N]; 8]> {
        let mut s = [[0i16; N]; 8];
        for i in 0..N {
            let o = (y + i) * stride + x - 4;
            let line = pix.get(o..).and_then(|p| p.first_chunk::<8>())?;
            for k in 0..8 {
                s[k][i] = line[k] as i16;
            }
        }
        Some(s)
    }
    #[inline(always)]
    fn store_vertical<const N: usize>(pix: &mut [u16], stride: usize, x: usize, y: usize, s: &[[i16; N]; 8]) {
        for i in 0..N {
            let o = (y + i) * stride + x - 4;
            if let Some(line) = pix.get_mut(o..).and_then(|p| p.first_chunk_mut::<8>()) {
                for k in 0..8 {
                    line[k] = s[k][i] as u16;
                }
            }
        }
    }
}

/// Planar sample buffers of a whole picture (MB-aligned dimensions, stride == width).
#[derive(Clone)]
pub struct Planes<S> {
    pub y: Vec<S>,
    pub cb: Vec<S>,
    pub cr: Vec<S>,
    pub width: usize,
    pub height: usize,
    pub cwidth: usize,
    pub fmt: Format,
}

impl<S: Sample> Planes<S> {
    /// Black luma (0) and grey chroma: what a picture shows where no macroblock was decoded.
    pub fn new(width: usize, height: usize, fmt: Format) -> Self {
        let (cw, ch) = (width / 2, height >> fmt.sy());
        let gc = S::from_i32(S::half(fmt.bit_depth_c));
        Planes { y: vec![S::default(); width * height], cb: vec![gc; cw * ch], cr: vec![gc; cw * ch], width, height, cwidth: cw, fmt }
    }
    pub fn gray(width: usize, height: usize, fmt: Format) -> Self {
        let mut p = Self::new(width, height, fmt);
        p.y.fill(S::from_i32(S::half(fmt.bit_depth_y)));
        p
    }
}

/// Macroblock coding category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum MbKind {
    #[default]
    None,
    I4x4,
    I8x8,
    I16x16,
    IPcm,
    PSkip,
    BSkip,
    BDirect16x16,
    /// Any other P or B inter macroblock.
    Inter,
}

impl MbKind {
    #[inline]
    pub fn is_intra(self) -> bool {
        matches!(self, MbKind::I4x4 | MbKind::I8x8 | MbKind::I16x16 | MbKind::IPcm)
    }
}

/// Per-macroblock state kept for the whole picture (neighbour derivations, deblocking, co-located data).
#[derive(Clone, Copy)]
pub struct MbState {
    /// Slice number within the picture; `u32::MAX` = not (yet) decoded.
    pub slice_num: u32,
    pub kind: MbKind,
    pub transform_8x8: bool,
    /// coded_block_pattern: bits 0..3 luma, bits 4..5 chroma.
    pub cbp: u8,
    /// QPY (−QpBdOffsetY..=51; negative only above 8 bits).
    pub qp: i8,
    /// QPC for Cb / Cr (used by deblocking; −QpBdOffsetC..=39).
    pub qpc: [i8; 2],
    pub intra_chroma_mode: u8,
    /// Intra4x4PredMode / Intra8x8PredMode per 4x4 block (raster order).
    pub intra_modes: [u8; 16],
    /// CAVLC: TotalCoeff per luma 4x4 block (raster). CABAC: coded_block_flag.
    pub nnz: [u8; 16],
    /// Same for chroma AC blocks [Cb, Cr][raster 2x2 (4:2:0) or 2x4 (4:2:2)].
    pub nnz_c: [[u8; 8]; 2],
    /// CABAC coded_block_flag of DC blocks: bit0 luma (I16x16), bit1 Cb, bit2 Cr.
    pub cbf_dc: u8,
    /// Luma 4x4 blocks (raster bit) with non-zero coefficients, for deblocking bS = 2.
    pub nz_mask: u16,
    pub ref_idx: [[i8; 4]; 2],
    /// Motion vectors per 4x4 block (raster order).
    pub mv: [[[i16; 2]; 16]; 2],
    /// |mvd| per 4x4 block (raster), saturated, for CABAC context selection.
    pub mvd: [[[u8; 2]; 16]; 2],
    /// Bit per 8x8 block predicted in direct mode.
    pub direct8x8: u8,
}

impl MbState {
    #[inline]
    pub fn kind_is_skip(&self) -> bool {
        matches!(self.kind, MbKind::PSkip | MbKind::BSkip)
    }
}

impl Default for MbState {
    fn default() -> Self {
        MbState {
            slice_num: u32::MAX,
            kind: MbKind::None,
            transform_8x8: false,
            cbp: 0,
            qp: 0,
            qpc: [0; 2],
            intra_chroma_mode: 0,
            intra_modes: [2; 16],
            nnz: [0; 16],
            nnz_c: [[0; 8]; 2],
            cbf_dc: 0,
            nz_mask: 0,
            ref_idx: [[-1; 4]; 2],
            mv: [[[0; 2]; 16]; 2],
            mvd: [[[0; 2]; 16]; 2],
            direct8x8: 0,
        }
    }
}

/// Motion data of one published macroblock row, the co-located information of direct prediction.
pub struct RowMotion {
    /// Per 4x4 block: index mb_x * 16 + raster.
    pub mv: [Box<[[i16; 2]]>; 2],
    /// Per 8x8 block: index mb_x * 4 + b8.
    pub ref_idx: [Box<[i8]>; 2],
    /// Unique id of the referenced picture per 8x8 block (u32::MAX = none).
    pub ref_id: [Box<[u32]>; 2],
    pub intra: Box<[bool]>,
}

impl RowMotion {
    #[cfg(not(feature = "threads"))]
    /// No motion: every macroblock intra.
    fn intra(mb_w: usize) -> Self {
        let per_list = |n: usize| -> [Box<[_]>; 2] { [vec![u32::MAX; n].into(), vec![u32::MAX; n].into()] };
        RowMotion {
            mv: [vec![[0; 2]; mb_w * 16].into(), vec![[0; 2]; mb_w * 16].into()],
            ref_idx: [vec![-1; mb_w * 4].into(), vec![-1; mb_w * 4].into()],
            ref_id: per_list(mb_w * 4),
            intra: vec![true; mb_w].into(),
        }
    }
}

/// One published macroblock row of a decoded frame: final (deblocked) samples plus the motion data
/// needed as co-located information for direct prediction.
pub struct FrameRow<S> {
    /// 16 luma lines.
    pub y: Box<[S]>,
    /// MbHeightC lines (8 for 4:2:0, 16 for 4:2:2) per chroma component.
    pub cb: Box<[S]>,
    pub cr: Box<[S]>,
    pub motion: RowMotion,
}

/// The macroblock rows of a frame, in its storage type.
pub enum Rows {
    U8(Box<[OnceLock<FrameRow<u8>>]>),
    U16(Box<[OnceLock<FrameRow<u16>>]>),
}

impl Rows {
    fn len(&self) -> usize {
        match self {
            Rows::U8(r) => r.len(),
            Rows::U16(r) => r.len(),
        }
    }
}

/// A decoded (or in-progress) frame as stored in the DPB and referenced by later pictures.
///
/// Rows are published in order by the thread decoding the picture; readers block (per row) until the
/// data they need is available, which lets several pictures decode concurrently.
pub struct Frame {
    pub id: u32,
    pub poc: i32,
    pub width: usize,
    pub height: usize,
    pub cwidth: usize,
    pub cheight: usize,
    pub mb_w: usize,
    pub fmt: Format,
    rows: Rows,
}

impl Frame {
    pub fn new(id: u32, poc: i32, mb_w: usize, mb_h: usize, fmt: Format) -> Self {
        fn cells<S: Sample>(n: usize) -> Rows {
            S::wrap_rows((0..n).map(|_| OnceLock::new()).collect())
        }
        let rows = if fmt.stores_u8() { cells::<u8>(mb_h) } else { cells::<u16>(mb_h) };
        Frame { id, poc, width: mb_w * 16, height: mb_h * 16, cwidth: mb_w * 8, cheight: mb_h * fmt.mb_height_c(), mb_w, fmt, rows }
    }

    pub fn mb_h(&self) -> usize {
        self.rows.len()
    }

    /// Publish MB row `r` (ignored if already published, or if the frame stores another type).
    pub fn publish<S: Sample>(&self, r: usize, row: FrameRow<S>) {
        if let Some(cell) = S::rows(self).and_then(|rows| rows.get(r)) {
            let _ = cell.set(row);
        }
    }

    pub fn is_published(&self, r: usize) -> bool {
        match &self.rows {
            Rows::U8(rows) => rows.get(r).is_some_and(|c| c.get().is_some()),
            Rows::U16(rows) => rows.get(r).is_some_and(|c| c.get().is_some()),
        }
    }

    #[cfg(test)]
    /// MB row `r` of a frame stored as `S`, waiting until it has been published (None for a
    /// frame stored with the other type, or past the last row).
    #[inline]
    pub fn row<S: Sample>(&self, r: usize) -> Option<&FrameRow<S>> {
        Self::cell_row(S::rows(self)?.get(r)?, self)
    }

    #[inline(always)]
    fn cell_row<'a, S: Sample>(cell: &'a OnceLock<FrameRow<S>>, f: &Frame) -> Option<&'a FrameRow<S>> {
        #[cfg(feature = "threads")]
        {
            let _ = f;
            Some(cell.wait())
        }
        #[cfg(not(feature = "threads"))]
        {
            // Rows are decoded in order here, so a row is published before it is read; one that
            // is not reads as gray with no motion instead of panicking.
            Some(cell.get_or_init(|| f.gray_row()))
        }
    }

    /// The motion data of MB row `r`, waiting until it has been published.
    pub fn motion(&self, r: usize) -> Option<&RowMotion> {
        match &self.rows {
            Rows::U8(rows) => Self::cell_row(rows.get(r)?, self).map(|row| &row.motion),
            Rows::U16(rows) => Self::cell_row(rows.get(r)?, self).map(|row| &row.motion),
        }
    }

    /// A gray intra row with no motion, what "non-existing" frames publish.
    #[cfg(not(feature = "threads"))]
    fn gray_row<S: Sample>(&self) -> FrameRow<S> {
        let (gy, gc) = (S::from_i32(S::half(self.fmt.bit_depth_y)), S::from_i32(S::half(self.fmt.bit_depth_c)));
        let lines_c = self.fmt.mb_height_c();
        FrameRow {
            y: vec![gy; self.width * 16].into(),
            cb: vec![gc; self.cwidth * lines_c].into(),
            cr: vec![gc; self.cwidth * lines_c].into(),
            motion: RowMotion::intra(self.mb_w),
        }
    }

    /// Wait until the whole frame is available.
    pub fn wait_complete(&self) {
        for r in 0..self.mb_h() {
            self.motion(r);
        }
    }

    /// Build a published row from a contiguous plane set (rows of MB row `r`) and motion data.
    pub fn make_row<S: Sample>(planes: &Planes<S>, r: usize, mbs: &[MbState], ref_ids: &dyn Fn(&MbState, usize, i8) -> u32) -> FrameRow<S> {
        let w = planes.width;
        let cw = planes.cwidth;
        let lc = planes.fmt.mb_height_c();
        let mb_w = w / 16;
        let lines = |p: &[S], n: usize, width: usize| -> Box<[S]> {
            match p.get(r * n * width..(r + 1) * n * width) {
                Some(s) => s.into(),
                None => vec![S::default(); n * width].into(),
            }
        };
        let y = lines(&planes.y, 16, w);
        let cb = lines(&planes.cb, lc, cw);
        let cr = lines(&planes.cr, lc, cw);
        let row_mbs = mbs.get(r * mb_w..(r + 1) * mb_w).unwrap_or_default();
        let mut mv = [vec![[0i16; 2]; mb_w * 16], vec![[0i16; 2]; mb_w * 16]];
        let mut ref_idx = [vec![-1i8; mb_w * 4], vec![-1i8; mb_w * 4]];
        let mut ref_id = [vec![u32::MAX; mb_w * 4], vec![u32::MAX; mb_w * 4]];
        let mut intra = vec![true; mb_w];
        for (x, st) in row_mbs.iter().enumerate() {
            if st.slice_num == u32::MAX {
                continue;
            }
            intra[x] = st.kind.is_intra();
            for l in 0..2 {
                mv[l][x * 16..x * 16 + 16].copy_from_slice(&st.mv[l]);
                for b in 0..4 {
                    let ri = st.ref_idx[l][b];
                    ref_idx[l][x * 4 + b] = ri;
                    if ri >= 0 {
                        ref_id[l][x * 4 + b] = ref_ids(st, l, ri);
                    }
                }
            }
        }
        let [mv0, mv1] = mv;
        let [ri0, ri1] = ref_idx;
        let [id0, id1] = ref_id;
        let motion = RowMotion { mv: [mv0.into(), mv1.into()], ref_idx: [ri0.into(), ri1.into()], ref_id: [id0.into(), id1.into()], intra: intra.into() };
        FrameRow { y, cb, cr, motion }
    }

    /// A frame with every row published from `planes` and no motion (intra), e.g. "non-existing"
    /// frames inferred for frame_num gaps.
    pub fn from_planes<S: Sample>(id: u32, poc: i32, planes: &Planes<S>) -> Self {
        let (mb_w, mb_h) = (planes.width / 16, planes.height / 16);
        let f = Frame::new(id, poc, mb_w, mb_h, planes.fmt);
        let blank = vec![MbState::default(); mb_w * mb_h];
        for r in 0..mb_h {
            f.publish(r, Frame::make_row(planes, r, &blank, &|_, _, _| u32::MAX));
        }
        f
    }

    /// A copy of the whole (uncropped) frame as planes (waits for completion); None when it is not
    /// stored as `S`.
    pub fn to_planes<S: Sample>(&self) -> Option<Planes<S>> {
        let rows = S::rows(self)?;
        let mut p = Planes { y: Vec::new(), cb: Vec::new(), cr: Vec::new(), width: self.width, height: self.height, cwidth: self.cwidth, fmt: self.fmt };
        for cell in rows.iter() {
            let row = Self::cell_row(cell, self)?;
            p.y.extend_from_slice(&row.y);
            p.cb.extend_from_slice(&row.cb);
            p.cr.extend_from_slice(&row.cr);
        }
        Some(p)
    }

    /// Copy the w x h luma window at (x0, y0) (clamped to the picture) into `out` (stride `os`).
    /// A frame stored as another type (never a reference of a conforming stream) reads as grey.
    #[inline(always)]
    pub fn luma_window<S: Sample>(&self, x0: i32, y0: i32, w: usize, h: usize, out: &mut [S], os: usize) {
        let (width, height) = (self.width, self.height);
        let Some(rows) = S::rows(self) else {
            return fill_window(out, os, w, h, S::from_i32(S::half(self.fmt.bit_depth_y)));
        };
        for r in 0..h {
            let yy = (y0 + r as i32).clamp(0, height as i32 - 1) as usize;
            let Some(row) = rows.get(yy >> 4).and_then(|c| Self::cell_row(c, self)) else { continue };
            let Some(line) = row.y.get((yy & 15) * width..(yy & 15) * width + width) else { continue };
            copy_line::<24, S>(line, x0, &mut out[r * os..], w);
        }
    }

    /// Copy the w x h window of chroma component `c` at (x0, y0) (clamped) into `out`.
    #[inline(always)]
    pub fn chroma_window<S: Sample>(&self, c: usize, x0: i32, y0: i32, w: usize, h: usize, out: &mut [S], os: usize) {
        let (cw, ch) = (self.cwidth, self.cheight);
        let Some(rows) = S::rows(self) else {
            return fill_window(out, os, w, h, S::from_i32(S::half(self.fmt.bit_depth_c)));
        };
        let shift = self.fmt.log2_mb_height_c();
        let mask = (1 << shift) - 1;
        for r in 0..h {
            let yy = (y0 + r as i32).clamp(0, ch as i32 - 1) as usize;
            let Some(row) = rows.get(yy >> shift).and_then(|c| Self::cell_row(c, self)) else { continue };
            let plane = if c == 0 { &row.cb } else { &row.cr };
            let Some(line) = plane.get((yy & mask) * cw..(yy & mask) * cw + cw) else { continue };
            copy_line::<16, S>(line, x0, &mut out[r * os..], w);
        }
    }

    /// Crop and copy the frame into planar buffers (waits for completion). The chroma rectangle
    /// follows the sampling: half the width, and half (4:2:0) or all (4:2:2) of the lines.
    pub fn copy_cropped(&self, crop: (usize, usize, usize, usize)) -> (Plane, Plane, Plane) {
        fn planes<S: Sample>(f: &Frame, rows: &[OnceLock<FrameRow<S>>], crop: (usize, usize, usize, usize)) -> (Plane, Plane, Plane) {
            let (cx, cy, cw, ch) = crop;
            let mut y = S::alloc(cw * ch);
            for r in cy..cy + ch {
                if let Some(row) = rows.get(r >> 4).and_then(|c| Frame::cell_row(c, f))
                    && let Some(line) = row.y.get((r & 15) * f.width + cx..(r & 15) * f.width + cx + cw)
                {
                    y.extend_from_slice(line);
                }
            }
            let sy = f.fmt.sy();
            let (ccx, ccy, ccw, cch) = (cx / 2, cy >> sy, cw.div_ceil(2), if sy == 1 { ch.div_ceil(2) } else { ch });
            let (shift, mask) = (f.fmt.log2_mb_height_c(), f.fmt.mb_height_c() - 1);
            let mut u = S::alloc(ccw * cch);
            let mut v = S::alloc(ccw * cch);
            for r in ccy..ccy + cch {
                if let Some(row) = rows.get(r >> shift).and_then(|c| Frame::cell_row(c, f)) {
                    let o = (r & mask) * f.cwidth + ccx;
                    if let (Some(a), Some(b)) = (row.cb.get(o..o + ccw), row.cr.get(o..o + ccw)) {
                        u.extend_from_slice(a);
                        v.extend_from_slice(b);
                    }
                }
            }
            (S::into_plane(y), S::into_plane(u), S::into_plane(v))
        }
        match &self.rows {
            Rows::U8(rows) => planes::<u8>(self, rows, crop),
            Rows::U16(rows) => planes::<u16>(self, rows, crop),
        }
    }
}

fn fill_window<S: Sample>(out: &mut [S], os: usize, w: usize, h: usize, v: S) {
    for r in 0..h {
        if let Some(line) = out.get_mut(r * os..r * os + w) {
            line.fill(v);
        }
    }
}

static PLANE_ALLOCATOR: std::sync::OnceLock<fn(usize) -> Vec<u8>> = std::sync::OnceLock::new();

/// Where 8-bit output planes come from: an empty buffer with room for the given number of
/// samples. The host sets it once to recycle the planes of pictures it is done with; the first
/// call wins.
pub fn set_plane_allocator(alloc: fn(usize) -> Vec<u8>) {
    let _ = PLANE_ALLOCATOR.set(alloc);
}

fn plane_allocator() -> fn(usize) -> Vec<u8> {
    PLANE_ALLOCATOR.get().copied().unwrap_or(Vec::with_capacity)
}

/// Copy `w` samples of `line` starting at x0 into `out`; when possible a fixed-size block of `N`
/// samples is copied instead (cheaper than a variable-length copy; `out` must then have room for `N`).
#[inline(always)]
fn copy_line<const N: usize, S: Copy>(line: &[S], x0: i32, out: &mut [S], w: usize) {
    if w <= N
        && x0 >= 0
        && let Some(src) = line.get(x0 as usize..).and_then(|l| l.first_chunk::<N>())
        && let Some(dst) = out.first_chunk_mut::<N>()
    {
        *dst = *src;
    } else {
        copy_clamped(line, x0, &mut out[..w]);
    }
}

/// Copy `out.len()` samples of `line` starting at x0, replicating edge samples outside the line.
#[inline(always)]
fn copy_clamped<S: Copy>(line: &[S], x0: i32, out: &mut [S]) {
    let w = out.len();
    if x0 >= 0 && x0 as usize + w <= line.len() {
        out.copy_from_slice(&line[x0 as usize..x0 as usize + w]);
    } else {
        let max = line.len() as i32 - 1;
        for (c, o) in out.iter_mut().enumerate() {
            *o = line[(x0 + c as i32).clamp(0, max) as usize];
        }
    }
}

pub type FrameRef = Arc<Frame>;

/// An entry of RefPicList0/1 for the current slice.
#[derive(Clone)]
pub struct RefPic {
    pub frame: FrameRef,
    pub long_term: bool,
}

impl RefPic {
    pub fn poc(&self) -> i32 {
        self.frame.poc
    }
    pub fn id(&self) -> u32 {
        self.frame.id
    }
}

#[cfg(test)]
mod format_tests {
    use super::*;

    /// The picture-size budget counts every plane and the bytes per sample: a 10-bit 4:2:2
    /// picture takes four times the bytes of its luma samples, an 8-bit 4:2:0 one 1.5 times.
    #[test]
    fn picture_bytes_count_every_plane_and_the_sample_size() {
        let f = |chroma, bits| Format { chroma, bit_depth_y: bits, bit_depth_c: bits };
        assert_eq!(f(ChromaFormat::C420, 8).picture_bytes(1000), 1500);
        assert_eq!(f(ChromaFormat::C422, 8).picture_bytes(1000), 2000);
        assert_eq!(f(ChromaFormat::C420, 10).picture_bytes(1000), 3000);
        assert_eq!(f(ChromaFormat::C422, 10).picture_bytes(1000), 4000);
        // 8-bit luma with 10-bit chroma is stored in 16 bits
        assert_eq!(Format { chroma: ChromaFormat::C420, bit_depth_y: 8, bit_depth_c: 10 }.picture_bytes(1000), 3000);
        assert_eq!(f(ChromaFormat::C422, 10).picture_bytes(u64::MAX), u64::MAX);
    }

    /// 4:2:2 frames keep every chroma line: the window reads and the crop follow MbHeightC.
    #[test]
    fn a_422_frame_crops_chroma_by_lines() {
        let fmt = Format { chroma: ChromaFormat::C422, bit_depth_y: 10, bit_depth_c: 10 };
        let mut p = Planes::<u16>::new(32, 32, fmt);
        assert_eq!((p.cwidth, p.cb.len()), (16, 16 * 32));
        for (i, s) in p.cb.iter_mut().enumerate() {
            *s = (i / 16) as u16; // chroma line number
        }
        let f = Frame::from_planes(1, 0, &p);
        assert_eq!(f.cheight, 32);
        let (y, u, _) = f.copy_cropped((0, 3, 30, 27));
        assert_eq!((y.len(), u.len()), (30 * 27, 15 * 27));
        assert_eq!(u.as_u16().map(|u| (u[0], u[15 * 26])), Some((3, 29)), "chroma starts at line 3, one per luma line");
        let mut w = [0u16; 16 * 4];
        f.chroma_window(0, 0, 17, 4, 3, &mut w, 16);
        assert_eq!((w[0], w[16], w[32]), (17, 18, 19));
        // a type mismatch reads grey instead of failing
        let mut g = [0u8; 16];
        f.chroma_window(0, 0, 0, 4, 1, &mut g, 16);
        assert_eq!(g[0], 128);
        assert!(f.row::<u8>(0).is_none() && f.row::<u16>(0).is_some() && f.row::<u16>(2).is_none());
    }
}

#[cfg(all(test, not(feature = "threads")))]
mod row_tests {
    use super::*;

    /// Without threads a row read before it was published is gray, not a panic.
    #[test]
    fn an_unpublished_row_reads_as_gray_without_threads() {
        let f = Frame::new(0, 0, 2, 2, Format::YUV420_8);
        let row = f.row::<u8>(1).unwrap();
        assert!(row.y.iter().chain(row.cb.iter()).chain(row.cr.iter()).all(|&s| s == 128));
        let m = &row.motion;
        assert_eq!((row.y.len(), row.cb.len(), m.mv[0].len(), m.ref_idx[1].len(), m.intra.len()), (32 * 16, 16 * 8, 32, 8, 2));
        assert!(m.intra.iter().all(|&i| i) && m.ref_id[0].iter().all(|&id| id == u32::MAX));
        let f = Frame::new(0, 0, 2, 2, Format { chroma: ChromaFormat::C422, bit_depth_y: 10, bit_depth_c: 9 });
        let row = f.row::<u16>(1).unwrap();
        assert_eq!((row.y[0], row.cb[0], row.cb.len()), (512, 256, 16 * 16));
    }
}
