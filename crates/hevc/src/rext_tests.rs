//! Range-extension coding tools, which libx265 never writes. A randomised encoder writes streams
//! that use them: transform-skip rotation, contexts and block sizes up to 32x32, implicit and
//! explicit residual DPCM, intra smoothing off, persistent Rice adaptation (with WPP and several
//! slices), cross-component prediction, chroma QP offset lists, SAO offset scaling and
//! high-precision weighted-prediction offsets, in 4:2:0, 4:2:2, 4:4:4 and monochrome at 8 to 12
//! bits. Block structure, prediction modes, motion, cbfs and coefficients are random; the encoder
//! derives every context the way the decoder does (9.3.4.2) and computes no pictures. Each
//! stream is decoded by ffmpeg, the oracle, and by this decoder single- and multi-threaded, and
//! every plane of every picture must agree sample for sample (a stream that ffmpeg reports
//! errors on fails the test, so a mistake in the encoder cannot pass unnoticed).

use crate::cabac::{LOG2_RES_SCALE_ABS, RES_SCALE_SIGN, STAT_COEFF, init_contexts};
use crate::slicedec::sig_ctx_table;
use crate::spec_tables::*;
use crate::synth_tests::{Enc, nal};
use crate::tables::{INTRA_CHROMA_MODE_422, LEVEL_SCALE, SCAN_4, qpc, scan_order};
use crate::transform;
use crate::{Decoder, Picture};
use filmcraft_bitstream::{BitWriter, escape_rbsp};

const LOG2_CTB: u32 = 5;
const CTB: usize = 1 << LOG2_CTB;
const LOG2_MIN_CB: u32 = 3;
const LOG2_MAX_TB: u32 = 5;
const MAX_TH_DEPTH: u32 = 2;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// 0..n
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n.max(1) as u64) as u32
    }
    /// true with probability `percent` / 100
    fn chance(&mut self, percent: u32) -> bool {
        self.below(100) < percent
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        lo + self.below((hi - lo + 1) as u32) as i32
    }
}

/// What a test stream uses.
#[derive(Clone, Debug)]
struct Cfg {
    name: &'static str,
    /// chroma_format_idc 0..=3.
    chroma: u32,
    bd_y: u32,
    bd_c: u32,
    width: usize,
    height: usize,
    /// Pictures: an IDR, then P pictures predicting from the previous one (intra pictures with
    /// `intra_only`).
    pictures: usize,
    intra_only: bool,
    seed: u64,
    init_qp: i32,
    // SPS range extension
    rotation: bool,
    ts_context: bool,
    implicit_rdpcm: bool,
    explicit_rdpcm: bool,
    smoothing_off: bool,
    high_precision: bool,
    persistent_rice: bool,
    /// The tools this decoder refuses (for the refusal tests).
    extended_precision: bool,
    bypass_alignment: bool,
    // SPS / PPS
    strong_smoothing: bool,
    sao: bool,
    transform_skip: bool,
    log2_max_ts: u32,
    bypass: bool,
    sign_hiding: bool,
    ccp: bool,
    /// (diff_cu_chroma_qp_offset_depth, (cb, cr) list)
    chroma_qp_offsets: Option<(u32, Vec<(i32, i32)>)>,
    sao_scale: (u32, u32),
    cb_cr_qp_offset: (i32, i32),
    weighted: bool,
    wpp: bool,
    /// CTB rows per slice (0: one slice per picture).
    slice_rows: usize,
    /// Large coefficient levels (Rice parameters past 4).
    big_levels: bool,
}

impl Cfg {
    /// Every tool off, 4:4:4 8-bit, 120 x 72 (partial CTBs at the right and bottom).
    fn base(name: &'static str) -> Self {
        Cfg {
            name,
            chroma: 3,
            bd_y: 8,
            bd_c: 8,
            width: 120,
            height: 72,
            pictures: 3,
            intra_only: false,
            seed: 0x9e37_79b9_7f4a_7c15,
            init_qp: 30,
            rotation: false,
            ts_context: false,
            implicit_rdpcm: false,
            explicit_rdpcm: false,
            smoothing_off: false,
            high_precision: false,
            persistent_rice: false,
            extended_precision: false,
            bypass_alignment: false,
            strong_smoothing: true,
            sao: true,
            transform_skip: true,
            log2_max_ts: 2,
            bypass: true,
            sign_hiding: true,
            ccp: false,
            chroma_qp_offsets: None,
            sao_scale: (0, 0),
            cb_cr_qp_offset: (0, 0),
            weighted: false,
            wpp: false,
            slice_rows: 0,
            big_levels: false,
        }
    }
    /// Every range-extension tool the chroma format and depth allow, without lossless CUs.
    fn all_tools(name: &'static str, chroma: u32, bd: u32) -> Self {
        Cfg {
            chroma,
            bd_y: bd,
            bd_c: bd,
            rotation: true,
            ts_context: true,
            implicit_rdpcm: true,
            explicit_rdpcm: true,
            // (high-precision weighted prediction offsets are refused above 8 bits)
            high_precision: bd == 8,
            persistent_rice: true,
            log2_max_ts: 5,
            bypass: false,
            ccp: chroma == 3,
            chroma_qp_offsets: (chroma != 0).then(|| (1, vec![(-5, 7)])),
            sao_scale: (bd.saturating_sub(10), bd.saturating_sub(10)),
            cb_cr_qp_offset: (2, -3),
            weighted: true,
            big_levels: true,
            ..Cfg::base(name)
        }
    }
    /// The range-extension tools that combine with lossless (transquant bypass) CUs (residual
    /// rotation and implicit RDPCM do not: refused).
    fn lossless_tools(name: &'static str, chroma: u32, bd: u32) -> Self {
        Cfg {
            rotation: false,
            implicit_rdpcm: false,
            bypass: true,
            chroma_qp_offsets: (chroma != 0).then(|| (0, vec![(4, -6)])),
            ..Cfg::all_tools(name, chroma, bd)
        }
    }
    fn has_chroma(&self) -> bool {
        self.chroma != 0
    }
    /// log2 (SubWidthC, SubHeightC).
    fn shifts(&self) -> (u32, u32) {
        match self.chroma {
            2 => (1, 0),
            3 => (0, 0),
            _ => (1, 1),
        }
    }
    fn wctb(&self) -> usize {
        self.width.div_ceil(CTB)
    }
    fn hctb(&self) -> usize {
        self.height.div_ceil(CTB)
    }
    fn pix_fmt(&self) -> String {
        let base = match self.chroma {
            0 => "gray",
            1 => "yuv420p",
            2 => "yuv422p",
            _ => "yuv444p",
        };
        if self.bd_y == 8 { base.to_string() } else { format!("{base}{}le", self.bd_y) }
    }
}

/// How often the encoder used each tool (so a test can check its stream exercises them).
#[derive(Default, Debug)]
struct Usage {
    transform_skip: usize,
    bypass: usize,
    rotated: usize,
    implicit_rdpcm: usize,
    explicit_rdpcm: usize,
    ccp: usize,
    ccp_without_chroma_residual: usize,
    chroma_qp_offset: usize,
    large_ts: usize,
    rice_above_4: usize,
    intra_4x4_chroma: usize,
    weighted: usize,
}

fn ptl(w: &mut BitWriter) {
    w.write_bits(0, 2); // profile_space
    w.write_bits(0, 1); // tier
    w.write_bits(4, 5); // format range extensions
    w.write_bits(1 << (31 - 4), 32); // compatibility flag 4
    w.write_bits(0b1001, 4); // progressive, interlaced, non_packed, frame_only
    w.write_bits(0, 32); // RExt constraint flags (none) and reserved bits
    w.write_bits(0, 11);
    w.write_bits(0, 1);
    w.write_bits(120, 8); // level 4
}

fn vps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(0, 4);
    w.write_bits(3, 2);
    w.write_bits(0, 6);
    w.write_bits(0, 3);
    w.write_bits(1, 1);
    w.write_bits(0xffff, 16);
    ptl(&mut w);
    w.write_bits(1, 1);
    w.write_ue(4);
    w.write_ue(0);
    w.write_ue(0);
    w.write_bits(0, 6);
    w.write_ue(0);
    w.write_bits(0, 1);
    w.write_bits(0, 1);
    w.rbsp_trailing();
    nal(32, &w.finish())
}

fn sps(c: &Cfg) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(0, 4);
    w.write_bits(0, 3);
    w.write_bits(1, 1);
    ptl(&mut w);
    w.write_ue(0); // sps id
    w.write_ue(c.chroma);
    if c.chroma == 3 {
        w.write_bit(false); // separate_colour_plane_flag
    }
    w.write_ue(c.width as u32);
    w.write_ue(c.height as u32);
    w.write_bit(false); // conformance window
    w.write_ue(c.bd_y - 8);
    w.write_ue(c.bd_c - 8);
    w.write_ue(4); // log2_max_poc_lsb = 8
    w.write_bit(true);
    w.write_ue(3); // max_dec_pic_buffering_minus1
    w.write_ue(0);
    w.write_ue(0);
    w.write_ue(LOG2_MIN_CB - 3);
    w.write_ue(LOG2_CTB - LOG2_MIN_CB);
    w.write_ue(0); // min TB 4
    w.write_ue(LOG2_MAX_TB - 2);
    w.write_ue(MAX_TH_DEPTH); // inter
    w.write_ue(MAX_TH_DEPTH); // intra
    w.write_bit(false); // scaling lists
    w.write_bit(false); // amp
    w.write_bit(c.sao);
    w.write_bit(false); // pcm
    w.write_ue(0); // num_short_term_ref_pic_sets
    w.write_bit(false); // long-term
    w.write_bit(false); // temporal mvp
    w.write_bit(c.strong_smoothing);
    w.write_bit(false); // vui
    w.write_bit(true); // sps_extension_present_flag
    w.write_bits(0x80, 8); // sps_range_extension_flag only
    for f in [
        c.rotation,
        c.ts_context,
        c.implicit_rdpcm,
        c.explicit_rdpcm,
        c.extended_precision,
        c.smoothing_off,
        c.high_precision,
        c.persistent_rice,
        c.bypass_alignment,
    ] {
        w.write_bit(f);
    }
    w.rbsp_trailing();
    nal(33, &w.finish())
}

fn pps(c: &Cfg) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_ue(0);
    w.write_ue(0);
    w.write_bit(false); // dependent slices
    w.write_bit(false); // output flag
    w.write_bits(0, 3);
    w.write_bit(c.sign_hiding);
    w.write_bit(false); // cabac_init_present
    w.write_ue(0);
    w.write_ue(0);
    w.write_se(c.init_qp - 26);
    w.write_bit(false); // constrained intra
    w.write_bit(c.transform_skip);
    w.write_bit(false); // cu_qp_delta
    w.write_se(c.cb_cr_qp_offset.0);
    w.write_se(c.cb_cr_qp_offset.1);
    w.write_bit(false); // slice chroma qp offsets
    w.write_bit(c.weighted);
    w.write_bit(false); // weighted bipred
    w.write_bit(c.bypass);
    w.write_bit(false); // tiles
    w.write_bit(c.wpp);
    w.write_bit(true); // loop filter across slices
    w.write_bit(false); // deblocking control
    w.write_bit(false); // scaling list
    w.write_bit(false); // lists modification
    w.write_ue(0); // log2_parallel_merge_level - 2
    w.write_bit(false); // slice header extension
    w.write_bit(true); // pps_extension_present_flag
    w.write_bits(0x80, 8); // pps_range_extension_flag only
    if c.transform_skip {
        w.write_ue(c.log2_max_ts - 2);
    }
    w.write_bit(c.ccp);
    w.write_bit(c.chroma_qp_offsets.is_some());
    if let Some((depth, list)) = &c.chroma_qp_offsets {
        w.write_ue(*depth);
        w.write_ue(list.len() as u32 - 1);
        for &(cb, cr) in list {
            w.write_se(cb);
            w.write_se(cr);
        }
    }
    w.write_ue(c.sao_scale.0);
    w.write_ue(c.sao_scale.1);
    w.rbsp_trailing();
    nal(34, &w.finish())
}

/// The encoder state of one picture.
struct PicEnc<'c> {
    c: &'c Cfg,
    rng: Rng,
    usage: Usage,
    intra_slice: bool,
    /// Per 4x4 block: CU depth, luma intra mode (DC for inter blocks), skip flag, decoded.
    depth: Vec<u8>,
    ipm: Vec<u8>,
    skip: Vec<bool>,
    w4: usize,
    /// Slice index of each CTB (raster) for availability.
    ctb_slice: Vec<usize>,
    slice: usize,
    // current CU
    cu_bypass: bool,
    cu_intra: bool,
    cu_log2: u32,
    intra_split: bool,
    chroma_modes: [u32; 4],
    chroma_dm: [bool; 4],
    max_trafo_depth: u32,
    cqo_coded: bool,
    log2_min_cqo: u32,
    cqo_enabled: bool,
    /// Per CTB: chroma SAO in effect (merges included), and that of the current CTB.
    sao_chroma_on: Vec<bool>,
    ctb_chroma_sao: bool,
    /// SliceQpY and CuQpOffsetCb / Cr (for the residual bound of the random coefficients).
    slice_qp: i32,
    cu_qp_offset: [i32; 2],
}

impl PicEnc<'_> {
    fn blk(&self, x: usize, y: usize) -> usize {
        (y >> 2) * self.w4 + (x >> 2)
    }
    /// Availability of luma location (xn, yn) for the current block (6.4.1): inside the picture
    /// and the slice (left and above neighbours are always earlier in z-order).
    fn avail(&self, xn: i32, yn: i32) -> bool {
        if xn < 0 || yn < 0 || xn as usize >= self.c.width || yn as usize >= self.c.height {
            return false;
        }
        let rs = (yn as usize >> LOG2_CTB) * self.c.wctb() + (xn as usize >> LOG2_CTB);
        self.ctb_slice[rs] == self.slice
    }

    /// sao() (7.3.8.3) of CTB (rx, ry) for the slice's luma / chroma SAO flags.
    fn sao(&mut self, e: &mut Enc, rx: usize, ry: usize, (luma, chroma): (bool, bool)) {
        let wctb = self.c.wctb();
        let rs = ry * wctb + rx;
        if rx > 0 && self.ctb_slice[rs - 1] == self.slice {
            let merge = self.rng.chance(20);
            e.decision(SAO_MERGE, merge as u32);
            if merge {
                self.sao_chroma_on[rs] = self.sao_chroma_on[rs - 1];
                return;
            }
        }
        if ry > 0 && self.ctb_slice[rs - wctb] == self.slice {
            let merge = self.rng.chance(20);
            e.decision(SAO_MERGE, merge as u32);
            if merge {
                self.sao_chroma_on[rs] = self.sao_chroma_on[rs - wctb];
                return;
            }
        }
        let mut ty = [0u32; 3];
        for c in 0..3 {
            if !(if c == 0 { luma } else { chroma }) {
                continue;
            }
            if c == 2 {
                ty[2] = ty[1];
            } else {
                ty[c] = self.rng.below(3);
                e.decision(SAO_TYPE, (ty[c] != 0) as u32);
                if ty[c] != 0 {
                    e.bypass((ty[c] == 2) as u32);
                }
            }
            self.sao_chroma_on[rs] |= c > 0 && ty[c] != 0;
            if ty[c] == 0 {
                continue;
            }
            let bd = if c == 0 { self.c.bd_y } else { self.c.bd_c };
            let cmax = (1u32 << (bd.min(10) - 5)) - 1;
            let mut abs = [0u32; 4];
            for a in abs.iter_mut() {
                *a = if self.rng.chance(30) { 0 } else { self.rng.below(cmax + 1) };
                for _ in 0..*a {
                    e.bypass(1);
                }
                if *a < cmax {
                    e.bypass(0);
                }
            }
            if ty[c] == 1 {
                for &a in &abs {
                    if a != 0 {
                        e.bypass(self.rng.below(2));
                    }
                }
                e.bypass_bits(self.rng.below(32), 5);
            } else if c < 2 {
                e.bypass_bits(self.rng.below(4), 2);
            }
        }
    }

    fn quadtree(&mut self, e: &mut Enc, x0: usize, y0: usize, log2: u32, depth: u32) {
        let size = 1usize << log2;
        let (w, h) = (self.c.width, self.c.height);
        let split = if x0 + size <= w && y0 + size <= h && log2 > LOG2_MIN_CB {
            let mut inc = 0;
            if self.avail(x0 as i32 - 1, y0 as i32) && self.depth[self.blk(x0 - 1, y0)] as u32 > depth {
                inc += 1;
            }
            if self.avail(x0 as i32, y0 as i32 - 1) && self.depth[self.blk(x0, y0 - 1)] as u32 > depth {
                inc += 1;
            }
            let split = self.rng.chance(if log2 == LOG2_CTB { 60 } else { 45 });
            e.decision(SPLIT_CU + inc, split as u32);
            split
        } else {
            log2 > LOG2_MIN_CB
        };
        if self.cqo_enabled && log2 >= self.log2_min_cqo {
            self.cqo_coded = false;
        }
        if split {
            let half = size / 2;
            for (dx, dy) in [(0, 0), (half, 0), (0, half), (half, half)] {
                if x0 + dx < w && y0 + dy < h {
                    self.quadtree(e, x0 + dx, y0 + dy, log2 - 1, depth + 1);
                }
            }
        } else {
            self.coding_unit(e, x0, y0, log2, depth);
        }
    }

    /// Set the per-4x4 state of the CU's blocks inside the picture.
    fn fill(&mut self, x0: usize, y0: usize, n: usize, f: impl Fn(&mut Self, usize)) {
        for y in (y0..(y0 + n).min(self.c.height)).step_by(4) {
            for x in (x0..(x0 + n).min(self.c.width)).step_by(4) {
                let i = self.blk(x, y);
                f(self, i);
            }
        }
    }

    fn coding_unit(&mut self, e: &mut Enc, x0: usize, y0: usize, log2: u32, depth: u32) {
        let n = 1usize << log2;
        self.cu_log2 = log2;
        self.fill(x0, y0, n, |s, i| {
            s.depth[i] = depth as u8;
            s.skip[i] = false;
            s.ipm[i] = 1;
        });
        // (ffmpeg 9 applies chroma SAO to some lossless CUs in 4:2:0 and 4:2:2, which the
        // specification exempts, as this decoder does: no lossless CUs in CTBs with chroma SAO)
        let sao_exempt = self.ctb_chroma_sao && matches!(self.c.chroma, 1 | 2);
        self.cu_bypass = self.c.bypass && !sao_exempt && self.rng.chance(30);
        if self.c.bypass {
            e.decision(CU_TRANSQUANT_BYPASS, self.cu_bypass as u32);
        }
        self.usage.bypass += self.cu_bypass as usize;
        if !self.intra_slice {
            let mut inc = 0;
            if self.avail(x0 as i32 - 1, y0 as i32) && self.skip[self.blk(x0 - 1, y0)] {
                inc += 1;
            }
            if self.avail(x0 as i32, y0 as i32 - 1) && self.skip[self.blk(x0, y0 - 1)] {
                inc += 1;
            }
            let skip = self.rng.chance(15);
            e.decision(CU_SKIP + inc, skip as u32);
            if skip {
                self.fill(x0, y0, n, |s, i| s.skip[i] = true);
                self.merge_idx(e);
                return;
            }
        }
        self.cu_intra = self.intra_slice || self.rng.chance(35);
        if !self.intra_slice {
            e.decision(PRED_MODE, self.cu_intra as u32);
        }
        self.intra_split = false;
        if self.cu_intra && log2 == LOG2_MIN_CB {
            self.intra_split = self.rng.chance(50);
            e.decision(PART_MODE, !self.intra_split as u32);
        } else if !self.cu_intra {
            e.decision(PART_MODE, 1); // 2Nx2N
        }
        if self.cu_intra {
            self.intra_modes(e, x0, y0, n);
            self.max_trafo_depth = MAX_TH_DEPTH + self.intra_split as u32;
            self.transform_tree(e, x0, y0, x0, y0, log2, 0, 0, [[true; 2]; 2]);
        } else {
            let merge = self.rng.chance(50);
            e.decision(MERGE_FLAG, merge as u32);
            if merge {
                self.merge_idx(e);
            } else {
                let mvd = [self.rng.range(-20, 20), self.rng.range(-20, 20)];
                self.mvd(e, mvd);
                e.decision(MVP_FLAG, self.rng.below(2));
            }
            let root_cbf = merge || self.rng.chance(80);
            if !merge {
                e.decision(RQT_ROOT_CBF, root_cbf as u32);
            }
            if root_cbf {
                self.max_trafo_depth = MAX_TH_DEPTH;
                self.transform_tree(e, x0, y0, x0, y0, log2, 0, 0, [[true; 2]; 2]);
            }
        }
    }

    fn merge_idx(&mut self, e: &mut Enc) {
        // MaxNumMergeCand 5: TR cMax 4, the first bin context coded
        let v = self.rng.below(5);
        e.decision(MERGE_IDX, (v > 0) as u32);
        for k in 1..4 {
            if v < k {
                break;
            }
            e.bypass((v > k) as u32);
        }
    }

    fn mvd(&mut self, e: &mut Enc, mvd: [i32; 2]) {
        let a = mvd.map(|v| v.unsigned_abs());
        for &v in &a {
            e.decision(ABS_MVD_GT0, (v > 0) as u32);
        }
        for &v in &a {
            if v > 0 {
                e.decision(ABS_MVD_GT1, (v > 1) as u32);
            }
        }
        for k in 0..2 {
            if a[k] == 0 {
                continue;
            }
            if a[k] > 1 {
                exp_golomb(e, a[k] - 2, 1);
            }
            e.bypass((mvd[k] < 0) as u32);
        }
    }

    /// The MPM list of the prediction block at (x, y) (8.4.2).
    fn mpm(&self, x: usize, y: usize) -> [u32; 3] {
        let cand = |s: &Self, xn: i32, yn: i32, same_ctb_row: bool| -> u32 {
            if !s.avail(xn, yn) || !same_ctb_row {
                return 1;
            }
            s.ipm[s.blk(xn as usize, yn as usize)] as u32
        };
        let a = cand(self, x as i32 - 1, y as i32, true);
        let b = cand(self, x as i32, y as i32 - 1, !y.is_multiple_of(CTB));
        if a == b {
            if a < 2 { [0, 1, 26] } else { [a, 2 + ((a + 29) % 32), 2 + ((a - 2 + 1) % 32)] }
        } else {
            let c = if a != 0 && b != 0 {
                0
            } else if a != 1 && b != 1 {
                1
            } else {
                26
            };
            [a, b, c]
        }
    }

    fn intra_modes(&mut self, e: &mut Enc, x0: usize, y0: usize, n: usize) {
        let nb = if self.intra_split { 2 } else { 1 };
        let pb = n / nb;
        // decide each block's mode in order (the MPMs of later blocks depend on earlier ones),
        // then write all prev_intra_luma_pred_flags, then the indices / remainders
        let mut coded = Vec::new();
        for j in 0..nb {
            for i in 0..nb {
                let (x, y) = (x0 + i * pb, y0 + j * pb);
                let mode = match self.rng.below(10) {
                    0..=3 => [10, 26][self.rng.below(2) as usize],
                    4 | 5 => self.rng.below(2),
                    _ => self.rng.below(35),
                };
                let list = self.mpm(x, y);
                coded.push(match list.iter().position(|&m| m == mode) {
                    Some(k) => (true, k as u32),
                    None => {
                        let mut sorted = list;
                        sorted.sort_unstable();
                        let rem = mode - sorted.iter().filter(|&&m| m < mode).count() as u32;
                        (false, rem)
                    }
                });
                self.fill(x, y, pb, |s, i| s.ipm[i] = mode as u8);
            }
        }
        for &(prev, _) in &coded {
            e.decision(PREV_INTRA_LUMA, prev as u32);
        }
        for &(prev, v) in &coded {
            if prev {
                e.bypass((v > 0) as u32);
                if v > 0 {
                    e.bypass((v > 1) as u32);
                }
            } else {
                e.bypass_bits(v, 5);
            }
        }
        if !self.c.has_chroma() {
            return;
        }
        let per_pu = if self.c.chroma == 3 { nb } else { 1 };
        for j in 0..per_pu {
            for i in 0..per_pu {
                let icpm = self.rng.below(5);
                if icpm == 4 {
                    e.decision(INTRA_CHROMA, 0);
                } else {
                    e.decision(INTRA_CHROMA, 1);
                    e.bypass_bits(icpm, 2);
                }
                let luma = self.ipm[self.blk(x0 + i * pb, y0 + j * pb)] as u32;
                let mode = match icpm {
                    4 => luma,
                    _ => {
                        let m = [0, 26, 10, 1][icpm as usize];
                        if m == luma { 34 } else { m }
                    }
                };
                let k = j * per_pu + i;
                self.chroma_modes[k] = if self.c.chroma == 2 { INTRA_CHROMA_MODE_422[mode as usize] as u32 } else { mode };
                self.chroma_dm[k] = icpm == 4;
            }
        }
    }

    /// The chroma prediction block of chroma (= luma in 4:4:4) position (x, y).
    fn chroma_pu(&self, x: usize, y: usize) -> usize {
        if self.intra_split && self.c.chroma == 3 {
            let h = self.cu_log2 - 1;
            (((y >> h) & 1) << 1) | ((x >> h) & 1)
        } else {
            0
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_tree(&mut self, e: &mut Enc, x0: usize, y0: usize, xb: usize, yb: usize, log2: u32, depth: u32, blk_idx: u32, parent: [[bool; 2]; 2]) {
        let split = if log2 <= LOG2_MAX_TB && log2 > 2 && depth < self.max_trafo_depth && !(self.intra_split && depth == 0) {
            let s = self.rng.chance(40);
            e.decision(SPLIT_TRANSFORM + (5 - log2 as usize), s as u32);
            s
        } else {
            log2 > LOG2_MAX_TB || (self.intra_split && depth == 0)
        };
        let mut cbf_c = [[false; 2]; 2];
        let chroma = self.c.chroma;
        if self.c.has_chroma() && (log2 > 2 || chroma == 3) {
            let second = chroma == 2 && (!split || log2 == 3);
            for c in 0..2 {
                if depth == 0 || parent[c][0] {
                    for t in 0..1 + second as usize {
                        cbf_c[c][t] = self.rng.chance(60);
                        e.decision(CBF_CHROMA + depth as usize, cbf_c[c][t] as u32);
                    }
                }
            }
        } else if self.c.has_chroma() && depth > 0 {
            cbf_c = parent;
        }
        if split {
            let h = 1usize << (log2 - 1);
            for (k, (dx, dy)) in [(0, 0), (h, 0), (0, h), (h, h)].into_iter().enumerate() {
                self.transform_tree(e, x0 + dx, y0 + dy, x0, y0, log2 - 1, depth + 1, k as u32, cbf_c);
            }
            return;
        }
        let any_c = cbf_c.iter().flatten().any(|&b| b);
        let cbf_luma = if self.cu_intra || depth != 0 || any_c {
            let v = self.rng.chance(70);
            e.decision(CBF_LUMA + (depth == 0) as usize, v as u32);
            v
        } else {
            true
        };
        self.transform_unit(e, x0, y0, xb, yb, log2, blk_idx, cbf_luma, cbf_c);
    }

    #[allow(clippy::too_many_arguments)]
    fn transform_unit(&mut self, e: &mut Enc, x0: usize, y0: usize, xb: usize, yb: usize, log2: u32, blk_idx: u32, cbf_luma: bool, cbf_c: [[bool; 2]; 2]) {
        let chroma = self.c.chroma;
        let chroma_here = self.c.has_chroma() && (log2 > 2 || blk_idx == 3 || chroma == 3);
        let cbf_chroma = cbf_c.iter().flatten().any(|&b| b);
        if cbf_chroma && !self.cu_bypass && self.cqo_enabled && !self.cqo_coded {
            let list_len = self.c.chroma_qp_offsets.as_ref().map_or(1, |(_, l)| l.len()) as u32;
            let on = self.rng.chance(60);
            e.decision(CU_CHROMA_QP_OFFSET_FLAG, on as u32);
            if on && list_len > 1 {
                let idx = self.rng.below(list_len);
                for k in 0..list_len - 1 {
                    e.decision(CU_CHROMA_QP_OFFSET_IDX, (idx > k) as u32);
                    if idx == k {
                        break;
                    }
                }
            }
            self.usage.chroma_qp_offset += on as usize;
            self.cu_qp_offset = match &self.c.chroma_qp_offsets {
                Some((_, l)) if on => l.first().map_or([0, 0], |&(cb, cr)| [cb, cr]),
                _ => [0, 0],
            };
            self.cqo_coded = true;
        }
        let ccp = self.c.ccp && cbf_luma && (!self.cu_intra || self.chroma_dm[self.chroma_pu(x0, y0)]);
        if cbf_luma {
            self.residual(e, x0, y0, log2, 0);
        }
        if !chroma_here {
            return;
        }
        let (sx, sy) = self.c.shifts();
        let (xc, yc, log2c) = if log2 > 2 || chroma == 3 { (x0 >> sx, y0 >> sy, log2 - sx) } else { (xb >> sx, yb >> sy, 2) };
        let blocks = if chroma == 2 { 2 } else { 1 };
        for c in 1..3 {
            if ccp {
                let v = self.rng.below(5);
                for k in 0..4 {
                    e.decision(LOG2_RES_SCALE_ABS + 4 * (c - 1) + k as usize, (v > k) as u32);
                    if v == k {
                        break;
                    }
                }
                if v > 0 {
                    e.decision(RES_SCALE_SIGN + c - 1, self.rng.below(2));
                    self.usage.ccp += 1;
                    self.usage.ccp_without_chroma_residual += !cbf_c[c - 1][0] as usize;
                }
            }
            for t in 0..blocks {
                if cbf_c[c - 1][t] {
                    self.usage.intra_4x4_chroma += (self.cu_intra && log2c == 2) as usize;
                    self.residual(e, xc, yc + (t << log2c), log2c, c);
                }
            }
        }
    }

    /// Halve the levels of `coef` (an n x n block of component `c`) until its residual stays
    /// within +-12000 (computed as the decoder does: scaling, transform or transform skip,
    /// residual DPCM).
    fn bound_residual(&self, coef: &mut [i32], log2: u32, c: usize, ts: bool, bypass: bool, rdpcm: Option<bool>) {
        let n = 1usize << log2;
        let bd = if c == 0 { self.c.bd_y } else { self.c.bd_c };
        let qp_bd = |bd: u32| 6 * (bd as i32 - 8);
        let qp = if c == 0 {
            self.slice_qp + qp_bd(bd)
        } else {
            let pps_off = if c == 1 { self.c.cb_cr_qp_offset.0 } else { self.c.cb_cr_qp_offset.1 };
            let qpi = (self.slice_qp + pps_off + self.cu_qp_offset[c - 1]).clamp(-qp_bd(bd), 57);
            qpc(qpi, self.c.chroma == 1) + qp_bd(bd)
        };
        let peak = |coef: &[i32]| -> i32 {
            let mut r: Vec<i32> = coef.to_vec();
            if !bypass {
                let shift = bd + log2 - 5;
                let scale = (LEVEL_SCALE[(qp % 6) as usize] as i64) << (qp / 6);
                for v in r.iter_mut() {
                    *v = ((*v as i64 * 16 * scale + (1 << (shift - 1))) >> shift).clamp(-32768, 32767) as i32;
                }
                if ts {
                    transform::transform_skip(&mut r, n, 20 - bd);
                } else {
                    transform::inverse_transform(&mut r, n, self.cu_intra && c == 0 && n == 4, n - 1, n - 1, 20 - bd);
                }
            }
            if let Some(vertical) = rdpcm {
                for i in 0..n * n {
                    let prev = if vertical {
                        i.checked_sub(n)
                    } else if i % n > 0 {
                        Some(i - 1)
                    } else {
                        None
                    };
                    if let Some(p) = prev {
                        r[i] += r[p];
                    }
                }
            }
            r.iter().map(|v| v.abs()).max().unwrap_or(0)
        };
        for _ in 0..64 {
            if peak(coef) <= 12000 {
                return;
            }
            let big = coef.iter().any(|v| v.abs() > 1);
            let mut keep = true;
            for v in coef.iter_mut().filter(|v| **v != 0) {
                if big {
                    if v.abs() > 1 {
                        *v /= 2;
                    }
                } else {
                    // every level is 1 already: drop every other coefficient (keeping one)
                    if !keep {
                        *v = 0;
                    }
                    keep = !keep;
                }
            }
        }
    }

    /// predModeIntra of component `c` at component position (x, y) of the current intra CU.
    fn intra_mode(&self, x: usize, y: usize, c: usize) -> u32 {
        if c == 0 { self.ipm[self.blk(x, y)] as u32 } else { self.chroma_modes[self.chroma_pu(x, y)] }
    }

    /// residual_coding() of random coefficients (7.3.8.11).
    fn residual(&mut self, e: &mut Enc, x0: usize, y0: usize, log2: u32, c: usize) {
        let n = 1usize << log2;
        let bypass = self.cu_bypass;
        let mut ts = false;
        if self.c.transform_skip && !bypass && log2 <= self.c.log2_max_ts {
            ts = self.rng.chance(if log2 == 2 { 60 } else { 35 });
            e.decision(TRANSFORM_SKIP + (c > 0) as usize, ts as u32);
        }
        self.usage.transform_skip += ts as usize;
        self.usage.large_ts += (ts && log2 > 2) as usize;
        // residual DPCM: Some(vertical)
        let mut rdpcm_dir = None;
        if !self.cu_intra && self.c.explicit_rdpcm && (ts || bypass) {
            let on = self.rng.chance(60);
            e.decision(EXPLICIT_RDPCM + (c > 0) as usize, on as u32);
            if on {
                let vertical = self.rng.chance(50);
                e.decision(EXPLICIT_RDPCM_DIR + (c > 0) as usize, vertical as u32);
                rdpcm_dir = Some(vertical);
                self.usage.explicit_rdpcm += 1;
            }
        }
        let mode = if self.cu_intra { self.intra_mode(x0, y0, c) } else { 1 };
        if self.cu_intra && self.c.implicit_rdpcm && (ts || bypass) && (mode == 10 || mode == 26) {
            rdpcm_dir = Some(mode == 26);
            self.usage.implicit_rdpcm += 1;
        }
        let rdpcm = rdpcm_dir.is_some();
        self.usage.rotated += ((ts || bypass) && n == 4 && self.cu_intra && self.c.rotation) as usize;
        // random coefficients: few to many, mostly small levels (smaller in lossless and
        // transform-skipped blocks, whose residual DPCM accumulates them)
        let mut coef = vec![0i32; n * n];
        let count = match self.rng.below(4) {
            0 => 1,
            1 => 1 + self.rng.below(4),
            2 => 1 + self.rng.below((n * n / 4) as u32),
            _ => 1 + self.rng.below((n * n) as u32),
        };
        let small = bypass || ts;
        // a quarter of the transformed blocks have large levels throughout (the Rice parameter
        // and StatCoeff grow past 4 / 16)
        let big_block = self.c.big_levels && !small && self.rng.chance(25);
        for _ in 0..count {
            let (x, y) = if self.rng.chance(70) {
                (self.rng.below(n.min(8) as u32), self.rng.below(n.min(8) as u32))
            } else {
                (self.rng.below(n as u32), self.rng.below(n as u32))
            };
            let mag = match self.rng.below(100) {
                _ if big_block => self.rng.range(40, 3000),
                0..=59 => 1,
                60..=79 => 2,
                80..=89 => self.rng.range(3, 5),
                90..=96 => self.rng.range(6, 50),
                _ if self.c.big_levels && !small => self.rng.range(50, 3000),
                _ => self.rng.range(20, 120),
            };
            coef[y as usize * n + x as usize] = if self.rng.chance(50) { -mag } else { mag };
        }
        // keep the residual within 16 bits, as a real encoder's is (ffmpeg holds residuals in
        // 16 bits; 12-bit transform skip and residual DPCM would otherwise overflow them)
        self.bound_residual(&mut coef, log2, c, ts, bypass, rdpcm_dir);
        // scanIdx (7.4.9.11)
        let scan_idx = if self.cu_intra && (log2 == 2 || (log2 == 3 && (c == 0 || self.c.chroma == 3))) {
            if (6..=14).contains(&mode) {
                2
            } else if (22..=30).contains(&mode) {
                1
            } else {
                0
            }
        } else {
            0
        };
        let log2sb = log2 - 2;
        let sb_scan = scan_order(log2sb, scan_idx);
        let pos_scan = &SCAN_4[scan_idx];
        let at = |coef: &[i32], i: usize, p: usize| -> i32 {
            let (xs, ys) = (sb_scan[i].0 as usize, sb_scan[i].1 as usize);
            let (xp, yp) = (pos_scan[p].0 as usize, pos_scan[p].1 as usize);
            coef[(ys * 4 + yp) * n + xs * 4 + xp]
        };
        // the last significant coefficient in scan order
        let nsb = 1usize << (2 * log2sb);
        let mut last = (0, 0);
        for i in 0..nsb {
            for p in 0..16 {
                if at(&coef, i, p) != 0 {
                    last = (i, p);
                }
            }
        }
        let (last_sb, last_pos) = last;
        let (xs, ys) = (sb_scan[last_sb].0 as u32, sb_scan[last_sb].1 as u32);
        let (lx, ly) = (xs * 4 + pos_scan[last_pos].0 as u32, ys * 4 + pos_scan[last_pos].1 as u32);
        if at(&coef, last_sb, last_pos) == 0 {
            // (no coefficient at all: make the DC one)
            coef[0] = 1;
        }
        let (cx, cy) = if scan_idx == 2 { (ly, lx) } else { (lx, ly) };
        let (off, shift) = if c == 0 { (3 * (log2 - 2) + ((log2 - 1) >> 2), (log2 + 1) >> 2) } else { (15, log2 - 2) };
        let cmax = (log2 << 1) - 1;
        let split = |v: u32| -> (u32, u32, u32) {
            if v < 4 {
                return (v, 0, 0);
            }
            let k = 31 - v.leading_zeros();
            let b = (v >> (k - 1)) & 1;
            (2 * k + b, v - ((2 + b) << (k - 1)), k - 1)
        };
        let (px, sx_v, sx_n) = split(cx);
        let (py, sy_v, sy_n) = split(cy);
        for (prefix, ctx) in [(px, LAST_X_PREFIX), (py, LAST_Y_PREFIX)] {
            for k in 0..cmax {
                e.decision(ctx + (off + (k >> shift)) as usize, (k < prefix) as u32);
                if k == prefix {
                    break;
                }
            }
        }
        if px > 3 {
            e.bypass_bits(sx_v, sx_n);
        }
        if py > 3 {
            e.bypass_bits(sy_v, sy_n);
        }
        // the actual last position of the (possibly changed) coefficients
        let (last_sb, last_pos) = {
            let mut l = (0, 0);
            for i in 0..nsb {
                for p in 0..16 {
                    if at(&coef, i, p) != 0 {
                        l = (i, p);
                    }
                }
            }
            l
        };
        let sdh = self.c.sign_hiding && !bypass && !rdpcm;
        let ts_ctx = self.c.ts_context && (ts || bypass);
        let sb_type = STAT_COEFF + if c == 0 { 2 } else { 0 } + (ts || bypass) as usize;
        let rice_max = if self.c.persistent_rice { u32::MAX } else { 4 };
        let sb_w = 1usize << log2sb;
        let mut csbf = [0u8; 64];
        let mut prev_g1: Option<u32> = None;
        let c_off_sig = if c == 0 { 0 } else { 27 };
        for i in (0..=last_sb).rev() {
            let (xs, ys) = (sb_scan[i].0 as usize, sb_scan[i].1 as usize);
            let right = if xs + 1 < sb_w { csbf[ys * 8 + xs + 1] } else { 0 };
            let below = if ys + 1 < sb_w { csbf[(ys + 1) * 8 + xs] } else { 0 };
            let any = (0..16).any(|p| at(&coef, i, p) != 0);
            let mut infer_dc = false;
            if i < last_sb && i > 0 {
                let inc = (right | below).min(1) as usize + if c > 0 { 2 } else { 0 };
                e.decision(CODED_SUB_BLOCK + inc, any as u32);
                csbf[ys * 8 + xs] = any as u8;
                infer_dc = true;
            } else {
                csbf[ys * 8 + xs] = 1;
            }
            // significant positions, descending
            let mut sig: Vec<usize> = Vec::new();
            let start = if i == last_sb {
                sig.push(last_pos);
                last_pos as i32 - 1
            } else {
                15
            };
            if csbf[ys * 8 + xs] != 0 {
                let prev_csbf = right + (below << 1);
                let tab = if ts_ctx { [if c == 0 { 42 } else { 16 }; 16] } else { sig_ctx_table(log2, c, scan_idx, prev_csbf, xs + ys == 0) };
                let mut p = start;
                while p >= 0 {
                    let nz = at(&coef, i, p as usize) != 0;
                    if p > 0 || !infer_dc {
                        e.decision(SIG_COEFF + c_off_sig + tab[p as usize] as usize, nz as u32);
                        if nz {
                            sig.push(p as usize);
                            infer_dc = false;
                        }
                    } else {
                        // inferred significant DC
                        sig.push(0);
                    }
                    p -= 1;
                }
            }
            if sig.is_empty() {
                continue;
            }
            let nsig = sig.len();
            // sign data hiding: the first (lowest) position's sign follows the level parity
            let sign_hidden = sdh && sig[0] as i32 - sig[nsig - 1] as i32 > 3;
            if sign_hidden {
                let sum: i32 = sig.iter().map(|&p| at(&coef, i, p).abs()).sum();
                let (xp, yp) = (pos_scan[sig[nsig - 1]].0 as usize, pos_scan[sig[nsig - 1]].1 as usize);
                let k = (ys * 4 + yp) * n + xs * 4 + xp;
                coef[k] = if sum & 1 == 1 { -coef[k].abs() } else { coef[k].abs() };
            }
            let levels: Vec<i32> = sig.iter().map(|&p| at(&coef, i, p)).collect();
            // greater1 / greater2 flags
            let mut ctx_set = if i == 0 || c > 0 { 0 } else { 2 };
            if prev_g1 == Some(0) {
                ctx_set += 1;
            }
            let mut g1ctx = 1u32;
            let mut first_g1 = None;
            let c_off_g1 = if c > 0 { 16 } else { 0 };
            for k in 0..nsig.min(8) {
                let f = levels[k].abs() > 1;
                e.decision(GT1 + ctx_set * 4 + g1ctx.min(3) as usize + c_off_g1, f as u32);
                if f {
                    g1ctx = 0;
                    if first_g1.is_none() {
                        first_g1 = Some(k);
                    }
                } else if g1ctx > 0 {
                    g1ctx += 1;
                }
            }
            prev_g1 = Some(g1ctx);
            if let Some(k) = first_g1 {
                e.decision(GT2 + ctx_set + if c > 0 { 4 } else { 0 }, (levels[k].abs() > 2) as u32);
            }
            for (k, &l) in levels.iter().enumerate() {
                if !(sign_hidden && k == nsig - 1) {
                    e.bypass((l < 0) as u32);
                }
            }
            // remaining levels
            let mut rice = if self.c.persistent_rice { e.ctx[sb_type] as u32 / 4 } else { 0 };
            let mut update_stat = self.c.persistent_rice;
            for (k, &l) in levels.iter().enumerate() {
                let a = l.abs();
                let mut base = if k < 8 { 1 + (a > 1) as i32 } else { 1 };
                if Some(k) == first_g1 {
                    base += (a > 2) as i32;
                }
                let thresh = if k < 8 { if Some(k) == first_g1 { 3 } else { 2 } } else { 1 };
                if base == thresh {
                    let rem = (a - base) as u32;
                    self.usage.rice_above_4 += (rice > 4) as usize;
                    level_remaining(e, rem, rice);
                    if update_stat {
                        update_stat = false;
                        let stat = e.ctx[sb_type];
                        let s = (stat / 4) as u32;
                        if rem as u64 >= 3u64 << s {
                            e.ctx[sb_type] = stat + 1;
                        } else if 2 * (rem as u64) < 1u64 << s && stat > 0 {
                            e.ctx[sb_type] = stat - 1;
                        }
                    }
                    if a as i64 > 3i64 << rice {
                        rice = (rice + 1).min(rice_max);
                    }
                }
            }
        }
    }
}

/// k-th order Exp-Golomb bypass bins of `v` (9.3.3.3).
fn exp_golomb(e: &mut Enc, mut v: u32, mut k: u32) {
    while v >= 1 << k {
        e.bypass(1);
        v -= 1 << k;
        k += 1;
    }
    e.bypass(0);
    e.bypass_bits(v, k);
}

/// coeff_abs_level_remaining (9.3.3.11): a TR prefix of up to four bins with cMax 4 << rice, then
/// an EG(rice + 1) suffix.
fn level_remaining(e: &mut Enc, v: u32, rice: u32) {
    let prefix = v >> rice;
    if prefix < 4 {
        for _ in 0..prefix {
            e.bypass(1);
        }
        e.bypass(0);
        e.bypass_bits(v & ((1 << rice) - 1), rice);
    } else {
        for _ in 0..4 {
            e.bypass(1);
        }
        exp_golomb(e, v - (4 << rice), rice + 1);
    }
}

/// The slice segments of one picture: NAL units, and what the encoder used.
fn encode_picture(c: &Cfg, poc: u32, rng: &mut Rng, usage: &mut Usage) -> Vec<u8> {
    let (idr, intra) = (poc == 0, poc == 0 || c.intra_only);
    let (wctb, hctb) = (c.wctb(), c.hctb());
    let slice_rows = if c.slice_rows == 0 { hctb } else { c.slice_rows };
    let slices: Vec<(usize, usize)> = (0..hctb).step_by(slice_rows).map(|r| (r, (r + slice_rows).min(hctb))).collect();
    let mut ctb_slice = vec![usize::MAX; wctb * hctb];
    let w4 = c.width.div_ceil(4);
    let mut st = PicEnc {
        c,
        rng: Rng(rng.next() | 1),
        usage: Usage::default(),
        intra_slice: intra,
        depth: vec![0; w4 * c.height.div_ceil(4)],
        ipm: vec![1; w4 * c.height.div_ceil(4)],
        skip: vec![false; w4 * c.height.div_ceil(4)],
        w4,
        ctb_slice: Vec::new(),
        slice: 0,
        cu_bypass: false,
        cu_intra: true,
        cu_log2: 3,
        intra_split: false,
        chroma_modes: [0; 4],
        chroma_dm: [false; 4],
        max_trafo_depth: 0,
        cqo_coded: false,
        log2_min_cqo: LOG2_CTB - c.chroma_qp_offsets.as_ref().map_or(0, |(d, _)| *d),
        cqo_enabled: false,
        sao_chroma_on: vec![false; wctb * hctb],
        ctb_chroma_sao: false,
        slice_qp: c.init_qp,
        cu_qp_offset: [0, 0],
    };
    let mut out = Vec::new();
    for (si, &(r0, r1)) in slices.iter().enumerate() {
        for v in &mut ctb_slice[r0 * wctb..r1 * wctb] {
            *v = si;
        }
        st.ctb_slice = ctb_slice.clone();
        st.slice = si;
        let qp = c.init_qp + st.rng.range(-3, 3);
        st.slice_qp = qp;
        let init_type = if intra { 0 } else { 1 };
        let sao = c.sao.then(|| (st.rng.chance(80), c.has_chroma() && st.rng.chance(80)));
        st.cqo_enabled = c.chroma_qp_offsets.is_some() && st.rng.chance(85);
        let mut e = Enc::new(BitWriter::new(), qp, init_type);
        let mut starts = vec![0usize];
        let mut wpp_saved = None;
        for ry in r0..r1 {
            for rx in 0..wctb {
                if c.wpp && rx == 0 && ry > r0 {
                    // WPP: contexts (and StatCoeff) of the CTB above-right, stored after it
                    match wpp_saved {
                        Some(saved) => e.ctx = saved,
                        None => init_contexts(&mut e.ctx, qp, init_type),
                    }
                }
                if let Some(flags) = sao
                    && (flags.0 || flags.1)
                {
                    st.sao(&mut e, rx, ry, flags);
                }
                st.ctb_chroma_sao = st.sao_chroma_on[ry * wctb + rx];
                st.quadtree(&mut e, rx * CTB, ry * CTB, LOG2_CTB, 0);
                if c.wpp && rx == 1 {
                    wpp_saved = Some(e.ctx);
                }
                let last = ry + 1 == r1 && rx + 1 == wctb;
                e.terminate(last as u32);
                if last {
                    e.w.align_zero();
                } else if c.wpp && rx + 1 == wctb {
                    e.terminate(1); // end_of_subset_one_bit
                    e.w.align_zero();
                    starts.push(e.w.bit_len() / 8);
                    e.reset_engine();
                }
            }
        }
        let data = e.w.finish();
        let nal_type = if idr { 19 } else { 1 };
        let mut offsets: Vec<u32> = starts.windows(2).map(|s| (s[1] - s[0]) as u32).collect();
        loop {
            let mut w = BitWriter::new();
            w.write_bit(si == 0); // first_slice_segment_in_pic_flag
            if idr {
                w.write_bit(false); // no_output_of_prior_pics
            }
            w.write_ue(0); // pps id
            if si > 0 {
                let bits = 32 - ((wctb * hctb) as u32 - 1).leading_zeros();
                w.write_bits((r0 * wctb) as u32, bits);
            }
            w.write_ue(if intra { 2 } else { 1 });
            if !idr {
                w.write_bits(poc & 255, 8);
                w.write_bit(false); // short_term_ref_pic_set_sps_flag
                // the previous picture (kept, and used by P pictures)
                w.write_ue(1); // one negative picture
                w.write_ue(0);
                w.write_ue(0); // delta_poc_s0_minus1
                w.write_bit(!intra); // used_by_curr_pic_s0_flag
            }
            if let Some((luma, chroma)) = sao {
                w.write_bit(luma);
                if c.has_chroma() {
                    w.write_bit(chroma);
                }
            }
            if !intra {
                w.write_bit(false); // num_ref_idx_active_override_flag
                if c.weighted {
                    pred_weight_table(&mut w, c, &mut st.rng);
                    st.usage.weighted += 1;
                }
                w.write_ue(0); // five_minus_max_num_merge_cand
            }
            w.write_se(qp - c.init_qp);
            if c.chroma_qp_offsets.is_some() {
                w.write_bit(st.cqo_enabled);
            }
            // slice_loop_filter_across_slices_enabled_flag (deblocking is on)
            w.write_bit(true);
            if c.wpp {
                w.write_ue(offsets.len() as u32);
                if !offsets.is_empty() {
                    w.write_ue(31);
                    for &o in &offsets {
                        w.write_bits(o - 1, 32);
                    }
                }
            }
            w.write_bit(true);
            w.align_zero();
            let mut rbsp = w.finish();
            let hdr_len = rbsp.len();
            rbsp.extend_from_slice(&data);
            let esc = |idx: usize| escape_rbsp(&rbsp[..idx]).len();
            let s: Vec<usize> = starts.iter().map(|&s| esc(hdr_len + s)).collect();
            let new: Vec<u32> = s.windows(2).map(|p| (p[1] - p[0]) as u32).collect();
            if new == offsets {
                out.extend(nal(nal_type, &rbsp));
                break;
            }
            offsets = new;
        }
    }
    let u = &st.usage;
    for (total, add) in [
        (&mut usage.transform_skip, u.transform_skip),
        (&mut usage.bypass, u.bypass),
        (&mut usage.rotated, u.rotated),
        (&mut usage.implicit_rdpcm, u.implicit_rdpcm),
        (&mut usage.explicit_rdpcm, u.explicit_rdpcm),
        (&mut usage.ccp, u.ccp),
        (&mut usage.ccp_without_chroma_residual, u.ccp_without_chroma_residual),
        (&mut usage.chroma_qp_offset, u.chroma_qp_offset),
        (&mut usage.large_ts, u.large_ts),
        (&mut usage.rice_above_4, u.rice_above_4),
        (&mut usage.intra_4x4_chroma, u.intra_4x4_chroma),
        (&mut usage.weighted, u.weighted),
    ] {
        *total += add;
    }
    out
}

/// pred_weight_table() for one reference (7.3.6.3), random weights and offsets in their ranges
/// (wider offsets with high_precision_offsets_enabled_flag).
fn pred_weight_table(w: &mut BitWriter, c: &Cfg, rng: &mut Rng) {
    let luma_denom = rng.below(8);
    w.write_ue(luma_denom);
    if c.has_chroma() {
        let chroma_denom = rng.below(8);
        w.write_se(chroma_denom as i32 - luma_denom as i32);
    }
    let half = |bd: u32| if c.high_precision { 1i32 << (bd - 1) } else { 128 };
    let (luma_flag, chroma_flag) = (rng.chance(80), c.has_chroma() && rng.chance(80));
    w.write_bit(luma_flag);
    if c.has_chroma() {
        w.write_bit(chroma_flag);
    }
    if luma_flag {
        w.write_se(rng.range(-30, 30));
        let h = half(c.bd_y);
        w.write_se(rng.range(-h, h - 1));
    }
    if chroma_flag {
        let h = half(c.bd_c);
        for _ in 0..2 {
            w.write_se(rng.range(-30, 30));
            w.write_se(rng.range(-4 * h, 4 * h - 1));
        }
    }
}

fn stream(c: &Cfg) -> (Vec<u8>, Usage) {
    let mut s = Vec::new();
    s.extend(vps());
    s.extend(sps(c));
    s.extend(pps(c));
    let mut rng = Rng(c.seed);
    let mut usage = Usage::default();
    for poc in 0..c.pictures as u32 {
        s.extend(encode_picture(c, poc, &mut rng, &mut usage));
    }
    (s, usage)
}

fn decode_all(stream: &[u8], threads: usize) -> Vec<Picture> {
    let mut dec = Decoder::with_threads(threads);
    let mut pics = dec.decode(stream, 0).unwrap_or_else(|e| panic!("decode: {e}"));
    pics.extend(dec.flush());
    if let Some(e) = dec.take_error() {
        panic!("decode: {e}");
    }
    pics
}

/// Decode `c`'s stream with this decoder (1 and 4 threads) and ffmpeg and compare every plane;
/// returns the encoder's tool usage (None when ffmpeg is absent).
fn check(c: &Cfg) -> Option<Usage> {
    let (s, usage) = stream(c);
    if let Ok(d) = std::env::var("HEVC_SYNTH_DUMP") {
        std::fs::write(format!("{d}/{}.hevc", c.name), &s).unwrap();
    }
    let ours = decode_all(&s, 1);
    assert_eq!(ours.len(), c.pictures, "{}: pictures", c.name);
    let threaded = decode_all(&s, 4);
    for (a, b) in ours.iter().zip(&threaded) {
        assert!(a.y == b.y && a.u == b.u && a.v == b.v, "{}: threaded decode differs (poc {})", c.name, a.poc);
    }
    let ff = filmcraft_testkit::ffmpeg_or_skip(c.name)?;
    let dir = filmcraft_testkit::fixtures_dir("hevc");
    let src = filmcraft_testkit::temp_path(&dir.join(format!("rext_{}.hevc", c.name)));
    let yuv = filmcraft_testkit::temp_path(&dir.join(format!("rext_{}.yuv", c.name)));
    std::fs::write(&src, &s).unwrap();
    let o = std::process::Command::new(ff)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&src)
        .args(["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", &c.pix_fmt()])
        .arg(&yuv)
        .output()
        .unwrap();
    // (ffmpeg notes that no conformance stream exercised its cu_chroma_qp_offset_idx parsing;
    // anything else it reports is an error in the stream)
    let log = String::from_utf8_lossy(&o.stderr);
    let errors: Vec<&str> = log.lines().filter(|l| !l.contains("cu_chroma_qp_offset_idx not yet tested") && !l.contains("Last message repeated")).collect();
    assert!(o.status.success() && errors.is_empty(), "{}: ffmpeg: {}", c.name, errors.join("\n"));
    let r = std::fs::read(&yuv).unwrap();
    let _ = (std::fs::remove_file(&yuv), std::fs::remove_file(&src));
    let bps = if c.bd_y > 8 { 2 } else { 1 };
    let (sx, sy) = c.shifts();
    let (cw, ch) = if c.has_chroma() { (c.width >> sx, c.height >> sy) } else { (0, 0) };
    let fs = (c.width * c.height + 2 * cw * ch) * bps;
    assert_eq!(r.len(), fs * c.pictures, "{}: ffmpeg decoded {} pictures", c.name, r.len() / fs);
    for (i, p) in ours.iter().enumerate() {
        let f = &r[i * fs..(i + 1) * fs];
        let ys = c.width * c.height * bps;
        for (name, plane, exp, pw) in [("Y", &p.y, &f[..ys], c.width), ("U", &p.u, &f[ys..ys + cw * ch * bps], cw), ("V", &p.v, &f[ys + cw * ch * bps..], cw)] {
            let got = plane.to_le_bytes();
            if got != exp {
                let k = got.iter().zip(exp).position(|(a, b)| a != b).unwrap_or(0) / bps;
                panic!("{}: picture {i} plane {name} differs from ffmpeg first at ({}, {}); usage {usage:?}", c.name, k % pw.max(1), k / pw.max(1));
            }
        }
    }
    Some(usage)
}

/// Every tool of the format and depth, intra pictures only (rotation, implicit RDPCM, intra
/// smoothing off, transform-skip contexts and 32x32 transform skip, persistent Rice adaptation,
/// cross-component prediction from intra CUs, a chroma QP offset list, 4x4 chroma blocks).
#[test]
fn rext_intra_tools_444_8bit() {
    let c = Cfg { pictures: 2, intra_only: true, smoothing_off: true, seed: 0x1234_5678, ..Cfg::all_tools("intra_444_8", 3, 8) };
    let Some(u) = check(&c) else { return };
    assert!(
        u.rotated > 0 && u.implicit_rdpcm > 0 && u.ccp > 0 && u.chroma_qp_offset > 0 && u.large_ts > 0 && u.rice_above_4 > 0 && u.intra_4x4_chroma > 0,
        "{u:?}"
    );
}

/// Lossless (transquant bypass) CUs with the tools that take them: transform-skip contexts,
/// persistent Rice adaptation (the bypass sbTypes), explicit RDPCM of inter CUs, cross-component
/// prediction, the chroma QP offset list (not coded for bypass CUs).
#[test]
fn rext_lossless_tools_444_10bit() {
    let c = Cfg { pictures: 3, seed: 0x7777_0001, ..Cfg::lossless_tools("lossless_444_10", 3, 10) };
    let Some(u) = check(&c) else { return };
    assert!(u.bypass > 0 && u.explicit_rdpcm > 0 && u.ccp > 0 && u.rice_above_4 > 0, "{u:?}");
}

/// P pictures with WPP: explicit RDPCM, cross-component prediction of inter CUs (with and
/// without chroma residuals), weighted prediction, and StatCoeff carried by the WPP context
/// synchronisation.
#[test]
fn rext_all_tools_444_10bit_wpp() {
    let c = Cfg { wpp: true, smoothing_off: true, pictures: 4, seed: 0xdead_beef, ..Cfg::all_tools("all_444_10_wpp", 3, 10) };
    let Some(u) = check(&c) else { return };
    assert!(u.explicit_rdpcm > 0 && u.ccp_without_chroma_residual > 0 && u.weighted > 0 && u.rice_above_4 > 0, "{u:?}");
}

/// 12 bits: SAO offsets scaled by 4 (log2_sao_offset_scale 2), one slice per CTB row (StatCoeff
/// and the contexts start again in each).
#[test]
fn rext_all_tools_444_12bit_sao_scale_slices() {
    let c = Cfg { slice_rows: 1, pictures: 3, seed: 0x5151_aaaa, ..Cfg::all_tools("all_444_12_slices", 3, 12) };
    assert_eq!(c.sao_scale, (2, 2));
    check(&c);
    check(&Cfg { seed: 0x5151_aaab, ..Cfg::lossless_tools("lossless_444_12_slices", 3, 12) });
}

/// 4:2:0 at 8 bits, with high-precision weighted-prediction offsets (identical to the 8-bit ones
/// at this depth, so ffmpeg agrees).
#[test]
fn rext_all_tools_420_8bit() {
    let Some(u) = check(&Cfg { pictures: 4, seed: 0x4200_0001, high_precision: true, ..Cfg::all_tools("all_420_8", 1, 8) }) else { return };
    assert!(u.implicit_rdpcm > 0 && u.explicit_rdpcm > 0 && u.weighted > 0, "{u:?}");
    check(&Cfg { pictures: 3, seed: 0x4200_0002, ..Cfg::lossless_tools("lossless_420_8", 1, 8) });
}

/// 4:2:2 (the implicit RDPCM direction follows the Table 8-3 chroma mode; two chroma blocks per
/// unit) with WPP.
#[test]
fn rext_all_tools_422_10bit() {
    check(&Cfg { pictures: 4, seed: 0x4220_0010, wpp: true, ..Cfg::all_tools("all_422_10", 2, 10) });
    check(&Cfg { pictures: 3, seed: 0x4220_0011, ..Cfg::lossless_tools("lossless_422_10", 2, 10) });
}

/// Monochrome at 12 bits (luma SAO offsets scaled by 4).
#[test]
fn rext_all_tools_400_12bit() {
    check(&Cfg { pictures: 4, seed: 0x4000_0012, ..Cfg::all_tools("all_400_12", 0, 12) });
    check(&Cfg { pictures: 3, seed: 0x4000_0013, wpp: true, ..Cfg::lossless_tools("lossless_400_12", 0, 12) });
}

/// The first message of this decoder on `c`'s stream (an error it refuses the stream with).
fn refusal(c: &Cfg) -> Option<String> {
    let (s, _) = stream(c);
    let mut dec = Decoder::with_threads(1);
    let r = dec.decode(&s, 0).err().map(|e| e.to_string());
    let _ = dec.flush();
    r.or_else(|| dec.take_error().map(|e| e.to_string()))
}

/// The range-extension uses whose decoded pictures no available decoder confirms are refused as
/// unsupported, never decoded to unverified pictures: ffmpeg, the oracle, decodes each of these
/// streams otherwise than this decoder reads the specification (lossless CUs: no residual
/// rotation and keeping the intra boundary filters; cu_chroma_qp_offset_idx, which it reports as
/// untested; weighted-prediction offsets above 8 bits scaled as 8-bit ones), and extended
/// precision and CABAC bypass alignment it does not implement. The same streams without the
/// refused combination decode (the tests above).
#[test]
fn unverifiable_range_extension_uses_are_refused() {
    let base = |n| Cfg { pictures: 3, ..Cfg::base(n) };
    let cases = [
        (Cfg { rotation: true, ..base("rotation_lossless") }, "rotation with lossless"),
        (Cfg { implicit_rdpcm: true, ..base("implicit_lossless") }, "implicit RDPCM with lossless"),
        (Cfg { chroma_qp_offsets: Some((1, vec![(-3, 4), (6, -2)])), ..base("cqo_list") }, "chroma QP offset lists of more than one entry"),
        (Cfg { weighted: true, high_precision: true, bd_y: 10, bd_c: 10, ..base("hp_offsets_10") }, "high-precision weighted prediction offsets"),
        (Cfg { extended_precision: true, ..base("extended_precision") }, "extended_precision_processing_flag"),
        (Cfg { bypass_alignment: true, ..base("bypass_alignment") }, "cabac_bypass_alignment_enabled_flag"),
    ];
    for (c, what) in cases {
        let err = refusal(&c).unwrap_or_else(|| panic!("{}: decoded", c.name));
        assert!(err.starts_with("unsupported") && err.contains(what), "{}: {err}", c.name);
    }
    // without the refused combination they decode
    for c in [
        Cfg { rotation: true, bypass: false, ..base("rotation_lossy") },
        Cfg { implicit_rdpcm: true, bypass: false, ..base("implicit_lossy") },
        Cfg { chroma_qp_offsets: Some((1, vec![(-3, 4)])), ..base("cqo_one") },
        Cfg { weighted: true, high_precision: true, ..base("hp_offsets_8") },
        Cfg { high_precision: true, bd_y: 10, bd_c: 10, ..base("hp_unweighted_10") },
    ] {
        assert_eq!(refusal(&c), None, "{}", c.name);
    }
}

/// Range-extension fields beyond their SPS limits are invalid (7.4.3.3.2): SAO offset scales
/// above Max(0, BitDepth - 10), cross-component prediction outside 4:4:4.
#[test]
fn range_extension_fields_are_checked_against_the_sps() {
    for (c, ok) in [
        (Cfg { sao_scale: (1, 0), bd_y: 10, bd_c: 10, ..Cfg::base("sao_scale_10") }, false),
        (Cfg { sao_scale: (0, 3), bd_y: 12, bd_c: 12, ..Cfg::base("sao_scale_12_3") }, false),
        (Cfg { sao_scale: (2, 1), bd_y: 12, bd_c: 11, ..Cfg::base("sao_scale_12_2") }, true),
        (Cfg { ccp: true, chroma: 1, ..Cfg::base("ccp_420") }, false),
        (Cfg { ccp: true, chroma: 2, ..Cfg::base("ccp_422") }, false),
    ] {
        let r = refusal(&Cfg { pictures: 1, ..c.clone() });
        if ok {
            assert_eq!(r, None, "{}", c.name);
        } else {
            let err = r.unwrap_or_else(|| panic!("{}: decoded", c.name));
            assert!(err.starts_with("invalid bitstream"), "{}: {err}", c.name);
        }
    }
}

/// Damaged range-extension streams (bit flips, overwritten bytes, truncation) never panic or
/// hang, single- and multi-threaded: errors are fine.
#[test]
fn damaged_range_extension_streams_never_panic() {
    let seeds = [
        Cfg { pictures: 3, wpp: true, ..Cfg::all_tools("fuzz_444_10", 3, 10) },
        Cfg { pictures: 2, ..Cfg::lossless_tools("fuzz_420_12", 1, 12) },
        Cfg { pictures: 2, ..Cfg::all_tools("fuzz_400_8", 0, 8) },
    ];
    for c in seeds {
        let (s, _) = stream(&c);
        let mut rng = Rng(0x0bad_cafe ^ s.len() as u64);
        for round in 0..150 {
            let mut d = s.clone();
            for _ in 0..1 + rng.below(4) {
                let p = 4 + rng.below(d.len() as u32 - 4) as usize;
                if rng.chance(50) {
                    d[p] ^= 1 << rng.below(8);
                } else {
                    d[p] = rng.next() as u8;
                }
            }
            if round % 10 == 9 {
                d.truncate(d.len() * 2 / 3);
            }
            for threads in [1, 3] {
                let r = std::panic::catch_unwind(|| {
                    let mut dec = Decoder::with_threads(threads);
                    let _ = dec.decode(&d, 0);
                    let _ = dec.flush();
                    dec.take_error().map(|e| e.to_string())
                });
                let Ok(err) = r else { panic!("{} round {round} threads {threads}: panicked", c.name) };
                assert!(!err.as_deref().unwrap_or("").contains("internal error"), "{} round {round}: a decoding job panicked", c.name);
            }
        }
    }
}
