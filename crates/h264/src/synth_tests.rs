//! Hand-built bitstreams exercising features libx264 never emits: I_PCM (CAVLC and CABAC), long-term
//! references, MMCO operations, frame_num gaps and reference list modification. Every picture consists
//! of I_PCM and P_Skip macroblocks only, so the expected output is known exactly (I_PCM samples are
//! lossless, P_Skip copies the selected reference with a zero motion vector, and deblocking cannot
//! modify samples at these QPs).

use crate::Decoder;
use crate::cabac::{Cabac, NEXT_STATE, RANGE_TAB_LPS};
use filmcraft_bitstream::{BitWriter, escape_rbsp};

const MB_W: usize = 4;
const MB_H: usize = 3;
const NMB: usize = MB_W * MB_H;

fn nal(ref_idc: u8, t: u8, rbsp: &[u8]) -> Vec<u8> {
    let mut v = vec![0, 0, 0, 1, (ref_idc << 5) | t];
    v.extend(escape_rbsp(rbsp));
    v
}

/// POC type used by the synthetic SPS: 0 (4-bit lsb), 1 (see below) or 2.
#[derive(Clone, Copy, PartialEq, Eq)]
enum PocType {
    T0,
    /// offset_for_non_ref_pic = -3, one reference frame per cycle with offset 4
    T1,
    T2,
}

fn sps(profile: u8, max_refs: u32, gaps: bool, poc: PocType) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(profile as u32, 8);
    w.write_bits(0, 8);
    w.write_bits(30, 8);
    w.write_ue(0); // sps id
    w.write_ue(0); // log2_max_frame_num - 4 -> 16
    match poc {
        PocType::T0 => {
            w.write_ue(0);
            w.write_ue(0); // log2_max_pic_order_cnt_lsb - 4 -> 4-bit lsb
        }
        PocType::T1 => {
            w.write_ue(1);
            w.write_bit(false); // delta_pic_order_always_zero_flag
            w.write_se(-3); // offset_for_non_ref_pic
            w.write_se(0); // offset_for_top_to_bottom_field
            w.write_ue(1); // num_ref_frames_in_pic_order_cnt_cycle
            w.write_se(4);
        }
        PocType::T2 => w.write_ue(2), // output order = decoding order
    }
    w.write_ue(max_refs);
    w.write_bit(gaps);
    w.write_ue(MB_W as u32 - 1);
    w.write_ue(MB_H as u32 - 1);
    w.write_bit(true); // frame_mbs_only
    w.write_bit(true); // direct_8x8_inference
    w.write_bit(false); // cropping
    w.write_bit(false); // vui
    w.rbsp_trailing();
    nal(3, 7, &w.finish())
}

fn pps(cabac: bool, weighted: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_ue(0);
    w.write_ue(0);
    w.write_bit(cabac);
    w.write_bit(false);
    w.write_ue(0); // slice groups
    w.write_ue(0); // num_ref_idx_l0_default - 1
    w.write_ue(0);
    w.write_bit(weighted); // weighted_pred
    w.write_bits(if weighted { 1 } else { 0 }, 2); // explicit weighted bi-prediction
    w.write_se(0); // pic_init_qp - 26
    w.write_se(0);
    w.write_se(0); // chroma_qp_index_offset
    w.write_bit(true); // deblocking_filter_control_present
    w.write_bit(false);
    w.write_bit(false);
    w.rbsp_trailing();
    nal(3, 8, &w.finish())
}

#[derive(Default, Clone)]
struct Pic {
    idr: bool,
    frame_num: u32,
    ref_idc: u8,
    long_term_ref: bool,
    /// ref_pic_list_modification entries (idc, value)
    mods: Vec<(u32, u32)>,
    /// MMCOs: (op, a, b) with a = difference_of_pic_nums_minus1 / long_term_pic_num /
    /// max_long_term_frame_idx_plus1 and b = long_term_frame_idx
    mmcos: Vec<(u32, u32, u32)>,
    /// Per MB: Some(samples) for I_PCM, None for P_Skip / B_Skip.
    mbs: Vec<Option<Vec<u8>>>,
    /// B slice (all B_Skip).
    b: bool,
    poc_lsb: Option<u32>,
    /// delta_pic_order_cnt[0] (POC type 1).
    poc_delta: Option<i32>,
    /// Explicit weights (luma, cb, cr) as (weight, offset) for refIdx 0 of list 0 and list 1;
    /// log2 denominators are 5 (luma) and 3 (chroma).
    wt: Option<[[(i32, i32); 3]; 2]>,
    /// first_mb_in_slice (CAVLC I and P slices: `mbs` are the slice's macroblocks from there).
    first_mb: u32,
}

impl Pic {
    fn is_p(&self) -> bool {
        !self.idr && !self.b && self.mbs.iter().any(|m| m.is_none())
    }
    fn inter(&self) -> bool {
        self.b || self.is_p()
    }
}

fn slice_header(w: &mut BitWriter, p: &Pic, cabac: bool) {
    w.write_ue(p.first_mb);
    w.write_ue(if p.b {
        6
    } else if p.is_p() {
        5
    } else {
        7
    });
    w.write_ue(0); // pps id
    w.write_bits(p.frame_num, 4);
    if p.idr {
        w.write_ue(0);
    }
    if let Some(lsb) = p.poc_lsb {
        w.write_bits(lsb, 4);
    }
    if let Some(d) = p.poc_delta {
        w.write_se(d);
    }
    if p.b {
        w.write_bit(true); // direct_spatial_mv_pred_flag
    }
    if p.inter() {
        w.write_bit(false); // num_ref_idx_override
        w.write_bit(!p.mods.is_empty());
        if !p.mods.is_empty() {
            for &(idc, v) in &p.mods {
                w.write_ue(idc);
                w.write_ue(v);
            }
            w.write_ue(3);
        }
        if p.b {
            w.write_bit(false); // no list 1 modification
        }
    }
    if let (Some(wt), true) = (p.wt, p.inter()) {
        w.write_ue(5);
        w.write_ue(3);
        for l in 0..if p.b { 2 } else { 1 } {
            w.write_bit(true);
            w.write_se(wt[l][0].0);
            w.write_se(wt[l][0].1);
            w.write_bit(true);
            for c in 1..3 {
                w.write_se(wt[l][c].0);
                w.write_se(wt[l][c].1);
            }
        }
    }
    if p.ref_idc != 0 {
        if p.idr {
            w.write_bit(false);
            w.write_bit(p.long_term_ref);
        } else {
            w.write_bit(!p.mmcos.is_empty());
            if !p.mmcos.is_empty() {
                for &(op, a, b) in &p.mmcos {
                    w.write_ue(op);
                    match op {
                        1 => w.write_ue(a),
                        2 => w.write_ue(a),
                        3 => {
                            w.write_ue(a);
                            w.write_ue(b);
                        }
                        4 => w.write_ue(a),
                        6 => w.write_ue(b),
                        _ => {}
                    }
                }
                w.write_ue(0);
            }
        }
    }
    if cabac && p.inter() {
        w.write_ue(0); // cabac_init_idc
    }
    w.write_se(0); // slice_qp_delta
    w.write_ue(0); // disable_deblocking_filter_idc
    w.write_se(0);
    w.write_se(0);
}

/// Minimal CABAC encoder (9.3.4, informative) writing into a BitWriter.
struct Enc {
    low: u32,
    range: u32,
    outstanding: u32,
    first: bool,
    ctx: [u8; 1024],
}

impl Enc {
    fn new(qp: i32, init: usize) -> Self {
        let c = Cabac::new(&[0, 0, 0, 0], 0, qp, init).unwrap();
        Enc { low: 0, range: 510, outstanding: 0, first: true, ctx: c.ctx }
    }
    fn init(&mut self) {
        self.low = 0;
        self.range = 510;
        self.outstanding = 0;
        self.first = true;
    }
    fn put(&mut self, w: &mut BitWriter, b: bool) {
        if self.first {
            self.first = false;
        } else {
            w.write_bit(b);
        }
        while self.outstanding > 0 {
            w.write_bit(!b);
            self.outstanding -= 1;
        }
    }
    fn renorm(&mut self, w: &mut BitWriter) {
        while self.range < 256 {
            if self.low < 256 {
                self.put(w, false);
            } else if self.low >= 512 {
                self.low -= 512;
                self.put(w, true);
            } else {
                self.low -= 256;
                self.outstanding += 1;
            }
            self.range <<= 1;
            self.low <<= 1;
        }
    }
    fn encode(&mut self, w: &mut BitWriter, ctx_idx: usize, bin: u32) {
        let s = self.ctx[ctx_idx] as usize;
        let lps = RANGE_TAB_LPS[s >> 1][((self.range >> 6) & 3) as usize] as u32;
        self.range -= lps;
        if bin != (s & 1) as u32 {
            self.low += self.range;
            self.range = lps;
            self.ctx[ctx_idx] = NEXT_STATE[s][1];
        } else {
            self.ctx[ctx_idx] = NEXT_STATE[s][0];
        }
        self.renorm(w);
    }
    fn terminate(&mut self, w: &mut BitWriter, bin: bool) {
        self.range -= 2;
        if bin {
            self.low += self.range;
            // EncodeFlush
            self.range = 2;
            self.renorm(w);
            self.put(w, (self.low >> 9) & 1 != 0);
            w.write_bits(((self.low >> 7) & 3) | 1, 2);
        } else {
            self.renorm(w);
        }
    }
}

fn write_pcm(w: &mut BitWriter, samples: &[u8]) {
    w.align_zero();
    w.write_bytes(samples);
}

fn slice_nal(p: &Pic, cabac: bool) -> Vec<u8> {
    let mut w = BitWriter::new();
    slice_header(&mut w, p, cabac);
    if !cabac && p.b {
        w.write_ue(NMB as u32);
        w.rbsp_trailing();
    } else if !cabac {
        let mut run = 0;
        for mb in &p.mbs {
            match mb {
                None => run += 1,
                Some(s) => {
                    if p.is_p() {
                        w.write_ue(run);
                        run = 0;
                        w.write_ue(30); // I_PCM in a P slice
                    } else {
                        w.write_ue(25);
                    }
                    write_pcm(&mut w, s);
                }
            }
        }
        if run > 0 {
            w.write_ue(run);
        }
        w.rbsp_trailing();
    } else {
        while !w.is_byte_aligned() {
            w.write_bit(true); // cabac_alignment_one_bit
        }
        let mut e = Enc::new(26, if p.inter() { 0 } else { 3 });
        let skip = |i: usize| p.mbs[i].is_none();
        for (i, mb) in p.mbs.iter().enumerate() {
            let (x, y) = (i % MB_W, i / MB_W);
            let a = if x > 0 { Some(i - 1) } else { None };
            let b = if y > 0 { Some(i - MB_W) } else { None };
            if p.inter() {
                let inc = a.map(|n| !skip(n) as usize).unwrap_or(0) + b.map(|n| !skip(n) as usize).unwrap_or(0);
                e.encode(&mut w, if p.b { 24 } else { 11 } + inc, skip(i) as u32);
            }
            if let Some(s) = mb {
                if p.is_p() {
                    e.encode(&mut w, 14, 1); // intra prefix
                    e.encode(&mut w, 17, 1); // not I_NxN
                } else {
                    // neighbours are I_PCM (not I_NxN) -> condTerm 1
                    let inc = a.is_some() as usize + b.is_some() as usize;
                    e.encode(&mut w, 3 + inc, 1);
                }
                e.terminate(&mut w, true); // I_PCM
                write_pcm(&mut w, s);
                e.init();
            }
            e.terminate(&mut w, i + 1 == NMB); // end_of_slice_flag
        }
        w.align_zero();
    }
    let t = if p.idr { 5 } else { 1 };
    nal(p.ref_idc, t, &w.finish())
}

fn random_mb(seed: &mut u32) -> Vec<u8> {
    (0..384)
        .map(|_| {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (*seed >> 24) as u8
        })
        .collect()
}

/// Expected cropped planes of a picture made of PCM blocks (`None` = copy from `reference`).
fn render(mbs: &[Option<Vec<u8>>], reference: Option<&(Vec<u8>, Vec<u8>, Vec<u8>)>) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h) = (MB_W * 16, MB_H * 16);
    let (mut y, mut u, mut v) = match reference {
        Some(r) => r.clone(),
        None => (vec![0; w * h], vec![0; w * h / 4], vec![0; w * h / 4]),
    };
    for (i, mb) in mbs.iter().enumerate() {
        let Some(s) = mb else { continue };
        let (x0, y0) = ((i % MB_W) * 16, (i / MB_W) * 16);
        for r in 0..16 {
            y[(y0 + r) * w + x0..(y0 + r) * w + x0 + 16].copy_from_slice(&s[r * 16..r * 16 + 16]);
        }
        for r in 0..8 {
            let o = (y0 / 2 + r) * (w / 2) + x0 / 2;
            u[o..o + 8].copy_from_slice(&s[256 + r * 8..256 + r * 8 + 8]);
            v[o..o + 8].copy_from_slice(&s[320 + r * 8..320 + r * 8 + 8]);
        }
    }
    (y, u, v)
}

fn build_and_check(cabac: bool, threads: usize) {
    let mut seed = 99u32;
    let pcm_pic = |seed: &mut u32| (0..NMB).map(|_| Some(random_mb(seed))).collect::<Vec<_>>();
    let a = pcm_pic(&mut seed);
    let b = pcm_pic(&mut seed);
    let mixed: Vec<Option<Vec<u8>>> = (0..NMB).map(|i| if i % 3 == 1 { Some(random_mb(&mut seed)) } else { None }).collect();
    let skip = vec![None; NMB];
    let pics = vec![
        // 0: IDR, all PCM, marked long-term (LongTermFrameIdx 0)
        Pic { idr: true, frame_num: 0, ref_idc: 3, long_term_ref: true, mbs: a.clone(), ..Default::default() },
        // 1: all PCM, short-term
        Pic { frame_num: 1, ref_idc: 2, mbs: b.clone(), ..Default::default() },
        // 2: P_Skip from the long-term picture via modification idc 2 (default list would use pic 1)
        Pic { frame_num: 2, ref_idc: 2, mods: vec![(2, 0)], mbs: skip.clone(), ..Default::default() },
        // 3: frame_num gap (3, 4 missing); P_Skip from pic 2 (PicNum 2) via idc 0: 5 - (2 + 1) = 2
        Pic { frame_num: 5, ref_idc: 2, mods: vec![(0, 2)], mbs: skip.clone(), ..Default::default() },
        // 4: mixed PCM / P_Skip from pic 1 (PicNum 1: 6 - 5); MMCO 4 raises MaxLongTermFrameIdx to 2,
        //    MMCO 2 drops long-term 0, MMCO 3 makes pic 1 (difference 6 - 1 - 1 = 4) long-term index 1
        Pic { frame_num: 6, ref_idc: 2, mods: vec![(0, 4)], mmcos: vec![(4, 3, 0), (2, 0, 0), (3, 4, 1)], mbs: mixed.clone(), ..Default::default() },
        // 5: non-reference P_Skip from long-term index 1 (= pic 1)
        Pic { frame_num: 7, ref_idc: 0, mods: vec![(2, 1)], mbs: skip.clone(), ..Default::default() },
        // 6: P_Skip with the default list: short-term by descending PicNum -> pic 4 first
        Pic { frame_num: 7, ref_idc: 2, mmcos: vec![(1, 0, 0), (6, 0, 2)], mbs: skip.clone(), ..Default::default() },
        // 7: MMCO 5 picture made of PCM, then 8: P_Skip referencing it (only reference left)
        Pic { frame_num: 8, ref_idc: 2, mmcos: vec![(5, 0, 0)], mbs: pcm_pic(&mut seed), ..Default::default() },
    ];
    let mut stream = sps(if cabac { 77 } else { 66 }, 8, true, PocType::T2);
    stream.extend(pps(cabac, false));
    for p in &pics {
        stream.extend(slice_nal(p, cabac));
    }
    let after5 = Pic { frame_num: 1, ref_idc: 2, mbs: skip.clone(), ..Default::default() };
    stream.extend(slice_nal(&after5, cabac));

    // expected output
    let pa = render(&a, None);
    let pb = render(&b, None);
    let p2 = pa.clone();
    let p3 = p2.clone();
    let p4 = render(&mixed, Some(&pb));
    let p5 = pb.clone();
    let p6 = p4.clone();
    let p7 = render(&pics[7].mbs, None);
    let p8 = p7.clone();
    let expected = [pa, pb, p2, p3, p4, p5, p6, p7, p8];

    if threads == 1 {
        cross_check_ffmpeg(&stream, &expected, if cabac { "synth_cabac" } else { "synth_cavlc" });
    }
    let mut dec = Decoder::with_threads(threads);
    let mut out = dec.decode(&stream, 0).unwrap();
    out.extend(dec.flush());
    assert!(dec.take_error().is_none());
    assert_eq!(out.len(), expected.len(), "cabac={cabac}");
    for (i, (p, e)) in out.iter().zip(expected.iter()).enumerate() {
        assert_eq!((p.width as usize, p.height as usize), (MB_W * 16, MB_H * 16));
        assert!(p.y == e.0 && p.u == e.1 && p.v == e.2, "picture {i} differs (cabac={cabac}, threads={threads})");
    }
    let s = dec.stats();
    assert!(s.mb_pcm > 0 && s.frame_num_gaps == 1 && s.long_term_marks >= 3, "{s:?}");
}

type Planes3 = (Vec<u8>, Vec<u8>, Vec<u8>);

/// Validate the hand-built stream and its expected output with ffmpeg when available.
fn cross_check_ffmpeg(stream: &[u8], expected: &[Planes3], name: &str) {
    let Some(ff) = filmcraft_testkit::ffmpeg_or_skip(name) else { return };
    let dir = filmcraft_testkit::fixtures_dir("h264");
    let src = filmcraft_testkit::temp_path(&dir.join(format!("{name}.h264")));
    let dst = filmcraft_testkit::temp_path(&dir.join(format!("{name}.yuv")));
    std::fs::write(&src, stream).unwrap();
    let ok = std::process::Command::new(&ff)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&src)
        .args(["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dst)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    assert!(ok, "ffmpeg failed to decode {name}");
    let got = std::fs::read(&dst).unwrap();
    let _ = std::fs::remove_file(&src);
    let _ = std::fs::remove_file(&dst);
    let want: Vec<u8> = expected.iter().flat_map(|(y, u, v)| y.iter().chain(u).chain(v).copied()).collect();
    assert!(got == want, "ffmpeg output differs from the expected output of {name} ({} vs {} bytes)", got.len(), want.len());
}

#[test]
fn pcm_long_term_mmco_gaps_cavlc() {
    build_and_check(false, 1);
    build_and_check(false, 4);
}

#[test]
fn pcm_long_term_mmco_gaps_cabac() {
    build_and_check(true, 1);
    build_and_check(true, 4);
}

fn weighted_check(cabac: bool, threads: usize) {
    let mut seed = 7u32;
    let a: Vec<Option<Vec<u8>>> = (0..NMB).map(|_| Some(random_mb(&mut seed))).collect();
    let b: Vec<Option<Vec<u8>>> = (0..NMB).map(|_| Some(random_mb(&mut seed))).collect();
    let skip = vec![None; NMB];
    let wp = [[(20, 3), (5, -1), (12, 2)], [(0, 0); 3]];
    let wb = [[(40, -2), (4, 1), (9, -3)], [(30, 7), (6, 0), (2, 4)]];
    // decoding order: I (POC 0), P (POC 8), B (POC 4, both refs), non-reference weighted P_Skip (POC 12)
    let pics = [
        Pic { idr: true, frame_num: 0, ref_idc: 3, poc_lsb: Some(0), mbs: a.clone(), ..Default::default() },
        Pic { frame_num: 1, ref_idc: 2, poc_lsb: Some(8), mbs: b.clone(), wt: Some(wp), ..Default::default() },
        Pic { frame_num: 2, ref_idc: 0, poc_lsb: Some(4), b: true, mbs: skip.clone(), wt: Some(wb), ..Default::default() },
        Pic { frame_num: 2, ref_idc: 0, poc_lsb: Some(12), mbs: skip.clone(), wt: Some(wp), ..Default::default() },
    ];
    let mut stream = sps(77, 2, false, PocType::T0);
    stream.extend(pps(cabac, true));
    for p in &pics {
        stream.extend(slice_nal(p, cabac));
    }
    let pa = render(&a, None);
    let pb = render(&b, None);
    let clip = |v: i32| v.clamp(0, 255) as u8;
    let mix = |x: &[u8], y: &[u8], c: usize| -> Vec<u8> {
        let (w0, o0) = wb[0][c];
        let (w1, o1) = wb[1][c];
        let d = if c == 0 { 5 } else { 3 };
        x.iter().zip(y).map(|(&p, &q)| clip(((p as i32 * w0 + q as i32 * w1 + (1 << d)) >> (d + 1)) + ((o0 + o1 + 1) >> 1))).collect()
    };
    let single = |x: &[u8], c: usize| -> Vec<u8> {
        let (w, o) = wp[0][c];
        let d = if c == 0 { 5 } else { 3 };
        x.iter().map(|&p| clip(((p as i32 * w + (1 << (d - 1))) >> d) + o)).collect()
    };
    let pbi = (mix(&pa.0, &pb.0, 0), mix(&pa.1, &pb.1, 1), mix(&pa.2, &pb.2, 2));
    let pp = (single(&pb.0, 0), single(&pb.1, 1), single(&pb.2, 2));
    let expected = [pa, pbi, pb, pp];
    if threads == 1 {
        cross_check_ffmpeg(&stream, &expected, if cabac { "synth_wp_cabac" } else { "synth_wp_cavlc" });
    }
    let mut dec = Decoder::with_threads(threads);
    let mut out = dec.decode(&stream, 0).unwrap();
    out.extend(dec.flush());
    assert!(dec.take_error().is_none());
    assert_eq!(out.len(), 4);
    for (i, (p, e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(p.y == e.0 && p.u == e.1 && p.v == e.2, "picture {i} differs (cabac={cabac}, threads={threads})");
    }
}

#[test]
fn explicit_weighted_prediction_p_and_b_skip() {
    for cabac in [false, true] {
        weighted_check(cabac, 1);
        weighted_check(cabac, 3);
    }
}

/// POC type 1: non-reference pictures are displayed before the preceding reference picture.
#[test]
fn poc_type1_reordering() {
    for cabac in [false, true] {
        let mut seed = 5u32;
        let contents: Vec<Vec<Option<Vec<u8>>>> = (0..5).map(|_| (0..NMB).map(|_| Some(random_mb(&mut seed))).collect()).collect();
        // (frame_num, ref_idc): POCs 0, 4, 1, 8, 5
        let spec = [(0, 3), (1, 2), (2, 0), (2, 2), (3, 0)];
        let mut stream = sps(77, 2, false, PocType::T1);
        stream.extend(pps(cabac, false));
        for (i, &(frame_num, ref_idc)) in spec.iter().enumerate() {
            let p = Pic { idr: i == 0, frame_num, ref_idc, poc_delta: Some(0), mbs: contents[i].clone(), ..Default::default() };
            stream.extend(slice_nal(&p, cabac));
        }
        let expected_order = [0usize, 2, 1, 4, 3];
        let expected_poc = [0, 1, 4, 5, 8];
        for threads in [1, 2] {
            let mut dec = Decoder::with_threads(threads);
            let mut out = dec.decode(&stream, 0).unwrap();
            out.extend(dec.flush());
            assert_eq!(out.len(), 5);
            for (k, p) in out.iter().enumerate() {
                let e = render(&contents[expected_order[k]], None);
                assert_eq!(p.poc, expected_poc[k]);
                assert!(p.y == e.0 && p.u == e.1 && p.v == e.2, "output {k} (cabac={cabac})");
            }
        }
    }
}

#[test]
fn avcc_accepts_generated_sps_and_pps() {
    let sps = sps(66, 4, false, PocType::T2);
    let pps = pps(false, false);
    let (sps, pps) = (&sps[4..], &pps[4..]);
    let mut avcc = vec![1, 66, 0, 30, 0xff, 0xe1];
    avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(sps);
    avcc.push(1);
    avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
    avcc.extend_from_slice(pps);
    let decoder = Decoder::from_avcc(&avcc).unwrap();
    assert_eq!(decoder.nal_length_size(), Some(4));
}

/// A slice NAL in the avcC PPS list was handled like stream data and started a picture; every
/// entry is now checked (SPS first, PPS after, none empty) before any is handled, and a refused
/// record leaves the NAL length size unset.
#[test]
fn avcc_rejects_a_valid_picture_slice_without_starting_a_picture() {
    let sequence = sps(66, 4, false, PocType::T0);
    let picture = pps(false, false);
    let slice = slice_nal(&Pic { idr: true, frame_num: 0, ref_idc: 3, mbs: vec![Some(vec![128; 384]); NMB], poc_lsb: Some(0), ..Default::default() }, false);
    let mut avcc = vec![1, 66, 0, 30, 0xff, 0xe1];
    let nal = &sequence[4..];
    avcc.extend_from_slice(&(nal.len() as u16).to_be_bytes());
    avcc.extend_from_slice(nal);
    avcc.push(2);
    avcc.extend_from_slice(&((picture.len() - 4) as u16).to_be_bytes());
    avcc.extend_from_slice(&picture[4..]);
    avcc.extend_from_slice(&((slice.len() - 4) as u16).to_be_bytes());
    avcc.extend_from_slice(&slice[4..]);

    let mut decoder = Decoder::with_threads(1);
    assert!(decoder.configure_avcc(&avcc).is_err());
    assert!(decoder.flush().is_empty());
    assert_eq!(decoder.stats().pictures, 0);
    assert_eq!(decoder.nal_length_size(), None);

    // An empty entry is refused too, in either list.
    let mut empty = vec![1, 66, 0, 30, 0xff, 0xe1, 0, 0, 0];
    assert!(Decoder::with_threads(1).configure_avcc(&empty).is_err());
    empty.truncate(6);
    empty.extend_from_slice(&(nal.len() as u16).to_be_bytes());
    empty.extend_from_slice(nal);
    empty.extend_from_slice(&[1, 0, 0]);
    assert!(Decoder::with_threads(1).configure_avcc(&empty).is_err());
}

/// A slice_qp_delta at the end of the Exp-Golomb range is an error, not an overflow of
/// pic_init_qp + slice_qp_delta.
#[test]
fn slice_qp_addition_does_not_overflow() {
    for qp_delta in [26, i32::MAX, -i32::MAX] {
        let mut w = BitWriter::new();
        w.write_ue(0); // first_mb
        w.write_ue(7); // I slice
        w.write_ue(0); // pps id
        w.write_bits(0, 4); // frame_num
        w.write_ue(0); // idr_pic_id
        w.write_bit(false); // no_output_of_prior_pics
        w.write_bit(false); // long_term_reference
        w.write_se(qp_delta);
        w.write_ue(1); // disable_deblocking_filter_idc
        w.rbsp_trailing();
        let mut stream = sps(66, 1, false, PocType::T2);
        stream.extend(pps(false, false));
        stream.extend(nal(3, 5, &w.finish()));
        let r = Decoder::with_threads(1).decode(&stream, 0).map(|pics| pics.len());
        assert!(matches!(&r, Err(crate::Error::Invalid(m)) if m.contains("slice QP")), "slice_qp_delta {qp_delta}: {r:?}");
    }
}

/// A chunk may start only so many pictures (64), however cheap each one is to ask for: 64
/// one-picture access units in one call decode, 65 are refused.
#[test]
fn a_chunk_with_too_many_pictures_is_refused() {
    let chunk = |pictures: u32| {
        let mut stream = sps(66, 1, false, PocType::T2);
        stream.extend(pps(false, false));
        for i in 0..pictures {
            let p = Pic { idr: i == 0, frame_num: i % 16, ref_idc: 3, mbs: vec![Some(vec![128; 384]); NMB], ..Default::default() };
            stream.extend(slice_nal(&p, false));
        }
        stream
    };
    for threads in [1, 2] {
        let mut dec = Decoder::with_threads(threads);
        let mut out = dec.decode(&chunk(64), 0).unwrap();
        out.extend(dec.flush());
        assert_eq!(out.len(), 64);
        let mut dec = Decoder::with_threads(threads);
        let r = dec.decode(&chunk(65), 0).map(|pics| pics.len());
        assert!(matches!(&r, Err(crate::Error::Invalid(m)) if m.contains("too many pictures")), "{r:?}");
        // The next call starts counting again.
        assert!(dec.decode(&chunk(1), 0).is_ok());
    }
}

/// A second slice that covers macroblocks the picture already decoded is refused instead of
/// doing the picture's prediction work again.
#[test]
fn a_slice_overlapping_decoded_macroblocks_is_refused() {
    let mut stream = sps(66, 1, false, PocType::T2);
    stream.extend(pps(false, false));
    let p = Pic { idr: true, frame_num: 0, ref_idc: 3, mbs: vec![Some(vec![128; 384]); NMB], ..Default::default() };
    stream.extend(slice_nal(&p, false));
    // A second slice of the same picture from macroblock 1: every one of them is decoded already.
    stream.extend(slice_nal(&Pic { first_mb: 1, mbs: vec![Some(vec![128; 384]); NMB - 1], ..p }, false));
    for threads in [1, 2] {
        let mut dec = Decoder::with_threads(threads);
        let first = dec.decode(&stream, 0).err();
        let _ = dec.flush();
        let err = first.or_else(|| dec.take_error());
        assert!(matches!(&err, Some(crate::Error::Invalid(m)) if m.contains("overlaps")), "{err:?}");
    }
}

// ---------------------------------------------------------------------------------------------
// High 10 sample formats: I_PCM at every supported bit depth (luma and chroma apart, which
// libx264 cannot produce), P_Skip copies, cropping, and hostile sequence headers.
// ---------------------------------------------------------------------------------------------

/// Chroma sampling and bit depths of a synthetic stream.
#[derive(Clone, Copy, Debug)]
struct Fmt {
    c422: bool,
    bd_y: u32,
    bd_c: u32,
}

impl Fmt {
    /// Chroma lines per macroblock.
    fn mbh_c(self) -> usize {
        if self.c422 { 16 } else { 8 }
    }
}

/// A High 4:2:2 profile SPS (profile_idc 122) with `chroma_idc` and the bit depths as coded
/// (bit_depth_*_minus8), `mbs` macroblocks, POC type 2, optional cropping (left, right, top,
/// bottom in crop units).
fn sps_fmt(chroma_idc: u32, depth_minus8: (u32, u32), mbs: (u32, u32), crop: Option<(u32, u32, u32, u32)>) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.write_bits(122, 8);
    w.write_bits(0, 8);
    w.write_bits(40, 8);
    w.write_ue(0); // sps id
    w.write_ue(chroma_idc);
    if chroma_idc == 3 {
        w.write_bit(false); // separate_colour_plane_flag
    }
    w.write_ue(depth_minus8.0);
    w.write_ue(depth_minus8.1);
    w.write_bit(false); // qpprime_y_zero_transform_bypass_flag
    w.write_bit(false); // seq_scaling_matrix_present_flag
    w.write_ue(0); // log2_max_frame_num - 4
    w.write_ue(2); // POC type 2
    w.write_ue(1); // max_num_ref_frames
    w.write_bit(false);
    w.write_ue(mbs.0 - 1);
    w.write_ue(mbs.1 - 1);
    w.write_bit(true); // frame_mbs_only
    w.write_bit(true); // direct_8x8_inference
    match crop {
        Some((l, r, t, b)) => {
            w.write_bit(true);
            for v in [l, r, t, b] {
                w.write_ue(v);
            }
        }
        None => w.write_bit(false),
    }
    w.write_bit(false); // vui
    w.rbsp_trailing();
    nal(3, 7, &w.finish())
}

/// Random I_PCM samples of one macroblock (256 luma, then MbWidthC * MbHeightC Cb and Cr, raster)
/// within the bit depths.
fn random_pcm(f: Fmt, seed: &mut u32) -> Vec<u16> {
    let n_c = 8 * f.mbh_c();
    (0..256 + 2 * n_c)
        .map(|i| {
            *seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let bd = if i < 256 { f.bd_y } else { f.bd_c };
            ((*seed >> 8) % (1 << bd)) as u16
        })
        .collect()
}

/// pcm_sample_luma / pcm_sample_chroma as coded: BitDepth bits each, MSB first (7.3.5).
fn pack_pcm(f: Fmt, samples: &[u16]) -> Vec<u8> {
    let mut w = BitWriter::new();
    for (i, &s) in samples.iter().enumerate() {
        w.write_bits(s as u32, if i < 256 { f.bd_y } else { f.bd_c });
    }
    w.finish()
}

type Planes16 = (Vec<u16>, Vec<u16>, Vec<u16>);

/// Expected (uncropped) planes of a picture of PCM blocks (`None` = copy from `reference`).
fn render_fmt(f: Fmt, mbs: &[Option<Vec<u16>>], reference: Option<&Planes16>) -> Planes16 {
    let (w, h) = (MB_W * 16, MB_H * 16);
    let (cw, ch, lc) = (w / 2, MB_H * f.mbh_c(), f.mbh_c());
    let mut out = match reference {
        Some(r) => r.clone(),
        None => (vec![0; w * h], vec![0; cw * ch], vec![0; cw * ch]),
    };
    for (i, mb) in mbs.iter().enumerate() {
        let Some(s) = mb else { continue };
        let (mx, my) = (i % MB_W, i / MB_W);
        for r in 0..16 {
            let o = (my * 16 + r) * w + mx * 16;
            out.0[o..o + 16].copy_from_slice(&s[r * 16..r * 16 + 16]);
        }
        for r in 0..lc {
            let o = (my * lc + r) * cw + mx * 8;
            out.1[o..o + 8].copy_from_slice(&s[256 + r * 8..256 + r * 8 + 8]);
            out.2[o..o + 8].copy_from_slice(&s[256 + 8 * lc + r * 8..256 + 8 * lc + r * 8 + 8]);
        }
    }
    out
}

/// The stream (IDR of PCM, P of PCM and P_Skip, P_Skip, non-reference mixed P) and its expected
/// pictures, `crop_bottom` luma lines cropped off the bottom.
fn format_stream(f: Fmt, cabac: bool, crop_bottom: u32) -> (Vec<u8>, Vec<Planes16>) {
    let mut seed = 1234 + f.bd_y * 7 + f.bd_c * 3 + f.c422 as u32;
    let pcm =
        |seed: &mut u32, every: usize| -> Vec<Option<Vec<u16>>> { (0..NMB).map(|i| if i % every == 0 { Some(random_pcm(f, seed)) } else { None }).collect() };
    let a = pcm(&mut seed, 1);
    let b = pcm(&mut seed, 3);
    let skip = vec![None; NMB];
    let d = pcm(&mut seed, 2);
    let packed = |m: &[Option<Vec<u16>>]| m.iter().map(|s| s.as_ref().map(|s| pack_pcm(f, s))).collect::<Vec<_>>();
    let pics = [
        Pic { idr: true, frame_num: 0, ref_idc: 3, mbs: packed(&a), ..Default::default() },
        Pic { frame_num: 1, ref_idc: 2, mbs: packed(&b), ..Default::default() },
        Pic { frame_num: 2, ref_idc: 2, mbs: packed(&skip), ..Default::default() },
        Pic { frame_num: 3, ref_idc: 0, mbs: packed(&d), ..Default::default() },
    ];
    // crop units: 2 lines in 4:2:0, 1 in 4:2:2
    let crop = (crop_bottom > 0).then_some((0, 0, 0, if f.c422 { crop_bottom } else { crop_bottom / 2 }));
    let mut stream = sps_fmt(1 + f.c422 as u32, (f.bd_y - 8, f.bd_c - 8), (MB_W as u32, MB_H as u32), crop);
    stream.extend(pps(cabac, false));
    for p in &pics {
        stream.extend(slice_nal(p, cabac));
    }
    let pa = render_fmt(f, &a, None);
    let pb = render_fmt(f, &b, Some(&pa));
    let pc = pb.clone();
    let pd = render_fmt(f, &d, Some(&pc));
    let (w, h) = (MB_W * 16, MB_H * 16 - crop_bottom as usize);
    let ch = if f.c422 { h } else { h / 2 };
    let crop_planes = |p: Planes16| -> Planes16 { (p.0[..w * h].to_vec(), p.1[..w / 2 * ch].to_vec(), p.2[..w / 2 * ch].to_vec()) };
    (stream, [pa, pb, pc, pd].into_iter().map(crop_planes).collect())
}

fn check_format(f: Fmt, cabac: bool, threads: usize, crop_bottom: u32) {
    let (stream, expected) = format_stream(f, cabac, crop_bottom);
    let mut dec = Decoder::with_threads(threads);
    let mut out = dec.decode(&stream, 0).unwrap_or_else(|e| panic!("{f:?} cabac {cabac}: {e}"));
    out.extend(dec.flush());
    assert!(dec.take_error().is_none(), "{f:?}");
    assert_eq!(out.len(), expected.len(), "{f:?}");
    let ch = expected[0].1.len() / (MB_W * 8);
    for (i, (p, e)) in out.iter().zip(&expected).enumerate() {
        let label = format!("{f:?} cabac {cabac} threads {threads} crop {crop_bottom}: picture {i}");
        let size = (p.width as usize, p.height as usize, p.chroma_width as usize, p.chroma_height as usize);
        assert_eq!(size, (MB_W * 16, e.0.len() / (MB_W * 16), MB_W * 8, ch), "{label}");
        let chroma = if f.c422 { crate::ChromaFormat::C422 } else { crate::ChromaFormat::C420 };
        assert_eq!((p.chroma, p.bit_depth, p.bit_depth_chroma), (chroma, f.bd_y, f.bd_c), "{label}");
        assert_eq!(p.y.as_u8().is_some(), f.bd_y == 8 && f.bd_c == 8, "{label}: storage type");
        for (plane, want, name) in [(&p.y, &e.0, "Y"), (&p.u, &e.1, "U"), (&p.v, &e.2, "V")] {
            let got: Vec<u16> = (0..plane.len()).map(|k| plane.get(k)).collect();
            assert!(&got == want, "{label}: plane {name} differs");
        }
    }
}

/// I_PCM pictures and P_Skip copies decode exactly at every supported combination of luma /
/// chroma bit depth (8, 9, 10, mixed), CAVLC and CABAC (whose engine restarts after the PCM
/// bytes), single- and multi-threaded, and cropped. Before, every one but 8/8 was refused as
/// unsupported (only 8-bit).
#[test]
#[allow(clippy::single_element_loop)] // 4:2:2 follows
fn pcm_pictures_at_every_depth_and_chroma_format() {
    for c422 in [false] {
        for (bd_y, bd_c) in [(8, 8), (10, 10), (9, 9), (8, 10), (10, 8), (9, 10)] {
            let f = Fmt { c422, bd_y, bd_c };
            for cabac in [false, true] {
                for threads in [1, 4] {
                    check_format(f, cabac, threads, 0);
                }
                check_format(f, cabac, 1, if c422 { 1 } else { 2 });
            }
        }
    }
}

/// The synthetic 10-bit PCM stream matches ffmpeg's decode too.
#[test]
#[allow(clippy::single_element_loop)] // 4:2:2 follows
fn pcm_formats_match_ffmpeg() {
    let Some(ff) = filmcraft_testkit::ffmpeg_or_skip("pcm formats") else { return };
    for (c422, bd) in [(false, 10)] {
        let f = Fmt { c422, bd_y: bd, bd_c: bd };
        let (stream, expected) = format_stream(f, true, if c422 { 1 } else { 0 });
        let pix = format!("yuv{}p{}", if c422 { "422" } else { "420" }, if bd == 8 { String::new() } else { format!("{bd}le") });
        let dir = filmcraft_testkit::fixtures_dir("h264");
        let src = filmcraft_testkit::temp_path(&dir.join(format!("synth_pcm_{pix}.h264")));
        let dst = filmcraft_testkit::temp_path(&dir.join(format!("synth_pcm_{pix}.yuv")));
        std::fs::write(&src, &stream).unwrap();
        let ok = std::process::Command::new(&ff)
            .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
            .arg(&src)
            .args(["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", &pix])
            .arg(&dst)
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        assert!(ok, "ffmpeg failed to decode the {pix} stream");
        let got = std::fs::read(&dst).unwrap();
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dst);
        let bytes = |s: u16| if bd == 8 { vec![s as u8] } else { s.to_le_bytes().to_vec() };
        let want: Vec<u8> = expected.iter().flat_map(|(y, u, v)| y.iter().chain(u).chain(v).flat_map(|&s| bytes(s)).collect::<Vec<u8>>()).collect();
        assert!(got == want, "ffmpeg's {pix} decode differs from the expected output ({} vs {} bytes)", got.len(), want.len());
    }
}

/// Sequence headers of every chroma format and bit depth a stream can code, at odd and huge
/// sizes, with crops up to the picture: no panic in the parser or a decoding job; what this
/// decoder does not implement (4:2:2, 4:4:4, monochrome, more than 10 bits) is `Unsupported`,
/// the rest decode or fail with an ordinary error.
#[test]
fn hostile_high_sequence_headers_never_panic() {
    let slice = slice_nal(&Pic { idr: true, frame_num: 0, ref_idc: 3, mbs: vec![Some(vec![0x5a; 640]); NMB], ..Default::default() }, false);
    let slice_cabac = slice_nal(&Pic { idr: true, frame_num: 0, ref_idc: 3, mbs: vec![Some(vec![0xa5; 640]); NMB], ..Default::default() }, true);
    for chroma in 0..=3 {
        for depths in [(0, 0), (2, 2), (0, 2), (2, 0), (1, 2), (3, 2), (2, 6), (6, 6), (7, 0)] {
            for (mbs, crop) in [
                ((MB_W as u32, MB_H as u32), None),
                ((MB_W as u32, MB_H as u32), Some((0, 1, 0, 1))),
                ((MB_W as u32, MB_H as u32), Some((7, 0, 0, 23))),
                ((1, 1), None),
                ((3, 1), Some((1, 0, 3, 0))),
                ((1024, 1024), None),
                ((256, 192), None),
                ((256, 196), None),
            ] {
                for (cabac, data) in [(false, &slice), (true, &slice_cabac)] {
                    let mut stream = sps_fmt(chroma, depths, mbs, crop);
                    stream.extend(pps(cabac, false));
                    stream.extend_from_slice(data);
                    for threads in [1, 3] {
                        let mut dec = Decoder::with_threads(threads);
                        let first = dec.decode(&stream, 0).err();
                        let _ = dec.flush();
                        let err = first.or_else(|| dec.take_error());
                        let label = format!("chroma {chroma} depths {depths:?} mbs {mbs:?} crop {crop:?} cabac {cabac} threads {threads}");
                        assert!(!matches!(&err, Some(crate::Error::Invalid(m)) if m.contains("internal error")), "{label}: a decoding job panicked");
                        // an SPS the parser refuses (depth 15, crop past the picture) never gets this far
                        if crate::params::Sps::parse(&filmcraft_bitstream::unescape_rbsp(&sps_fmt(chroma, depths, mbs, crop)[5..])).is_err() {
                            continue;
                        }
                        let unsupported_format = chroma != 1 || depths.0 > 2 || depths.1 > 2;
                        if unsupported_format {
                            assert!(matches!(&err, Some(crate::Error::Unsupported(_))), "{label}: {err:?}");
                        } else if mbs == (256, 196) && chroma == 2 && depths != (0, 0) {
                            // over the picture byte budget: 4096 x 3136 4:2:2 at 16 bits per sample
                            assert!(matches!(&err, Some(crate::Error::Unsupported(m)) if m.contains("memory limit")), "{label}: {err:?}");
                        }
                    }
                }
            }
        }
    }
}

/// The picture-size limit counts the bytes of every plane and per sample: the largest 8-bit 4:2:0
/// picture's memory (2^25 luma samples) is the budget, so 10-bit 4:2:0 fits up to 2^24 luma
/// samples; one macroblock row more is refused.
#[test]
fn the_picture_budget_counts_bytes_per_sample_and_every_plane() {
    let parse = |chroma: u32, depth: u32, mbs: (u32, u32)| {
        let s = sps_fmt(chroma, (depth, depth), mbs, None);
        crate::params::Sps::parse(&filmcraft_bitstream::unescape_rbsp(&s[5..])).unwrap().check_supported()
    };
    let limit = |r: crate::Result<()>| matches!(r, Err(crate::Error::Unsupported(m)) if m.contains("memory limit"));
    // 8-bit 4:2:0: 2^25 luma samples (8192 x 4096)
    assert!(parse(1, 0, (512, 256)).is_ok());
    assert!(limit(parse(1, 0, (512, 257))));
    // 10-bit 4:2:0: 2^24 (4096 x 4096)
    assert!(parse(1, 2, (256, 256)).is_ok());
    assert!(limit(parse(1, 2, (256, 257))));
    // 8-bit luma with 10-bit chroma is stored in 16 bits too
    let s = sps_fmt(1, (0, 2), (256, 257), None);
    assert!(limit(crate::params::Sps::parse(&filmcraft_bitstream::unescape_rbsp(&s[5..])).unwrap().check_supported()));
}

/// A sequence that switches sample format mid-stream (8-bit, then 10-bit on a non-IDR picture,
/// then a 10-bit IDR) flushes its references: the inter picture after the switch has none (an
/// error, not a read of the old 8-bit frames as 16-bit; the picture it started is output
/// undecoded, in the new format), the IDR decodes.
#[test]
fn a_format_switch_mid_stream_drops_the_old_references() {
    let f8 = Fmt { c422: false, bd_y: 8, bd_c: 8 };
    let f10 = Fmt { c422: false, bd_y: 10, bd_c: 10 };
    let mut seed = 3u32;
    let idr = |f: Fmt, seed: &mut u32| Pic {
        idr: true,
        frame_num: 0,
        ref_idc: 3,
        mbs: (0..NMB).map(|_| Some(pack_pcm(f, &random_pcm(f, seed)))).collect(),
        ..Default::default()
    };
    let mut stream = sps_fmt(1, (0, 0), (MB_W as u32, MB_H as u32), None);
    stream.extend(pps(true, false));
    stream.extend(slice_nal(&idr(f8, &mut seed), true));
    let mut switch = sps_fmt(1, (2, 2), (MB_W as u32, MB_H as u32), None);
    switch.extend(pps(true, false));
    switch.extend(slice_nal(&Pic { frame_num: 1, ref_idc: 2, mbs: vec![None; NMB], ..Default::default() }, true));
    let idr10 = slice_nal(&idr(f10, &mut seed), true);
    for threads in [1, 3] {
        let mut dec = Decoder::with_threads(threads);
        let mut out = dec.decode(&stream, 0).unwrap();
        let switched = dec.decode(&switch, 1);
        assert!(matches!(&switched, Err(crate::Error::Invalid(m)) if m.contains("no reference")), "{switched:?}");
        out.extend(dec.decode(&idr10, 2).unwrap());
        out.extend(dec.flush());
        assert!(dec.take_error().is_none());
        assert_eq!(out.len(), 3);
        assert!(out[0].y.as_u8().is_some() && out[0].chroma == crate::ChromaFormat::C420);
        for p in &out[1..] {
            assert!(p.y.as_u16().is_some() && p.chroma == crate::ChromaFormat::C420 && p.bit_depth == 10);
        }
    }
}
