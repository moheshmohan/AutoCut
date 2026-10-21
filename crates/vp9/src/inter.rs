//! Block inter prediction process (8.5.2.4): separable 8-tap sub-sample interpolation with edge
//! clamping, for unscaled (step 16) and scaled references.

use crate::frame::Frame;
use crate::tables::SUBPEL_FILTERS;

/// Filter taps of `filter` (0..3) at sub-sample position `frac` (0..15).
#[inline(always)]
fn taps(filter: u8, frac: usize) -> &'static [i16; 8] {
    const ZERO: [i16; 8] = [0, 0, 0, 128, 0, 0, 0, 0];
    let o = (filter as usize * 16 + frac) * 8;
    SUBPEL_FILTERS.get(o..o + 8).and_then(|t| t.first_chunk::<8>()).unwrap_or(&ZERO)
}

/// Plane `plane` of a reference frame, valid for x <= last_x, y <= last_y. Rows are read through
/// [`Frame::row`] / [`Frame::span`], which wait until the reference has published them.
pub struct RefPlane<'a> {
    pub frame: &'a Frame,
    pub plane: usize,
    pub last_x: i32,
    pub last_y: i32,
}

impl RefPlane<'_> {
    #[inline]
    fn row(&self, y: usize) -> &[u16] {
        self.frame.row(self.plane, y)
    }
}

/// Predict a `w` x `h` block whose top-left sample is at (`x`, `y`) in 1/16 sample units of the
/// reference, stepping `step_x` / `step_y` (16 = unscaled). Output row `i` goes to
/// `out[i * os..i * os + w]`. `tmp` must hold at least `(h * step_y / 16 + 8) * w` entries, `win`
/// at least `(w + 7) * (h + 7)` (edge-clamped source window).
#[allow(clippy::too_many_arguments)]
pub fn predict(
    r: &RefPlane,
    x: i32,
    y: i32,
    step_x: i32,
    step_y: i32,
    w: usize,
    h: usize,
    filter: u8,
    bit_depth: u8,
    out: &mut [u16],
    os: usize,
    tmp: &mut [u16],
    win: &mut [u16],
) {
    if step_x == 16 && step_y == 16 {
        if bit_depth == 8 {
            predict_unscaled::<false>(r, x, y, w, h, filter, bit_depth, out, os, tmp, win);
        } else {
            predict_unscaled::<true>(r, x, y, w, h, filter, bit_depth, out, os, tmp, win);
        }
    } else {
        predict_scaled(r, x, y, step_x, step_y, w, h, filter, bit_depth, out, os, tmp);
    }
}

/// Copy the (w + 7) x (h + 7) source window starting at (x0 - 3, y0 - 3) with edge clamping.
fn fetch_window(r: &RefPlane, x0: i32, y0: i32, w: usize, h: usize, win: &mut [u16]) -> usize {
    let ws = w + 7;
    let (xs, last_x) = (x0 - 3, r.last_x.max(0));
    // Window columns [0, left) lie left of the picture (they repeat column 0), [right, ws) right
    // of it (they repeat column last_x), [left, right) inside.
    let left = (-xs).clamp(0, ws as i32) as usize;
    let right = (last_x + 1 - xs).clamp(left as i32, ws as i32) as usize;
    let first = (xs + left as i32).max(0) as usize;
    for (row, dst) in win.chunks_exact_mut(ws).take(h + 7).enumerate() {
        let yy = (y0 - 3 + row as i32).clamp(0, r.last_y.max(0)) as usize;
        let src = r.row(yy);
        let Some((head, rest)) = dst.split_at_mut_checked(left) else { continue };
        let Some((mid, tail)) = rest.split_at_mut_checked(right - left) else { continue };
        head.fill(src.first().copied().unwrap_or(0));
        tail.fill(src.get(last_x as usize).copied().unwrap_or(0));
        if let Some(s) = src.get(first..first + mid.len()) {
            mid.copy_from_slice(s);
        }
    }
    ws
}

/// The source rows of one block: row `i` is reference row `y0 - 3 + i` (clamped to the picture),
/// from column `x0 - 3`, at least `w + 7` samples long.
enum Rows<'a> {
    /// Every row lies inside the picture and in one band: row `i` starts at `off + i * stride`.
    Span { data: &'a [u16], stride: usize, off: usize },
    /// Columns inside the picture, rows spread over several bands or clamped at the top / bottom.
    Bands { r: &'a RefPlane<'a>, y: i32, x: usize },
    /// Edge-clamped copy (columns outside the picture), `stride` = w + 7.
    Window { data: &'a [u16], stride: usize },
}

impl Rows<'_> {
    #[inline(always)]
    fn get(&self, i: usize) -> &[u16] {
        match *self {
            Rows::Span { data, stride, off } => data.get(off + i * stride..).unwrap_or(&[]),
            Rows::Bands { r, y, x } => r.row((y + i as i32).clamp(0, r.last_y.max(0)) as usize).get(x..).unwrap_or(&[]),
            Rows::Window { data, stride } => data.get(i * stride..).unwrap_or(&[]),
        }
    }
}

/// Unscaled prediction. `HBD`: bit depth 10 or 12 (32-bit sums), else 8 (16-bit sums).
#[allow(clippy::too_many_arguments)]
fn predict_unscaled<const HBD: bool>(
    r: &RefPlane,
    x: i32,
    y: i32,
    w: usize,
    h: usize,
    filter: u8,
    bit_depth: u8,
    out: &mut [u16],
    os: usize,
    tmp: &mut [u16],
    win: &mut [u16],
) {
    let (x0, y0) = (x >> 4, y >> 4);
    let (fx, fy) = ((x & 15) as usize, (y & 15) as usize);
    let max = (1i32 << bit_depth) - 1;
    // Columns x0 - 3 ..= x0 + w + 3 and rows y0 - 3 ..= y0 + h + 3 are read.
    let inside_x = x0 >= 3 && x0 + w as i32 + 3 <= r.last_x;
    let inside_y = y0 >= 3 && y0 + h as i32 + 3 <= r.last_y;
    let span = if inside_x && inside_y { r.frame.span(r.plane, (y0 - 3) as usize, (y0 + h as i32 + 4) as usize) } else { None };
    let rows = match span {
        Some((data, o)) => Rows::Span { data, stride: r.frame.strides[r.plane], off: o + (x0 - 3) as usize },
        None if inside_x => Rows::Bands { r, y: y0 - 3, x: (x0 - 3) as usize },
        None => {
            let ws = fetch_window(r, x0, y0, w, h, win);
            Rows::Window { data: win, stride: ws }
        }
    };
    let out_rows = out.chunks_mut(os.max(1)).take(h);
    match (fx, fy) {
        (0, 0) => {
            for (i, o) in out_rows.enumerate() {
                if let (Some(o), Some(s)) = (o.get_mut(..w), rows.get(i + 3).get(3..3 + w)) {
                    o.copy_from_slice(s);
                }
            }
        }
        (_, 0) => {
            let f = taps(filter, fx);
            for (i, o) in out_rows.enumerate() {
                filter_h::<HBD>(rows.get(i + 3), f, max, o.get_mut(..w).unwrap_or(&mut []));
            }
        }
        (0, _) => {
            let f = taps(filter, fy);
            let mut src: [&[u16]; 8] = std::array::from_fn(|t| rows.get(t).get(3..).unwrap_or(&[]));
            for (i, o) in out_rows.enumerate() {
                if i > 0 {
                    src.rotate_left(1);
                    src[7] = rows.get(i + 7).get(3..).unwrap_or(&[]);
                }
                filter_v::<HBD>(&src, f, max, o.get_mut(..w).unwrap_or(&mut []));
            }
        }
        _ => {
            let fh = taps(filter, fx);
            let fv = taps(filter, fy);
            for (i, t) in tmp.chunks_exact_mut(w.max(1)).take(h + 7).enumerate() {
                filter_h::<HBD>(rows.get(i), fh, max, t);
            }
            let tmp = &*tmp;
            for (i, o) in out_rows.enumerate() {
                let src: [&[u16]; 8] = std::array::from_fn(|t| tmp.get((i + t) * w..).unwrap_or(&[]));
                filter_v::<HBD>(&src, fv, max, o.get_mut(..w).unwrap_or(&mut []));
            }
        }
    }
}

/// Horizontal 8-tap filter of one row: `o[j]` from `s[j..j + 8]`. `o.len()` is the block width
/// (4, 8, 16, 32 or 64) and `s` holds at least `o.len() + 7` samples.
#[inline(always)]
fn filter_h<const HBD: bool>(s: &[u16], f: &[i16; 8], max: i32, o: &mut [u16]) {
    match o.len() {
        4 => h_row::<HBD, 4>(s, f, max, o),
        8 => h_row::<HBD, 8>(s, f, max, o),
        16 => h_row::<HBD, 16>(s, f, max, o),
        32 => h_row::<HBD, 32>(s, f, max, o),
        64 => h_row::<HBD, 64>(s, f, max, o),
        _ => {
            for (j, d) in o.iter_mut().enumerate() {
                let sum: i32 = (0..8).map(|t| f[t] as i32 * s.get(j + t).copied().unwrap_or(0) as i32).sum();
                *d = ((sum + 64) >> 7).max(0).min(max) as u16;
            }
        }
    }
}

/// Vertical 8-tap filter of one row: `o[j]` from `s[t][j]`, t = 0..8. `o.len()` is the block
/// width (4, 8, 16, 32 or 64) and every row holds at least `o.len()` samples.
#[inline(always)]
fn filter_v<const HBD: bool>(s: &[&[u16]; 8], f: &[i16; 8], max: i32, o: &mut [u16]) {
    match o.len() {
        4 => v_row::<HBD, 4>(s, f, max, o),
        8 => v_row::<HBD, 8>(s, f, max, o),
        16 => v_row::<HBD, 16>(s, f, max, o),
        32 => v_row::<HBD, 32>(s, f, max, o),
        64 => v_row::<HBD, 64>(s, f, max, o),
        _ => {
            for (j, d) in o.iter_mut().enumerate() {
                let sum: i32 = (0..8).map(|t| f[t] as i32 * s[t].get(j).copied().unwrap_or(0) as i32).sum();
                *d = ((sum + 64) >> 7).max(0).min(max) as u16;
            }
        }
    }
}

#[inline(always)]
fn h_row<const HBD: bool, const W: usize>(s: &[u16], f: &[i16; 8], max: i32, o: &mut [u16]) {
    let p = |t: usize| s.get(t..).and_then(|s| s.first_chunk::<W>());
    let (Some(o), Some(p0), Some(p1), Some(p2), Some(p3), Some(p4), Some(p5), Some(p6), Some(p7)) =
        (o.first_chunk_mut::<W>(), p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7))
    else {
        return;
    };
    taps8::<HBD, W>([p0, p1, p2, p3, p4, p5, p6, p7], f, max, o);
}

#[inline(always)]
fn v_row<const HBD: bool, const W: usize>(s: &[&[u16]; 8], f: &[i16; 8], max: i32, o: &mut [u16]) {
    let p = |t: usize| s[t].first_chunk::<W>();
    let (Some(o), Some(p0), Some(p1), Some(p2), Some(p3), Some(p4), Some(p5), Some(p6), Some(p7)) =
        (o.first_chunk_mut::<W>(), p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7))
    else {
        return;
    };
    taps8::<HBD, W>([p0, p1, p2, p3, p4, p5, p6, p7], f, max, o);
}

// Sample ranges. Reference samples lie in [0, 2^bitDepth - 1] (every reconstructed sample is
// clipped there), and so do the intermediate rows of the 2D filter (Round2 of the first pass,
// clipped). The positive taps of every VP9 filter add up to at most 182 and the negative ones to
// at least -54 (sharp filter, phase 8: -4 11 -23 80 80 -23 11 -4); the taps sum to 128.
//
// * 8-bit: sum + 64 lies in [-54 * 255 + 64, 182 * 255 + 64] = [-13706, 46474], a range of
//   60181 < 2^16 values, so the sum modulo 2^16 (u16 wrapping arithmetic, 8 lanes per 128-bit
//   vector) identifies it: wrapped, the non-negative sums stay below 0xC000 and the negative
//   ones (which clip to 0) land above it.
// * 10 / 12-bit: the range is 236 * 4095 ≈ 2^20, too wide for 16-bit sums. Samples (< 2^15) and
//   taps fit i16 exactly, so both operands are sign-extended i16 and every product and partial
//   sum (|sum| <= 182 * 4095 + 64 < 2^20) fits i32: the compiler uses widening 16 x 16 -> 32-bit
//   multiply-accumulates (8 products per pair of instructions) instead of widening the samples
//   to 32 bits first.
//
// The loops run over whole rows (taps outer, samples inner), so that the sample loop is the one
// the compiler vectorises; the first tap initialises the sums (a constant splat becomes a
// memset_pattern16 call on Darwin, which keeps the sums in memory).

/// Round2(sum of `f[t] * p[t][j]`, 7) clipped to [0, max] for each j.
#[inline(always)]
fn taps8<const HBD: bool, const W: usize>(p: [&[u16; W]; 8], f: &[i16; 8], max: i32, o: &mut [u16; W]) {
    if HBD {
        let c0 = f[0] as i32;
        let mut acc = [0i32; W];
        for j in 0..W {
            acc[j] = c0 * (p[0][j] as i16 as i32) + 64;
        }
        for t in 1..8 {
            let c = f[t] as i32;
            for j in 0..W {
                acc[j] += c * (p[t][j] as i16 as i32);
            }
        }
        for j in 0..W {
            // Saturate to u16 first (a saturating narrowing shift), then clip to max.
            o[j] = ((acc[j] >> 7).clamp(0, 0xFFFF) as u16).min(max as u16);
        }
    } else {
        let c0 = f[0] as u16;
        let mut acc = [0u16; W];
        for j in 0..W {
            acc[j] = c0.wrapping_mul(p[0][j]).wrapping_add(64);
        }
        for t in 1..8 {
            let c = f[t] as u16;
            for j in 0..W {
                acc[j] = acc[j].wrapping_add(c.wrapping_mul(p[t][j]));
            }
        }
        for j in 0..W {
            o[j] = if acc[j] < 0xC000 { (acc[j] >> 7).min(255) } else { 0 };
        }
    }
}

/// General process of 8.5.2.4 (any step).
#[allow(clippy::too_many_arguments)]
fn predict_scaled(
    r: &RefPlane,
    x: i32,
    y: i32,
    step_x: i32,
    step_y: i32,
    w: usize,
    h: usize,
    filter: u8,
    bit_depth: u8,
    out: &mut [u16],
    os: usize,
    tmp: &mut [u16],
) {
    let max = (1i32 << bit_depth) - 1;
    let ih = ((((h as i32 - 1) * step_y + 15) >> 4) + 8) as usize;
    for row in 0..ih {
        let yy = ((y >> 4) + row as i32 - 3).clamp(0, r.last_y.max(0)) as usize;
        let src = r.row(yy);
        for c in 0..w {
            let p = x + step_x * c as i32;
            let f = taps(filter, (p & 15) as usize);
            let mut acc = 0i32;
            for t in 0..8 {
                acc += f[t] as i32 * src.get(((p >> 4) + t as i32 - 3).clamp(0, r.last_x.max(0)) as usize).copied().unwrap_or(0) as i32;
            }
            if let Some(d) = tmp.get_mut(row * w + c) {
                *d = ((acc + 64) >> 7).max(0).min(max) as u16;
            }
        }
    }
    for rr in 0..h {
        let p = (y & 15) + step_y * rr as i32;
        let f = taps(filter, (p & 15) as usize);
        let base = (p >> 4) as usize;
        for c in 0..w {
            let mut acc = 0i32;
            for t in 0..8 {
                acc += f[t] as i32 * tmp.get((base + t) * w + c).copied().unwrap_or(0) as i32;
            }
            if let Some(d) = out.get_mut(rr * os + c) {
                *d = ((acc + 64) >> 7).max(0).min(max) as u16;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame_of(w: usize, h: usize, bit_depth: u8, sample: impl Fn(usize, usize) -> u16) -> Frame {
        let info = crate::frame::FrameInfo {
            width: w as u32,
            height: h as u32,
            ss_x: true,
            ss_y: true,
            bit_depth,
            color_space: 0,
            color_range: false,
            render_width: w as u32,
            render_height: h as u32,
            key: true,
            intra_only: false,
        };
        // Planes are allocated in whole superblocks.
        let (aw, ah) = (w.div_ceil(64) * 64, h.div_ceil(64) * 64);
        let mut full = vec![0u16; aw * ah];
        for y in 0..h {
            for x in 0..w {
                full[y * aw + x] = sample(x, y);
            }
        }
        let chroma = vec![0u16; (aw / 2) * (ah / 2)];
        Frame::from_planes(info, [&full, &chroma, &chroma])
    }

    /// The block inter prediction process of 8.5.2.4 written out literally (unscaled): the
    /// reference sample at (x, y) is clamped to the picture, the first pass rounds and clips.
    fn reference(f: &Frame, x: i32, y: i32, w: usize, h: usize, filter: u8, bit_depth: u8) -> Vec<u16> {
        let (lx, ly) = (f.info.width as i32 - 1, f.info.height as i32 - 1);
        let max = (1i32 << bit_depth) - 1;
        let px = |xx: i32, yy: i32| f.row(0, yy.clamp(0, ly) as usize)[xx.clamp(0, lx) as usize] as i32;
        let fh = taps(filter, (x & 15) as usize);
        let fv = taps(filter, (y & 15) as usize);
        let mut inter = vec![0i32; (h + 7) * w];
        for r in 0..h + 7 {
            for c in 0..w {
                let s: i32 = (0..8).map(|t| fh[t] as i32 * px((x >> 4) + c as i32 + t as i32 - 3, (y >> 4) + r as i32 - 3)).sum();
                inter[r * w + c] = ((s + 64) >> 7).clamp(0, max);
            }
        }
        let mut out = vec![0u16; w * h];
        for r in 0..h {
            for c in 0..w {
                let s: i32 = (0..8).map(|t| fv[t] as i32 * inter[(r + t) * w + c]).sum();
                out[r * w + c] = ((s + 64) >> 7).clamp(0, max) as u16;
            }
        }
        out
    }

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Every block size, filter, sub-sample phase and bit depth against the literal process, at
    /// random positions (inside, across band boundaries, beyond every edge), on random samples
    /// and on extreme ones (all 0, all max, max / 0 patterns that maximise the positive or
    /// negative taps' sums).
    #[test]
    fn unscaled_matches_literal_process() {
        let (w, h) = (150usize, 140usize);
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for bit_depth in [8u8, 10, 12] {
            let max = (1u16 << bit_depth) - 1;
            let seed = rng.next();
            let patterns: [Box<dyn Fn(usize, usize) -> u16>; 6] = [
                Box::new(move |x, y| (((x as u64 * 7919 + y as u64 * 104729) ^ seed).wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40) as u16 & max),
                Box::new(|_, _| 0),
                Box::new(move |_, _| max),
                Box::new(move |x, _| if x % 2 == 0 { max } else { 0 }),
                Box::new(move |_, y| if y % 4 < 2 { max } else { 0 }),
                Box::new(move |x, y| if (x / 3 + y / 5) % 2 == 0 { max } else { 0 }),
            ];
            for pat in patterns.iter() {
                let frame = frame_of(w, h, bit_depth, pat);
                let r = RefPlane { frame: &frame, plane: 0, last_x: w as i32 - 1, last_y: h as i32 - 1 };
                let mut tmp = vec![0u16; 64 * 71];
                let mut win = vec![0u16; 71 * 71];
                for &(bw, bh) in
                    &[(4usize, 4usize), (4, 8), (8, 4), (8, 8), (16, 8), (8, 16), (16, 16), (32, 16), (16, 32), (32, 32), (64, 32), (32, 64), (64, 64)]
                {
                    for _ in 0..12 {
                        let filter = rng.below(4) as u8;
                        let x = rng.below(16 * (w as u64 + 40)) as i32 - 16 * 20;
                        // Near band boundaries (rows 64 and 128) half of the time.
                        let y = if rng.below(2) == 0 {
                            rng.below(16 * (h as u64 + 40)) as i32 - 16 * 20
                        } else {
                            16 * (64 * (1 + rng.below(2) as i32) - bh as i32 / 2) + rng.below(16 * 9) as i32 - 16 * 4
                        };
                        let want = reference(&frame, x, y, bw, bh, filter, bit_depth);
                        // Write into a larger destination (stride 80) to check the stride.
                        let mut got = vec![0xFFFFu16; 80 * bh];
                        predict(&r, x, y, 16, 16, bw, bh, filter, bit_depth, &mut got, 80, &mut tmp, &mut win);
                        for row in 0..bh {
                            assert_eq!(
                                &got[row * 80..row * 80 + bw],
                                &want[row * bw..row * bw + bw],
                                "{bit_depth}-bit filter {filter} ({x},{y}) {bw}x{bh} row {row}"
                            );
                            assert!(got[row * 80 + bw..row * 80 + 80].iter().all(|&v| v == 0xFFFF), "wrote past the block");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn scaled_path_with_unit_step_matches_unscaled() {
        let (w, h) = (40usize, 30usize);
        let frame = frame_of(w, h, 8, |x, y| (((y * w + x) * 37 + y * 11) % 256) as u16);
        let r = RefPlane { frame: &frame, plane: 0, last_x: w as i32 - 1, last_y: h as i32 - 1 };
        let mut tmp = vec![0u16; 80 * 80];
        for filter in 0..4u8 {
            for &(x, y) in &[(0, 0), (5, 3), (16 * 7 + 5, 16 * 9 + 11), (-40, -3), (16 * 36 + 1, 16 * 25 + 15), (16 * 10, 16 * 10 + 8)] {
                for &(bw, bh) in &[(4usize, 4usize), (8, 8), (16, 8)] {
                    let mut a = vec![0u16; bw * bh];
                    let mut b = vec![0u16; bw * bh];
                    predict_unscaled::<false>(&r, x, y, bw, bh, filter, 8, &mut a, bw, &mut tmp, &mut [0u16; 71 * 71]);
                    predict_scaled(&r, x, y, 16, 16, bw, bh, filter, 8, &mut b, bw, &mut tmp);
                    assert_eq!(a, b, "filter {filter} pos ({x},{y}) {bw}x{bh}");
                }
            }
        }
    }
}
