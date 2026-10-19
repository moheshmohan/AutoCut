//! Turning the detectors' raw outputs into detections: YuNet's anchor-free decoding, LPD-YuNet's
//! prior boxes, and non-maximum suppression. Plain Rust on slices (no candle), so it is tested
//! with synthetic outputs and builds everywhere.

use crate::{Detection, Kind};

/// YuNet's three output levels and their strides.
pub const YUNET_STRIDES: [usize; 3] = [8, 16, 32];

/// One YuNet output level: a `rows` × `cols` grid of cells `stride` input pixels apart, each
/// with a class score, an objectness score (both after the sigmoid), a box (centre offset in cells,
/// log size in strides) and five landmarks (offsets in cells).
#[derive(Clone, Copy, Debug)]
pub struct YunetLevel<'a> {
    pub stride: usize,
    pub rows: usize,
    pub cols: usize,
    pub cls: &'a [f32],
    pub obj: &'a [f32],
    pub bbox: &'a [f32],
    pub kps: &'a [f32],
}

fn finite_or(v: f32, d: f32) -> f32 {
    if v.is_finite() { v } else { d }
}

/// Faces from YuNet's outputs, in input pixels, before suppression: the score is the geometric
/// mean of the class and objectness scores (each clamped to 0..1); cells below `threshold` are
/// skipped. A level whose slices are shorter than its grid is decoded as far as they go.
pub fn decode_yunet(levels: &[YunetLevel<'_>], threshold: f32) -> Vec<Detection> {
    let mut out = Vec::new();
    for l in levels {
        let s = l.stride as f32;
        let n = l.rows.saturating_mul(l.cols);
        for idx in 0..n {
            let (Some(&c), Some(&o)) = (l.cls.get(idx), l.obj.get(idx)) else { break };
            let score = (finite_or(c, 0.0).clamp(0.0, 1.0) * finite_or(o, 0.0).clamp(0.0, 1.0)).sqrt();
            if score.is_nan() || score < threshold {
                continue;
            }
            let (Some(b), Some(k)) = (l.bbox.get(idx * 4..idx * 4 + 4), l.kps.get(idx * 10..idx * 10 + 10)) else { break };
            let (r, col) = ((idx / l.cols.max(1)) as f32, (idx % l.cols.max(1)) as f32);
            let cx = (col + b[0]) * s;
            let cy = (r + b[1]) * s;
            // a log size far outside any face is clamped rather than overflowing
            let w = b[2].clamp(-20.0, 10.0).exp() * s;
            let h = b[3].clamp(-20.0, 10.0).exp() * s;
            if !(cx.is_finite() && cy.is_finite() && w.is_finite() && h.is_finite()) {
                continue;
            }
            let points = (0..5).map(|p| [(k[2 * p] + col) * s, (k[2 * p + 1] + r) * s]).collect();
            out.push(Detection { kind: Kind::Face, score, bbox: [cx - w / 2.0, cy - h / 2.0, w, h], points });
        }
    }
    out
}

/// LPD-YuNet's prior boxes for a `w` × `h` input: per level (strides 8, 16, 32, 64), per cell
/// row-major, per minimum size: `[cx, cy, sx, sy]` as shares of the input.
pub fn lpd_priors(w: usize, h: usize) -> Vec<[f32; 4]> {
    const MIN_SIZES: [&[f32]; 4] = [&[10.0, 16.0, 24.0], &[32.0, 48.0], &[64.0, 96.0], &[128.0, 192.0, 256.0]];
    const STEPS: [f32; 4] = [8.0, 16.0, 32.0, 64.0];
    let (wf, hf) = (w.max(1) as f32, h.max(1) as f32);
    // the network's feature sizes: a stride-2 convolution (ceil) then max pools (floor)
    let mut fh = h.div_ceil(2) / 2;
    let mut fw = w.div_ceil(2) / 2;
    let mut out = Vec::new();
    for (k, sizes) in MIN_SIZES.iter().enumerate() {
        fh /= 2;
        fw /= 2;
        for i in 0..fh {
            for j in 0..fw {
                for m in sizes.iter() {
                    let step = STEPS[k];
                    out.push([(j as f32 + 0.5) * step / wf, (i as f32 + 0.5) * step / hf, m / wf, m / hf]);
                }
            }
        }
    }
    out
}

/// Values per prior in LPD-YuNet's output: 14 location values (of which four corner points are
/// used), two class logits and an IoU estimate.
pub const LPD_STRIDE: usize = 17;

/// Plates from LPD-YuNet's raw head outputs (`raw`: [`LPD_STRIDE`] values per prior, in prior
/// order), in input pixels, before suppression. The score is the geometric mean of the plate
/// probability (softmax of the class logits) and the IoU estimate clamped to 0..1.
pub fn decode_lpd(raw: &[f32], priors: &[[f32; 4]], w: usize, h: usize, threshold: f32) -> Vec<Detection> {
    const VARIANCE: f32 = 0.1;
    let (wf, hf) = (w as f32, h as f32);
    let mut out = Vec::new();
    for (p, v) in priors.iter().zip(raw.as_chunks::<LPD_STRIDE>().0) {
        let (a, b) = (finite_or(v[14], 0.0), finite_or(v[15], 0.0));
        let m = a.max(b);
        let plate = (b - m).exp() / ((a - m).exp() + (b - m).exp());
        let iou = finite_or(v[16], 0.0).clamp(0.0, 1.0);
        let score = (plate * iou).sqrt();
        if score.is_nan() || score < threshold {
            continue;
        }
        // corner points: loc[4..6], [6..8], [10..12], [12..14]
        let pt = |i: usize| [(p[0] + v[i] * VARIANCE * p[2]) * wf, (p[1] + v[i + 1] * VARIANCE * p[3]) * hf];
        let points = vec![pt(4), pt(6), pt(10), pt(12)];
        if points.iter().any(|q| !(q[0].is_finite() && q[1].is_finite())) {
            continue;
        }
        let (x0, y0, x1, y1) =
            points.iter().fold((f32::MAX, f32::MAX, f32::MIN, f32::MIN), |(a, b, c, d), q| (a.min(q[0]), b.min(q[1]), c.max(q[0]), d.max(q[1])));
        out.push(Detection { kind: Kind::Plate, score, bbox: [x0, y0, (x1 - x0).max(0.0), (y1 - y0).max(0.0)], points });
    }
    out
}

/// Intersection over union of two `[x, y, w, h]` boxes (0 for empty boxes).
pub fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let ix = ((a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0])).max(0.0);
    let iy = ((a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let union = a[2].max(0.0) * a[3].max(0.0) + b[2].max(0.0) * b[3].max(0.0) - inter;
    if union > 0.0 && inter.is_finite() { inter / union } else { 0.0 }
}

/// Greedy non-maximum suppression: highest score first, dropping boxes that overlap a kept one
/// by more than `threshold` IoU; at most `top_k` kept.
pub fn nms(mut dets: Vec<Detection>, threshold: f32, top_k: usize) -> Vec<Detection> {
    dets.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = Vec::new();
    for d in dets {
        if kept.len() >= top_k {
            break;
        }
        if kept.iter().all(|k| iou(&k.bbox, &d.bbox) <= threshold) {
            kept.push(d);
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yunet_decodes_a_cell() {
        // a 2 × 2 grid at stride 8: cell (row 1, col 0) holds a face
        let cls = [0.0, 0.0, 0.81, 0.0];
        let obj = [0.0, 0.0, 1.0, 0.0];
        let mut bbox = [0.0f32; 16];
        bbox[8..12].copy_from_slice(&[0.5, 0.5, 2f32.ln(), 3f32.ln()]);
        let kps = [0.25f32; 40];
        let l = YunetLevel { stride: 8, rows: 2, cols: 2, cls: &cls, obj: &obj, bbox: &bbox, kps: &kps };
        let d = decode_yunet(&[l], 0.5);
        assert_eq!(d.len(), 1);
        assert!((d[0].score - 0.9).abs() < 1e-6);
        // centre (0.5 cells, 1.5 cells) = (4, 12) px; 16 × 24 px
        assert_eq!(d[0].bbox, [4.0 - 8.0, 12.0 - 12.0, 16.0, 24.0]);
        assert_eq!(d[0].points[0], [2.0, 10.0]);
        // short slices and non-finite values never panic
        let l = YunetLevel { stride: 8, rows: 100, cols: 100, cls: &[f32::NAN, 1.0], obj: &[1.0, 1.0], bbox: &[0.0, 0.0, 1e30, 0.0], kps: &[] };
        assert!(decode_yunet(&[l], 0.5).is_empty());
    }

    #[test]
    fn lpd_priors_follow_the_feature_sizes() {
        // 320 × 240: features 40 × 30, 20 × 15, 10 × 7, 5 × 3 cells
        let p = lpd_priors(320, 240);
        assert_eq!(p.len(), 40 * 30 * 3 + 20 * 15 * 2 + 10 * 7 * 2 + 5 * 3 * 3);
        assert_eq!(p.len(), 4385, "the ONNX file's own output size at 320 × 240");
        assert_eq!(p[0], [4.0 / 320.0, 4.0 / 240.0, 10.0 / 320.0, 10.0 / 240.0]);
        assert!(lpd_priors(0, 0).is_empty());
    }

    #[test]
    fn lpd_decodes_corners_and_scores() {
        let priors = vec![[0.5, 0.5, 0.1, 0.1]];
        let mut raw = vec![0.0f32; LPD_STRIDE];
        // corners at ±1 prior unit × variance
        raw[4..8].copy_from_slice(&[-10.0, -5.0, 10.0, -5.0]);
        raw[10..14].copy_from_slice(&[10.0, 5.0, -10.0, 5.0]);
        raw[14] = 0.0;
        raw[15] = 10.0;
        raw[16] = 0.81;
        let d = decode_lpd(&raw, &priors, 100, 100, 0.5);
        assert_eq!(d.len(), 1);
        assert!((d[0].score - 0.9).abs() < 1e-3);
        assert_eq!(d[0].points[0], [40.0, 45.0]);
        for (a, b) in d[0].bbox.iter().zip([40.0, 45.0, 20.0, 10.0]) {
            assert!((a - b).abs() < 1e-4, "{:?}", d[0].bbox);
        }
        raw[16] = 0.0;
        assert!(decode_lpd(&raw, &priors, 100, 100, 0.5).is_empty());
    }

    #[test]
    fn nms_keeps_the_best_of_overlapping_boxes() {
        let d = |s: f32, x: f32| Detection { kind: Kind::Face, score: s, bbox: [x, 0.0, 10.0, 10.0], points: Vec::new() };
        let kept = nms(vec![d(0.7, 1.0), d(0.9, 0.0), d(0.8, 50.0)], 0.3, 10);
        assert_eq!(kept.iter().map(|k| k.score).collect::<Vec<_>>(), vec![0.9, 0.8]);
        assert_eq!(nms(vec![d(0.7, 1.0), d(0.9, 100.0)], 0.3, 1).len(), 1);
        assert_eq!(iou(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0]), 0.0);
    }
}
