//! A person cut out of a picture, as the editor uses it: the points that tell the segmenter who
//! (from their pose and box, [`prompts`]), and the cut-out's outline as a closed polygon of a
//! fixed number of points ([`outline`]) that a keyframed mask path can follow from frame to frame.

use crate::head::Person;

/// A cut-out: coverage 0..255 a pixel, `w` × `h`.
#[derive(Clone, Debug, PartialEq)]
pub struct Matte {
    pub w: usize,
    pub h: usize,
    pub alpha: Vec<u8>,
    /// How sure the segmenter was of it (its predicted overlap, 0..1).
    pub score: f32,
}

impl Matte {
    pub fn at(&self, x: usize, y: usize) -> u8 {
        if x >= self.w || y >= self.h {
            return 0;
        }
        self.alpha.get(y * self.w + x).copied().unwrap_or(0)
    }

    /// The share of the picture it covers.
    pub fn area(&self) -> f32 {
        let n = self.alpha.iter().filter(|a| **a >= 128).count();
        n as f32 / (self.w * self.h).max(1) as f32
    }
}

/// A keypoint counts as a prompt with at least this score.
const KEYPOINT_SCORE: f32 = 0.35;
/// The body's keypoints used as prompts (COCO order: shoulders, elbows, hips, knees; the nose).
const BODY: [usize; 9] = [0, 5, 6, 7, 8, 11, 12, 13, 14];

/// Points on `person` (inside) and round them (outside), in picture pixels: the body's surest
/// keypoints, or the box's middle without a pose, and the middles of the box's sides pushed out
/// by a tenth of its size.
pub fn prompts(person: &Person, w: usize, h: usize) -> (Vec<(f32, f32)>, Vec<(f32, f32)>) {
    let [x, y, bw, bh] = person.bbox;
    let inside = |p: &(f32, f32)| p.0.is_finite() && p.1.is_finite() && p.0 >= 0.0 && p.1 >= 0.0 && p.0 < w as f32 && p.1 < h as f32;
    let mut pos: Vec<(f32, f32)> =
        BODY.iter().filter_map(|&i| person.keypoints.get(i)).filter(|k| k.score >= KEYPOINT_SCORE).map(|k| (k.x, k.y)).filter(inside).collect();
    if pos.len() < 2 {
        pos = vec![(x + bw * 0.5, y + bh * 0.35), (x + bw * 0.5, y + bh * 0.6)];
    }
    let (dx, dy) = (bw * 0.1 + 2.0, bh * 0.1 + 2.0);
    let neg: Vec<(f32, f32)> = [(x - dx, y + bh * 0.5), (x + bw + dx, y + bh * 0.5), (x + bw * 0.5, y - dy)].into_iter().filter(inside).collect();
    (pos.into_iter().filter(inside).collect(), neg)
}

/// The outline of the largest piece of `m` (coverage ≥ 128), traced round its edge and spaced
/// into `n` points at equal steps round it, clockwise from its topmost point, in the matte's
/// pixels (corners of pixels). `None` when nothing is covered.
pub fn outline(m: &Matte, n: usize) -> Option<Vec<(f32, f32)>> {
    let n = n.clamp(8, 512);
    let (w, h) = (m.w, m.h);
    if w == 0 || h == 0 || m.alpha.len() < w.checked_mul(h)? {
        return None;
    }
    let on = |x: i64, y: i64| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && m.at(x as usize, y as usize) >= 128;
    // the largest piece: label the covered pixels (4-connected), keep the biggest
    let mut label = vec![0u32; w * h];
    let (mut best, mut best_n, mut next) = (0u32, 0usize, 0u32);
    let mut stack = Vec::new();
    for start in 0..w * h {
        if label[start] != 0 || m.alpha[start] < 128 {
            continue;
        }
        next += 1;
        let mut count = 0usize;
        label[start] = next;
        stack.push(start);
        while let Some(i) = stack.pop() {
            count += 1;
            let (x, y) = ((i % w) as i64, (i / w) as i64);
            for (nx, ny) in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] {
                if on(nx, ny) {
                    let j = ny as usize * w + nx as usize;
                    if label[j] == 0 {
                        label[j] = next;
                        stack.push(j);
                    }
                }
            }
        }
        if count > best_n {
            (best, best_n) = (next, count);
        }
    }
    if best_n == 0 {
        return None;
    }
    let mine = |x: i64, y: i64| x >= 0 && y >= 0 && (x as usize) < w && (y as usize) < h && label[y as usize * w + x as usize] == best;
    // the edge, walked along pixel corners with the piece on the right (clockwise on screen):
    // start at the top-left corner of its topmost, leftmost pixel, heading right
    let first = label.iter().position(|l| *l == best)?;
    let (sx, sy) = ((first % w) as i64, (first / w) as i64);
    let (mut cx, mut cy, mut dir) = (sx, sy, 0u8); // 0 right, 1 down, 2 left, 3 up
    let mut corners: Vec<(f32, f32)> = Vec::new();
    let limit = 4 * (w + h) * 4 + 4 * best_n + 16;
    for _ in 0..limit {
        corners.push((cx as f32, cy as f32));
        // the pixels ahead-left and ahead-right of the corner, as seen heading `dir`
        let (l, r) = match dir {
            0 => (mine(cx, cy - 1), mine(cx, cy)),
            1 => (mine(cx, cy), mine(cx - 1, cy)),
            2 => (mine(cx - 1, cy), mine(cx - 1, cy - 1)),
            _ => (mine(cx - 1, cy - 1), mine(cx, cy - 1)),
        };
        dir = match (l, r) {
            (true, _) => (dir + 3) % 4,
            (false, true) => dir,
            (false, false) => (dir + 1) % 4,
        };
        // back at the start, about to set off as at first: closed
        if corners.len() > 1 && (cx, cy) == (sx, sy) && dir == 0 {
            corners.pop();
            break;
        }
        match dir {
            0 => cx += 1,
            1 => cy += 1,
            2 => cx -= 1,
            _ => cy -= 1,
        }
    }
    if corners.len() < 4 {
        return None;
    }
    resample(&corners, n)
}

/// `n` points at equal steps round the closed polygon `p`, from its first point.
fn resample(p: &[(f32, f32)], n: usize) -> Option<Vec<(f32, f32)>> {
    let seg = |i: usize| -> f32 {
        let (a, b) = (p[i], p[(i + 1) % p.len()]);
        ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt()
    };
    let total: f32 = (0..p.len()).map(seg).sum();
    if !(total.is_finite() && total > 0.0) {
        return None;
    }
    let step = total / n as f32;
    let mut out = Vec::with_capacity(n);
    let (mut i, mut walked) = (0usize, 0.0f32);
    for k in 0..n {
        let want = step * k as f32;
        while i < p.len() && walked + seg(i) < want {
            walked += seg(i);
            i += 1;
        }
        let i0 = i.min(p.len() - 1);
        let (a, b) = (p[i0], p[(i0 + 1) % p.len()]);
        let l = seg(i0).max(1e-6);
        let t = ((want - walked) / l).clamp(0.0, 1.0);
        out.push((a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t));
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::head::Keypoint;

    fn disc(w: usize, h: usize, cx: f32, cy: f32, r: f32) -> Matte {
        let alpha = (0..w * h).map(|i| if (((i % w) as f32 + 0.5 - cx).powi(2) + ((i / w) as f32 + 0.5 - cy).powi(2)).sqrt() <= r { 255 } else { 0 }).collect();
        Matte { w, h, alpha, score: 1.0 }
    }

    #[test]
    fn a_disc_is_outlined_round_its_edge() {
        let m = disc(64, 48, 30.0, 20.0, 12.0);
        let o = outline(&m, 64).unwrap();
        assert_eq!(o.len(), 64);
        for (x, y) in &o {
            let r = ((x - 30.0).powi(2) + (y - 20.0).powi(2)).sqrt();
            assert!((r - 12.0).abs() < 1.6, "({x}, {y}) is {r} from the centre");
        }
        // from the top, clockwise on screen (x grows first)
        assert!(o[0].1 <= 8.5 + 1.0 && o[3].0 > o[0].0, "{:?}", &o[..4]);
        // the signed area (screen y down) is positive for clockwise
        let area: f32 = (0..o.len()).map(|i| o[i].0 * o[(i + 1) % o.len()].1 - o[(i + 1) % o.len()].0 * o[i].1).sum::<f32>() / 2.0;
        assert!((area - std::f32::consts::PI * 144.0).abs() < 40.0, "{area}");
    }

    #[test]
    fn the_largest_piece_is_kept_and_nothing_is_none() {
        let mut m = disc(64, 48, 40.0, 24.0, 14.0);
        for (i, a) in disc(64, 48, 8.0, 8.0, 4.0).alpha.iter().enumerate() {
            m.alpha[i] = m.alpha[i].max(*a);
        }
        let o = outline(&m, 32).unwrap();
        assert!(o.iter().all(|(x, _)| *x > 20.0), "{o:?}");
        assert!(outline(&Matte { w: 8, h: 8, alpha: vec![0; 64], score: 0.0 }, 16).is_none());
        // hostile sizes
        assert!(outline(&Matte { w: 0, h: 0, alpha: vec![], score: 0.0 }, 16).is_none());
        assert!(outline(&Matte { w: 100, h: 100, alpha: vec![255; 10], score: 0.0 }, 16).is_none());
        assert_eq!(outline(&Matte { w: 1, h: 1, alpha: vec![255], score: 0.0 }, 0).map(|o| o.len()), Some(8));
    }

    #[test]
    fn a_pose_gives_points_on_the_body_and_round_it() {
        let mut kp = vec![Keypoint { x: 0.0, y: 0.0, score: 0.0 }; 17];
        for (i, (x, y)) in [(0, (50.0, 20.0)), (5, (40.0, 40.0)), (6, (60.0, 40.0)), (11, (45.0, 80.0)), (12, (55.0, 80.0))] {
            kp[i] = Keypoint { x, y, score: 0.9 };
        }
        let p = Person { bbox: [30.0, 10.0, 40.0, 100.0], score: 0.9, keypoints: kp };
        let (pos, neg) = prompts(&p, 200, 200);
        assert_eq!(pos.len(), 5);
        assert!(neg.iter().all(|(x, y)| !(*x > 30.0 && *x < 70.0 && *y > 10.0 && *y < 110.0)), "{neg:?}");
        // no pose: the box's middle
        let (pos, _) = prompts(&Person { keypoints: vec![], ..p.clone() }, 200, 200);
        assert_eq!(pos.len(), 2);
        // at the picture's edge, outside points are left out; NaN never gets through
        let edge = Person { bbox: [0.0, 0.0, f32::NAN, 50.0], score: 1.0, keypoints: vec![] };
        let (pos, neg) = prompts(&edge, 100, 100);
        assert!(pos.iter().chain(&neg).all(|(x, y)| x.is_finite() && y.is_finite()));
    }
}
