//! Pictures for the networks: RGBA8 frames resized (area-averaged when shrinking, bilinear when
//! growing) and normalised into the planar float layout each model expects.

/// An RGB picture, row-major, values 0..255.
#[derive(Clone, Debug, PartialEq)]
pub struct Rgb {
    pub w: usize,
    pub h: usize,
    pub px: Vec<[f32; 3]>,
}

/// `rgba` (`w`×`h`) resized to `nw`×`nh`. `None` for an empty or short buffer or a zero size.
pub fn resize(rgba: &[u8], w: usize, h: usize, nw: usize, nh: usize) -> Option<Rgb> {
    let need = w.checked_mul(h)?.checked_mul(4)?;
    if w == 0 || h == 0 || nw == 0 || nh == 0 || rgba.len() < need || nw.checked_mul(nh)? > 64 << 20 {
        return None;
    }
    let at = |x: usize, y: usize| -> [f32; 3] {
        let i = (y.min(h - 1) * w + x.min(w - 1)) * 4;
        match rgba.get(i..i + 3) {
            Some(p) => [f32::from(p[0]), f32::from(p[1]), f32::from(p[2])],
            None => [0.0; 3],
        }
    };
    let mut px = Vec::with_capacity(nw * nh);
    let (sx, sy) = (w as f64 / nw as f64, h as f64 / nh as f64);
    for y in 0..nh {
        for x in 0..nw {
            let v = if sx > 1.0 || sy > 1.0 {
                // the source area this pixel covers, averaged
                let (x0, x1) = ((x as f64 * sx) as usize, (((x + 1) as f64 * sx).ceil() as usize).clamp(1, w));
                let (y0, y1) = ((y as f64 * sy) as usize, (((y + 1) as f64 * sy).ceil() as usize).clamp(1, h));
                let (x0, y0) = (x0.min(x1 - 1), y0.min(y1 - 1));
                let mut acc = [0f32; 3];
                let mut n = 0f32;
                for yy in y0..y1 {
                    for xx in x0..x1 {
                        let p = at(xx, yy);
                        acc[0] += p[0];
                        acc[1] += p[1];
                        acc[2] += p[2];
                        n += 1.0;
                    }
                }
                let n = n.max(1.0);
                [acc[0] / n, acc[1] / n, acc[2] / n]
            } else {
                let fx = ((x as f64 + 0.5) * sx - 0.5).max(0.0);
                let fy = ((y as f64 + 0.5) * sy - 0.5).max(0.0);
                let (x0, y0) = (fx as usize, fy as usize);
                let (ax, ay) = ((fx - x0 as f64) as f32, (fy - y0 as f64) as f32);
                let (a, b, c, d) = (at(x0, y0), at(x0 + 1, y0), at(x0, y0 + 1), at(x0 + 1, y0 + 1));
                let mut o = [0f32; 3];
                for k in 0..3 {
                    o[k] = (a[k] * (1.0 - ax) + b[k] * ax) * (1.0 - ay) + (c[k] * (1.0 - ax) + d[k] * ax) * ay;
                }
                o
            };
            px.push(v);
        }
    }
    Some(Rgb { w: nw, h: nh, px })
}

/// The middle square-ish crop of `rgba` with aspect `aw:ah` (CLIP's centre crop), as
/// `(x, y, w, h)`.
pub fn center_crop(w: usize, h: usize, aw: usize, ah: usize) -> (usize, usize, usize, usize) {
    if w == 0 || h == 0 || aw == 0 || ah == 0 {
        return (0, 0, w, h);
    }
    // the largest aw:ah box inside w×h
    let (cw, ch) = if w * ah > h * aw { (h * aw / ah, h) } else { (w, w * ah / aw) };
    let (cw, ch) = (cw.clamp(1, w), ch.clamp(1, h));
    ((w - cw) / 2, (h - ch) / 2, cw, ch)
}

/// The sub-picture `(x, y, cw, ch)` of `rgba` as its own RGBA buffer.
pub fn crop(rgba: &[u8], w: usize, h: usize, (x, y, cw, ch): (usize, usize, usize, usize)) -> Option<Vec<u8>> {
    if x.checked_add(cw)? > w || y.checked_add(ch)? > h || rgba.len() < w.checked_mul(h)?.checked_mul(4)? {
        return None;
    }
    let mut out = Vec::with_capacity(cw * ch * 4);
    for yy in y..y + ch {
        let row = (yy * w + x) * 4;
        out.extend_from_slice(rgba.get(row..row + cw * 4)?);
    }
    Some(out)
}

/// Planar `[3, h, w]` floats: `(v / 255 - mean) / std` per channel.
pub fn planar(img: &Rgb, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let n = img.w * img.h;
    let mut out = vec![0f32; 3 * n];
    for (i, p) in img.px.iter().enumerate().take(n) {
        for c in 0..3 {
            let s = if std[c].abs() > 1e-6 { std[c] } else { 1.0 };
            out[c * n + i] = (p[c] / 255.0 - mean[c]) / s;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: usize, h: usize) -> Vec<u8> {
        let mut v = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                v[i] = (x * 255 / w.max(2).saturating_sub(1).max(1)) as u8;
                v[i + 1] = (y * 255 / h.max(2).saturating_sub(1).max(1)) as u8;
                v[i + 2] = 100;
                v[i + 3] = 255;
            }
        }
        v
    }

    #[test]
    fn resizes_both_ways() {
        let src = gradient(64, 32);
        let small = resize(&src, 64, 32, 16, 8).unwrap();
        assert_eq!((small.w, small.h, small.px.len()), (16, 8, 128));
        assert!(small.px[0][0] < small.px[15][0], "left to right");
        assert!((small.px[0][2] - 100.0).abs() < 1e-3);
        let big = resize(&src, 64, 32, 128, 64).unwrap();
        assert_eq!(big.px.len(), 128 * 64);
        assert!(big.px.iter().all(|p| p.iter().all(|v| (0.0..=255.0).contains(v))));
        assert!(resize(&src, 64, 32, 0, 8).is_none());
        assert!(resize(&src[..10], 64, 32, 8, 8).is_none());
        assert!(resize(&src, usize::MAX, 2, 8, 8).is_none());
        // a single pixel spreads
        let one = resize(&[10, 20, 30, 255], 1, 1, 3, 3).unwrap();
        assert!(one.px.iter().all(|p| (p[0] - 10.0).abs() < 1e-4 && (p[1] - 20.0).abs() < 1e-4 && (p[2] - 30.0).abs() < 1e-4));
    }

    #[test]
    fn crops_and_normalises() {
        assert_eq!(center_crop(1920, 1080, 1, 1), (420, 0, 1080, 1080));
        assert_eq!(center_crop(1080, 1920, 1, 1), (0, 420, 1080, 1080));
        assert_eq!(center_crop(0, 5, 1, 1), (0, 0, 0, 5));
        let src = gradient(8, 4);
        let c = crop(&src, 8, 4, (2, 0, 4, 4)).unwrap();
        assert_eq!(c.len(), 4 * 4 * 4);
        assert_eq!(&c[..4], &src[8..12]);
        assert!(crop(&src, 8, 4, (6, 0, 4, 4)).is_none());
        let p = planar(&Rgb { w: 1, h: 1, px: vec![[255.0, 0.0, 127.5]] }, [0.5; 3], [0.5; 3]);
        assert_eq!(p, vec![1.0, -1.0, 0.0]);
    }
}
