//! The compositor's working image: premultiplied linear-light RGBA f32, row-major.

use filmcraft_geom::Affine;
use rayon::prelude::*;

/// The two pixels of eight floats.
#[inline]
fn split4(p: [f32; 8]) -> ([f32; 4], [f32; 4]) {
    ([p[0], p[1], p[2], p[3]], [p[4], p[5], p[6], p[7]])
}

#[derive(Clone, Debug, PartialEq)]
pub struct Image {
    pub w: usize,
    pub h: usize,
    pub px: Vec<f32>,
}

impl Image {
    pub const MAX_PIXELS: usize = 268_435_456;

    pub fn new(w: usize, h: usize) -> Self {
        let len = w.checked_mul(h).filter(|&px| px <= Self::MAX_PIXELS).and_then(|px| px.checked_mul(4)).unwrap_or(0);
        let (w, h, px) = if len == 0 && (w > 0 || h > 0) {
            (0, 0, Vec::new())
        } else {
            let mut px = Vec::new();
            if px.try_reserve_exact(len).is_ok() {
                px.resize(len, 0.0);
                (w, h, px)
            } else {
                (0, 0, Vec::new())
            }
        };
        Self { w, h, px }
    }
    pub fn filled(w: usize, h: usize, c: [f32; 4]) -> Self {
        let len = w.checked_mul(h).filter(|&px| px <= Self::MAX_PIXELS).and_then(|px| px.checked_mul(4)).unwrap_or(0);
        let (w, h, px) = if len == 0 && (w > 0 || h > 0) {
            (0, 0, Vec::new())
        } else {
            let mut px = Vec::new();
            if px.try_reserve_exact(len).is_ok() {
                for _ in 0..w * h {
                    px.extend_from_slice(&c);
                }
                (w, h, px)
            } else {
                (0, 0, Vec::new())
            }
        };
        Self { w, h, px }
    }
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> [f32; 4] {
        let i = (y * self.w + x) * 4;
        [self.px[i], self.px[i + 1], self.px[i + 2], self.px[i + 3]]
    }
    /// Clamp-to-edge fetch.
    #[inline]
    pub fn get_clamped(&self, x: isize, y: isize) -> [f32; 4] {
        let x = x.clamp(0, self.w as isize - 1) as usize;
        let y = y.clamp(0, self.h as isize - 1) as usize;
        self.get(x, y)
    }
    /// Transparent-border fetch.
    #[inline]
    pub fn get_or_clear(&self, x: isize, y: isize) -> [f32; 4] {
        if x < 0 || y < 0 || x >= self.w as isize || y >= self.h as isize { [0.0; 4] } else { self.get(x as usize, y as usize) }
    }
    /// Bilinear sample at continuous pixel coords (pixel centres at +0.5), transparent outside.
    #[inline]
    pub fn sample_bilinear(&self, x: f32, y: f32) -> [f32; 4] {
        let fx = x - 0.5;
        let fy = y - 0.5;
        let x0 = fx.floor();
        let y0 = fy.floor();
        let tx = fx - x0;
        let ty = fy - y0;
        let (x0, y0) = (x0 as isize, y0 as isize);
        let a = self.get_or_clear(x0, y0);
        let b = self.get_or_clear(x0 + 1, y0);
        let c = self.get_or_clear(x0, y0 + 1);
        let d = self.get_or_clear(x0 + 1, y0 + 1);
        let mut o = [0.0; 4];
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bot = c[k] + (d[k] - c[k]) * tx;
            o[k] = top + (bot - top) * ty;
        }
        o
    }
    #[inline]
    pub fn sample_bilinear_clamped(&self, x: f32, y: f32) -> [f32; 4] {
        let fx = (x - 0.5).clamp(0.0, self.w as f32 - 1.0);
        let fy = (y - 0.5).clamp(0.0, self.h as f32 - 1.0);
        let x0 = fx.floor() as usize;
        let y0 = fy.floor() as usize;
        let x1 = (x0 + 1).min(self.w - 1);
        let y1 = (y0 + 1).min(self.h - 1);
        let tx = fx - x0 as f32;
        let ty = fy - y0 as f32;
        let (a, b, c, d) = (self.get(x0, y0), self.get(x1, y0), self.get(x0, y1), self.get(x1, y1));
        let mut o = [0.0; 4];
        for k in 0..4 {
            let top = a[k] + (b[k] - a[k]) * tx;
            let bot = c[k] + (d[k] - c[k]) * tx;
            o[k] = top + (bot - top) * ty;
        }
        o
    }

    /// Resample `self` into a `w`×`h` image through `m` (maps source pixels → destination pixels).
    /// Uses bilinear filtering, with a box pre-filter (mip level) when minifying by more than 2×.
    pub fn transformed(&self, w: usize, h: usize, m: &Affine) -> Image {
        if self.w == 0 || self.h == 0 || w == 0 || h == 0 {
            return Image::new(w, h);
        }
        let Some(inv) = m.inverse() else { return Image::new(w, h) };
        // minification factor
        let sx = (m.a * m.a + m.b * m.b).sqrt();
        let sy = (m.c * m.c + m.d * m.d).sqrt();
        let minify = 1.0 / sx.min(sy).max(1e-6);
        if minify >= 2.0 && self.w >= 4 && self.h >= 4 {
            let half = self.downsample2();
            let m2 = m.then_apply(&Affine::scale(2.0, 2.0));
            return half.transformed(w, h, &m2);
        }
        // past `MAX_PIXELS` (or empty) `Image::new` answers the empty image
        let Some(len) = w.checked_mul(h).filter(|&n| w > 0 && h > 0 && n <= Self::MAX_PIXELS).and_then(|n| n.checked_mul(4)) else { return Image::new(w, h) };
        // Every sample is written below (zero where nothing lands), so a recycled buffer serves
        // without the zero-fill: at 4K a placed clip's layer is 133 MB to clear per frame.
        let mut out = Image { w, h, px: filmcraft_frame::pool::take_f32_overwritten(len) };
        // Destination bounds of the source rect, to skip empty rows/cols.
        let b = m.bounds(&filmcraft_geom::Rect::new(0.0, 0.0, self.w as f64, self.h as f64));
        let y0 = (b.y.floor().max(0.0) as usize).min(h);
        let y1 = (b.bottom().ceil().max(0.0) as usize).min(h);
        let x0 = (b.x.floor().max(0.0) as usize).min(w);
        let x1 = (b.right().ceil().max(0.0) as usize).min(w).max(x0);
        let axis_aligned = m.b == 0.0 && m.c == 0.0;
        // Without rotation every row samples the same source columns with the same weights.
        let cols = if axis_aligned { self.column_taps(x0..x1, &inv) } else { Vec::new() };
        out.px.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
            if y < y0 || y >= y1 {
                row.fill(0.0);
                return;
            }
            let py = y as f64 + 0.5;
            if axis_aligned {
                self.scaled_row(row, x0, &cols, py, &inv);
                return;
            }
            row.fill(0.0);
            for x in x0..x1 {
                let px = x as f64 + 0.5;
                let (u, v) =
                    if axis_aligned { (inv.a * px + inv.e, inv.d * py + inv.f) } else { (inv.a * px + inv.c * py + inv.e, inv.b * px + inv.d * py + inv.f) };
                if u < -1.0 || v < -1.0 || u > self.w as f64 + 1.0 || v > self.h as f64 + 1.0 {
                    continue;
                }
                let s = self.sample_bilinear(u as f32, v as f32);
                row[x * 4..x * 4 + 4].copy_from_slice(&s);
            }
        });
        out
    }

    /// The source column and horizontal weight each destination column `xs` of an axis-aligned
    /// [`Self::transformed`] samples (`None`: nothing lands there), worked out as
    /// [`Self::sample_bilinear`] does.
    fn column_taps(&self, xs: std::ops::Range<usize>, inv: &Affine) -> Vec<Option<(isize, f32)>> {
        xs.map(|x| {
            let px = x as f64 + 0.5;
            let u = inv.a * px + inv.e;
            if u < -1.0 || u > self.w as f64 + 1.0 {
                return None;
            }
            let fx = u as f32 - 0.5;
            let x0 = fx.floor();
            Some((x0 as isize, fx - x0))
        })
        .collect()
    }

    /// One row of [`Self::transformed`] for a transform without rotation or shear (Motion's scale and
    /// position, what every placed clip has): the source rows and the vertical weight are the same
    /// for the whole row, so they are found once instead of per pixel, and the columns' taps
    /// (`cols`, from destination column `x0`) once for the whole image. The arithmetic is
    /// [`Self::sample_bilinear`]'s, so the pixels are the same to the bit. Every sample of `row` is
    /// written (zero where nothing lands).
    fn scaled_row(&self, row: &mut [f32], x0: usize, cols: &[Option<(isize, f32)>], py: f64, inv: &Affine) {
        let v = inv.d * py + inv.f;
        if v < -1.0 || v > self.h as f64 + 1.0 {
            row.fill(0.0);
            return;
        }
        let fy = v as f32 - 0.5;
        let y0 = fy.floor();
        let ty = fy - y0;
        let y0 = y0 as isize;
        // a source row, or none (transparent) above or below the picture
        let src_row = |y: isize| (y >= 0 && y < self.h as isize).then(|| self.px.get(y as usize * self.w * 4..(y as usize + 1) * self.w * 4)).flatten();
        let (top, bottom) = (src_row(y0), src_row(y0 + 1));
        let tap = |r: Option<&[f32]>, x: isize| -> [f32; 4] {
            match r {
                Some(r) if x >= 0 && (x as usize) < self.w => r.get(x as usize * 4..x as usize * 4 + 4).map_or([0.0; 4], |p| [p[0], p[1], p[2], p[3]]),
                _ => [0.0; 4],
            }
        };
        // two neighbouring source pixels at once, when both are inside the row
        let pair = |r: Option<&[f32]>, x: isize| -> Option<[f32; 8]> {
            let i = usize::try_from(x).ok()?.checked_mul(4)?;
            r?.get(i..i.checked_add(8)?).and_then(|p| <[f32; 8]>::try_from(p).ok())
        };
        let x1 = x0.saturating_add(cols.len()).min(row.len() / 4);
        let (left, rest) = row.split_at_mut((x0 * 4).min(row.len()));
        left.fill(0.0);
        let (mid, right) = rest.split_at_mut((x1.saturating_sub(x0) * 4).min(rest.len()));
        right.fill(0.0);
        for (o, col) in mid.as_chunks_mut::<4>().0.iter_mut().zip(cols) {
            let Some((sx, tx)) = *col else {
                *o = [0.0; 4];
                continue;
            };
            let ((a, b), (c, d)) = match (pair(top, sx), pair(bottom, sx)) {
                (Some(t), Some(u)) => (split4(t), split4(u)),
                _ => ((tap(top, sx), tap(top, sx + 1)), (tap(bottom, sx), tap(bottom, sx + 1))),
            };
            for k in 0..4 {
                let t = a[k] + (b[k] - a[k]) * tx;
                let b2 = c[k] + (d[k] - c[k]) * tx;
                o[k] = t + (b2 - t) * ty;
            }
        }
    }

    /// Half-size box downsample.
    pub fn downsample2(&self) -> Image {
        if self.w == 0 || self.h == 0 {
            return Image::new(0, 0);
        }
        let (w, h) = ((self.w / 2).max(1), (self.h / 2).max(1));
        let mut out = Image::new(w, h);
        out.px.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let a = self.get_clamped(2 * x as isize, 2 * y as isize);
                let b = self.get_clamped(2 * x as isize + 1, 2 * y as isize);
                let c = self.get_clamped(2 * x as isize, 2 * y as isize + 1);
                let d = self.get_clamped(2 * x as isize + 1, 2 * y as isize + 1);
                for k in 0..4 {
                    row[x * 4 + k] = (a[k] + b[k] + c[k] + d[k]) * 0.25;
                }
            }
        });
        out
    }

    /// `self × (1 − w) + other × w` (same size; premultiplied, so this is a proper cross-fade).
    pub fn lerp(mut self, other: &Image, w: f32) -> Image {
        if other.w != self.w || other.h != self.h {
            return self;
        }
        for (a, b) in self.px.iter_mut().zip(&other.px) {
            *a += (*b - *a) * w;
        }
        self
    }

    pub fn scale_alpha(&mut self, a: f32) {
        if (a - 1.0).abs() < 1e-6 {
            return;
        }
        self.px.par_iter_mut().for_each(|v| *v *= a);
    }

    /// Straight-alpha colour of a pixel.
    #[inline]
    pub fn unpremul(p: [f32; 4]) -> [f32; 3] {
        if p[3] <= 1e-6 { [0.0; 3] } else { [p[0] / p[3], p[1] / p[3], p[2] / p[3]] }
    }

    /// Apply a per-pixel function on straight-alpha colour (alpha preserved).
    pub fn map_rgb(&mut self, f: impl Fn([f32; 3], usize, usize) -> [f32; 3] + Sync) {
        if self.w == 0 || self.h == 0 {
            return;
        }
        let w = self.w;
        self.px.par_chunks_mut(w * 4).enumerate().for_each(|(y, row)| {
            for x in 0..w {
                let p = &mut row[x * 4..x * 4 + 4];
                let a = p[3];
                if a == 1.0 {
                    // opaque (most pictures): straight is premultiplied, no division needed
                    let c = f([p[0], p[1], p[2]], x, y);
                    p[..3].copy_from_slice(&c);
                    continue;
                }
                if a <= 1e-6 {
                    continue;
                }
                let c = f([p[0] / a, p[1] / a, p[2] / a], x, y);
                p[0] = c[0] * a;
                p[1] = c[1] * a;
                p[2] = c[2] * a;
            }
        });
    }

    /// [`Self::map_rgb`] for a function that takes the straight colours of many pixels at once (up
    /// to 64 of a row, in order) and replaces them in place: each pixel gets what `map_rgb` would
    /// give it with the function applied to its colour alone. Colour transforms whose stages wait
    /// on each other run several times faster on blocks (`InputTransform::apply_many`).
    pub fn map_rgb_many(&mut self, f: impl Fn(&mut [[f32; 3]]) + Sync) {
        const BLOCK: usize = 64;
        let w = self.w.max(1);
        self.px.par_chunks_mut(w * 4).for_each(|row| {
            let mut cs = [[0f32; 3]; BLOCK];
            for block in row.as_chunks_mut::<4>().0.chunks_mut(BLOCK) {
                for (c, p) in cs.iter_mut().zip(block.iter()) {
                    let a = p[3];
                    *c = if a == 1.0 {
                        [p[0], p[1], p[2]]
                    } else if a > 1e-6 {
                        [p[0] / a, p[1] / a, p[2] / a]
                    } else {
                        [0.0; 3]
                    };
                }
                let n = block.len();
                f(&mut cs[..n]);
                for (c, p) in cs.iter().zip(block.iter_mut()) {
                    let a = p[3];
                    if a == 1.0 {
                        p[..3].copy_from_slice(c);
                    } else if a > 1e-6 {
                        p[0] = c[0] * a;
                        p[1] = c[1] * a;
                        p[2] = c[2] * a;
                    }
                }
            }
        });
    }

    /// Convert to straight sRGB RGBA8.
    pub fn to_rgba8(&self) -> Vec<u8> {
        if self.w == 0 || self.h == 0 {
            return Vec::new();
        }
        let mut out = vec![0u8; self.w * self.h * 4];
        out.par_chunks_mut(self.w * 4).zip(self.px.par_chunks(self.w * 4)).for_each(|(o, s)| filmcraft_frame::linear_premul_to_srgb8(s, o));
        out
    }

    /// Opaque version over black (for monitors).
    pub fn over_black_rgba8(&self) -> Vec<u8> {
        if self.w == 0 || self.h == 0 {
            return Vec::new();
        }
        let mut out = vec![0u8; self.w * self.h * 4];
        // `linear_to_srgb_u8` per sample, with its table looked up once per frame (every exported frame comes through here)
        let table = filmcraft_color::linear_to_srgb_u8_table();
        let encode = |v: f32| table[((v.clamp(0.0, 1.0) * 4095.0 + 0.5) as usize).min(4095)];
        out.par_chunks_mut(self.w * 4).zip(self.px.par_chunks(self.w * 4)).for_each(|(o, s)| {
            for (o, s) in o.as_chunks_mut::<4>().0.iter_mut().zip(s.as_chunks::<4>().0) {
                *o = [encode(s[0]), encode(s[1]), encode(s[2]), 255];
            }
        });
        out
    }

    /// [`Self::over_black_rgba8`] with each pixel's alpha in the fourth byte: the color bytes are
    /// the same (the sRGB of the premultiplied linear color), so the picture still shows over
    /// black, and the alpha says where it is transparent.
    pub fn over_black_with_alpha_rgba8(&self) -> Vec<u8> {
        if self.w == 0 || self.h == 0 {
            return Vec::new();
        }
        let mut out = vec![0u8; self.w * self.h * 4];
        out.par_chunks_mut(self.w * 4).zip(self.px.par_chunks(self.w * 4)).for_each(|(o, s)| {
            for (o, s) in o.as_chunks_mut::<4>().0.iter_mut().zip(s.as_chunks::<4>().0) {
                o[0] = filmcraft_color::linear_to_srgb_u8(s[0]);
                o[1] = filmcraft_color::linear_to_srgb_u8(s[1]);
                o[2] = filmcraft_color::linear_to_srgb_u8(s[2]);
                o[3] = (s[3].clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`Image::transformed`] as it was, every pixel through [`Image::sample_bilinear`].
    fn transformed_per_pixel(img: &Image, w: usize, h: usize, m: &Affine) -> Image {
        let inv = m.inverse().unwrap();
        let mut out = Image::new(w, h);
        let b = m.bounds(&filmcraft_geom::Rect::new(0.0, 0.0, img.w as f64, img.h as f64));
        let ys = (b.y.floor().max(0.0) as usize).min(h)..(b.bottom().ceil().max(0.0) as usize).min(h);
        let xs = (b.x.floor().max(0.0) as usize).min(w)..(b.right().ceil().max(0.0) as usize).min(w);
        for y in ys {
            for x in xs.clone() {
                let (px, py) = (x as f64 + 0.5, y as f64 + 0.5);
                let (u, v) = (inv.a * px + inv.c * py + inv.e, inv.b * px + inv.d * py + inv.f);
                if u < -1.0 || v < -1.0 || u > img.w as f64 + 1.0 || v > img.h as f64 + 1.0 {
                    continue;
                }
                let s = img.sample_bilinear(u as f32, v as f32);
                out.px[(y * w + x) * 4..(y * w + x) * 4 + 4].copy_from_slice(&s);
            }
        }
        out
    }

    /// Mapping colours many at a time gives every pixel what mapping it alone gives: opaque,
    /// partly transparent, nearly and wholly transparent pixels, rows longer and shorter than a
    /// block.
    #[test]
    fn mapping_many_colours_matches_one_at_a_time() {
        let f = |c: [f32; 3]| [c[0] * 1.5 + c[2], c[1].sqrt(), (c[0] - c[1]) * 0.7];
        for (w, h) in [(1, 1), (63, 2), (64, 3), (130, 5)] {
            let alphas = [1.0f32, 0.5, 0.0, 1e-7, 0.25, 1.0, 0.999];
            let px: Vec<f32> = (0..w * h)
                .flat_map(|i| {
                    let a = alphas[i % alphas.len()];
                    [((i * 37) % 100) as f32 / 100.0 * a, ((i * 59) % 100) as f32 / 100.0 * a, ((i * 83) % 100) as f32 / 100.0 * a, a]
                })
                .collect();
            let (mut one, mut many) = (Image { w, h, px: px.clone() }, Image { w, h, px });
            one.map_rgb(|c, _, _| f(c));
            many.map_rgb_many(|cs| cs.iter_mut().for_each(|c| *c = f(*c)));
            assert!(one.px.iter().zip(&many.px).all(|(a, b)| a.to_bits() == b.to_bits()), "{w}x{h}");
        }
    }

    /// Placing a picture with Motion's scale and position (no rotation) samples whole rows at a
    /// time now; every pixel is what the per-pixel sampling gave, to the bit, enlarged or reduced,
    /// mirrored, moved by fractions of a pixel and partly or wholly off the frame.
    #[test]
    fn scaled_placement_samples_the_same_bits() {
        let (sw, sh) = (37usize, 23usize);
        let img = Image { w: sw, h: sh, px: (0..sw * sh * 4).map(|i| ((i * 7919) % 1000) as f32 / 997.0).collect() };
        let cases = [
            Affine::IDENTITY,
            Affine::scale(1.7777, 1.7777),
            Affine::scale(0.5625, 0.5625).then_apply(&Affine::translate(3.25, -2.5)),
            Affine::scale(2.0, 3.0).then_apply(&Affine::translate(-10.3, 4.7)),
            Affine::scale(-1.0, 1.0).then_apply(&Affine::translate(37.0, 0.0)),
            Affine::scale(1.0, -0.8).then_apply(&Affine::translate(0.5, 30.0)),
            Affine::translate(-40.0, 0.0),
            Affine::translate(0.0, 500.0),
        ];
        for m in cases {
            let want = transformed_per_pixel(&img, 64, 40, &m);
            let got = img.transformed(64, 40, &m);
            assert!(got.px.iter().zip(&want.px).all(|(a, b)| a.to_bits() == b.to_bits()), "{m:?}");
        }
    }

    /// The resampled image comes from the float pool with whatever its buffer held: every sample
    /// is still written, so a buffer full of NaN gives the same bits as a zeroed one, placed,
    /// rotated, off the frame or not.
    #[test]
    fn transformed_writes_every_sample_of_a_recycled_buffer() {
        let (sw, sh) = (41usize, 29usize);
        let img = Image { w: sw, h: sh, px: (0..sw * sh * 4).map(|i| ((i * 104_729) % 1000) as f32 / 999.0).collect() };
        let (w, h) = (130usize, 90usize);
        let cases = [
            Affine::scale(2.0, 2.0),
            Affine::scale(3.1, 3.1).then_apply(&Affine::translate(-7.3, 2.2)),
            Affine::scale(1.3, 0.9).then_apply(&Affine::translate(100.0, 70.0)),
            Affine::rotate_deg(17.0).then_apply(&Affine::translate(30.0, 5.0)),
            Affine::translate(-500.0, 0.0),
            Affine::translate(0.0, 1e6),
        ];
        for m in cases {
            let want = transformed_per_pixel(&img, w, h, &m);
            for _ in 0..3 {
                filmcraft_frame::pool::recycle_f32(vec![f32::NAN; w * h * 4]);
            }
            let got = img.transformed(w, h, &m);
            assert!(got.px.iter().zip(&want.px).all(|(a, b)| a.to_bits() == b.to_bits()), "{m:?}");
        }
        // degenerate sizes give an empty image, not a panic
        assert_eq!(img.transformed(0, 5, &Affine::IDENTITY).px.len(), 0);
        assert_eq!(img.transformed(5, 0, &Affine::IDENTITY).px.len(), 0);
    }
}
