//! A measured, subtle colour balance (`clip.autoColor`): what a picture's near-neutral greys,
//! skin and exposure measure, the Lumetri Basic Correction that balances them, and whether a
//! correction made things worse — greys more coloured, skin moved off the skin-tone line,
//! highlights clipped — which backs it off.
//!
//! Measured in display terms (sRGB-encoded Y′CbCr, BT.709 weights) on a small render of the
//! clip's own picture; corrected in linear light, as Lumetri does: white balance and exposure are
//! gains, `[1 + 0.35·T, 1 − 0.3·t, 1 − 0.35·T] · 2^stops` for Temperature `100·T` and Tint `100·t`
//! (`crate::effects`'s Lumetri). Nothing is copied from a reference shot: matching cameras only
//! nudges saturation and contrast towards the edit's middle, never a hue (a warm reference once
//! turned a whole set of clips sepia).
//!
//! - **Greys**: the mid-tone pixels about as little coloured as the least coloured fifth of them
//!   (within 1.3 times its chroma), at most [`MAX_GREY_CHROMA`] — a white wall under warm light is
//!   still the least coloured thing in the room, though under warm light it sits on the
//!   skin-tone line itself. Their mean, in linear light, is the light's colour. Under
//!   [`MIN_GREY_SHARE`] of the picture, or not clearly less coloured than the skin in it (a
//!   close-up of a face has no grey), white balance is left alone.
//! - **Skin**: mid-tone pixels near the skin-tone line (hue 105°–145° in the CbCr plane, chroma
//!   0.03–0.2) clearly more coloured than the greys: their mean hue.
//!   White balance goes as far as the greys are sure: all the way at a quarter of the picture
//!   grey, less with fewer, half as far in a colourful picture ([`COLOURFUL`]).
//! - **Exposure**: the median luma, aimed at [`TARGET_LUMA`] and left alone inside
//!   [`LUMA_OK`] (a dim room may be dim); brightening never pushes the brightest 2 % past white.

use crate::image::Image;

/// The least share of the picture that must be near-neutral grey to balance white by it.
pub const MIN_GREY_SHARE: f64 = 0.03;
/// The most colour a "grey" may have (CbCr chroma, encoded).
pub const MAX_GREY_CHROMA: f64 = 0.08;
/// A grey this colourless needs no white balance.
pub const NEUTRAL_CAST: f64 = 0.01;
/// Where the median luma is aimed (encoded), and the range left alone.
pub const TARGET_LUMA: f64 = 0.45;
pub const LUMA_OK: (f64, f64) = (0.32, 0.6);
/// The skin-tone line's band of hues (degrees in the CbCr plane) and chroma.
pub const SKIN_HUES: (f64, f64) = (105.0, 145.0);
const SKIN_CHROMA: (f64, f64) = (0.03, 0.2);
/// Caps on a correction: Temperature and Tint (Lumetri's ±100 scale) and exposure (stops).
pub const MAX_TEMPERATURE: f64 = 20.0;
pub const MAX_TINT: f64 = 10.0;
pub const MAX_STOPS: f64 = 0.5;
/// A picture this colourful (mean chroma) is balanced half as far: its "greys" pick up the
/// colour around them (a red shop front, a green park).
pub const COLOURFUL: f64 = 0.09;

/// What a picture measures.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Stats {
    /// Luma (encoded): the median, the 2nd and 98th percentiles, and the 5th to 95th spread.
    pub median: f64,
    pub p02: f64,
    pub p98: f64,
    pub contrast: f64,
    /// Shares crushed to black and blown to white.
    pub dark: f64,
    pub blown: f64,
    /// The greys' mean colour (linear RGB), their share of the picture and their chroma (`None`:
    /// no grey to tell the light by).
    pub grey: Option<[f64; 3]>,
    pub grey_share: f64,
    pub cast: f64,
    /// The skin's mean hue (degrees) and share.
    pub skin_hue: Option<f64>,
    pub skin_share: f64,
    /// Mean chroma of the mid-tones.
    pub saturation: f64,
}

/// A Lumetri Basic Correction: Temperature, Tint (±100), Exposure (stops), Saturation (100 none),
/// Contrast (±100).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Correction {
    pub temperature: f64,
    pub tint: f64,
    pub exposure: f64,
    pub saturation: f64,
    pub contrast: f64,
}

impl Default for Correction {
    fn default() -> Self {
        Correction { temperature: 0.0, tint: 0.0, exposure: 0.0, saturation: 100.0, contrast: 0.0 }
    }
}

impl Correction {
    pub fn is_none(&self) -> bool {
        *self == Correction::default()
    }

    /// The correction `k` of the way (0 none, 1 all of it).
    pub fn scaled(&self, k: f64) -> Correction {
        let k = if k.is_finite() { k.clamp(0.0, 1.0) } else { 0.0 };
        Correction {
            temperature: r1(self.temperature * k),
            tint: r1(self.tint * k),
            exposure: r2(self.exposure * k),
            saturation: r1(100.0 + (self.saturation - 100.0) * k),
            contrast: r1(self.contrast * k),
        }
    }

    /// Its white balance alone taken away.
    pub fn without_white_balance(&self) -> Correction {
        Correction { temperature: 0.0, tint: 0.0, ..*self }
    }
}

fn r1(x: f64) -> f64 {
    if x.is_finite() { (x * 10.0).round() / 10.0 } else { 0.0 }
}

fn r2(x: f64) -> f64 {
    if x.is_finite() { (x * 100.0).round() / 100.0 } else { 0.0 }
}

fn encode(v: f64) -> f64 {
    f64::from(filmcraft_color::linear_to_srgb(v.clamp(0.0, 1.0) as f32))
}

fn decode(v: f64) -> f64 {
    f64::from(filmcraft_color::srgb_to_linear(v.clamp(0.0, 1.0) as f32))
}

/// Encoded Y′, Cb, Cr of a linear RGB colour.
fn ycc(rgb: [f64; 3]) -> (f64, f64, f64) {
    let [r, g, b] = rgb.map(encode);
    let y = 0.2126 * r + 0.7152 * g + 0.0722 * b;
    (y, (b - y) / 1.8556, (r - y) / 1.5748)
}

fn hue(cb: f64, cr: f64) -> f64 {
    cr.atan2(cb).to_degrees().rem_euclid(360.0)
}

fn on_skin_line(h: f64, c: f64) -> bool {
    (SKIN_HUES.0..=SKIN_HUES.1).contains(&h) && (SKIN_CHROMA.0..=SKIN_CHROMA.1).contains(&c)
}

/// The percentile `q` (0..1) of a 256-bin histogram of values 0..1.
fn percentile(hist: &[u64; 256], total: u64, q: f64) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let want = (q * total as f64).ceil().max(1.0) as u64;
    let mut acc = 0u64;
    for (i, n) in hist.iter().enumerate() {
        acc += n;
        if acc >= want {
            return (i as f64 + 0.5) / 256.0;
        }
    }
    1.0
}

/// What `img` (premultiplied linear RGBA) measures; transparent pixels are left out.
pub fn measure(img: &Image) -> Stats {
    let mut hist = [0u64; 256];
    let mut total = 0u64;
    let (mut dark, mut blown) = (0u64, 0u64);
    // mid-tones: chroma, colour, Cb, Cr, on the skin-tone line
    let mut mids: Vec<(f64, [f64; 3], f64, f64, bool)> = Vec::new();
    let (mut chroma_sum, mut chroma_n) = (0.0, 0u64);
    for p in img.px.as_chunks::<4>().0 {
        let a = f64::from(p[3]);
        if !(a.is_finite() && a >= 0.5) {
            continue;
        }
        let rgb = [f64::from(p[0]) / a, f64::from(p[1]) / a, f64::from(p[2]) / a];
        if rgb.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let (y, cb, cr) = ycc(rgb);
        total += 1;
        let bin = ((y * 256.0) as usize).min(255);
        if let Some(h) = hist.get_mut(bin) {
            *h += 1;
        }
        if y <= 0.02 {
            dark += 1;
        }
        if y >= 0.98 {
            blown += 1;
        }
        let c = (cb * cb + cr * cr).sqrt();
        if (0.15..=0.85).contains(&y) {
            chroma_sum += c;
            chroma_n += 1;
            mids.push((c, rgb, cb, cr, y >= 0.2 && on_skin_line(hue(cb, cr), c)));
        }
    }
    let n = total.max(1) as f64;
    let mut st = Stats {
        median: percentile(&hist, total, 0.5),
        p02: percentile(&hist, total, 0.02),
        p98: percentile(&hist, total, 0.98),
        contrast: percentile(&hist, total, 0.95) - percentile(&hist, total, 0.05),
        dark: dark as f64 / n,
        blown: blown as f64 / n,
        saturation: if chroma_n > 0 { chroma_sum / chroma_n as f64 } else { 0.0 },
        ..Stats::default()
    };
    // the greys: about as little coloured as the least coloured fifth; the skin: clearly more
    mids.sort_by(|a, b| a.0.total_cmp(&b.0));
    let c20 = mids.get(mids.len() / 5).map_or(0.0, |m| m.0);
    let grey_limit = (c20 * 1.3).max(c20 + 0.005).min(MAX_GREY_CHROMA);
    let greys: Vec<&(f64, [f64; 3], f64, f64, bool)> = mids.iter().filter(|m| m.0 <= grey_limit).collect();
    let skin_from = if greys.is_empty() { 0.0 } else { 1.6 * grey_limit };
    let skin: Vec<&(f64, [f64; 3], f64, f64, bool)> = mids.iter().filter(|m| m.4 && m.0 > skin_from).collect();
    st.skin_share = skin.len() as f64 / n;
    let skin_chroma = if skin.is_empty() { 0.0 } else { skin.iter().map(|m| m.0).sum::<f64>() / skin.len() as f64 };
    if !skin.is_empty() {
        let k = skin.len() as f64;
        st.skin_hue = Some(hue(skin.iter().map(|m| m.2).sum::<f64>() / k, skin.iter().map(|m| m.3).sum::<f64>() / k));
    }
    st.grey_share = greys.len() as f64 / n;
    if !greys.is_empty() {
        let k = greys.len() as f64;
        let mean = [0, 1, 2].map(|i| greys.iter().map(|m| m.1[i]).sum::<f64>() / k);
        let (_, cb, cr) = ycc(mean);
        st.cast = (cb * cb + cr * cr).sqrt();
        // greys on the skin-tone line beside skin hardly more coloured than them, or greys nearly
        // as coloured as the skin: no grey to tell the light by
        let on_line = on_skin_line(hue(cb, cr), st.cast);
        let unclear = (on_line && !skin.is_empty() && skin_chroma < 2.0 * st.cast) || (st.skin_share > 0.05 && st.cast > 0.6 * skin_chroma);
        st.grey = (!unclear).then_some(mean);
    }
    st
}

/// The correction that balances `st`, `strength` of the way (0..1), and what it did in words.
pub fn balance(st: &Stats, strength: f64) -> (Correction, Vec<String>) {
    let k = if strength.is_finite() { strength.clamp(0.0, 1.0) } else { 0.0 };
    let mut c = Correction::default();
    let mut said = Vec::new();
    match st.grey {
        Some(g) if st.grey_share >= MIN_GREY_SHARE && st.cast >= NEUTRAL_CAST && g.iter().all(|v| *v > 1e-4) => {
            // red and blue level with each other, then green with them
            let (r, gr, b) = (g[0], g[1], g[2]);
            let t_full = (b - r) / (0.35 * (r + b));
            let level = (1.0 + 0.35 * t_full) * r;
            let tint_full = (1.0 - level / gr) / 0.3;
            // as far as the greys are sure
            let sure = ((st.grey_share - MIN_GREY_SHARE) / (0.25 - MIN_GREY_SHARE)).clamp(0.25, 1.0) * if st.saturation > COLOURFUL { 0.5 } else { 1.0 };
            c.temperature = r1((100.0 * t_full * k * sure).clamp(-MAX_TEMPERATURE, MAX_TEMPERATURE));
            c.tint = r1((100.0 * tint_full * k * sure).clamp(-MAX_TINT, MAX_TINT));
            said.push(format!(
                "white balance: the greys were {} (temperature {:+.0}, tint {:+.0})",
                if t_full < 0.0 { "warm" } else { "cool" },
                c.temperature,
                c.tint
            ));
        }
        Some(_) if st.grey_share >= MIN_GREY_SHARE => said.push("white balance: already neutral".into()),
        _ => said.push(format!("white balance: left alone (too little grey to measure by: {:.0} %)", st.grey_share * 100.0)),
    }
    if st.median > 0.0 && !(LUMA_OK.0..=LUMA_OK.1).contains(&st.median) {
        let mut stops = (decode(TARGET_LUMA) / decode(st.median).max(1e-4)).log2() * k;
        if stops > 0.0 && st.p98 > 0.0 {
            // the brightest 2 % stay under white
            stops = stops.min((decode(0.985) / decode(st.p98).max(1e-4)).log2().max(0.0));
        }
        let stops = stops.clamp(-MAX_STOPS, MAX_STOPS);
        if stops.abs() >= 0.1 {
            c.exposure = r2(stops);
            said.push(format!("exposure {:+.2} stops (median luma {:.2})", c.exposure, st.median));
        }
    }
    (c, said)
}

/// The saturation and contrast that bring a shot `k` of the way to the edit's middle (`target`
/// saturation and contrast): cameras matched by how vivid and how contrasty they look, never by
/// hue. Small: saturation 88–112, contrast ±10.
pub fn match_to(st: &Stats, target_saturation: f64, target_contrast: f64, k: f64) -> (f64, f64) {
    let k = if k.is_finite() { k.clamp(0.0, 1.0) } else { 0.0 };
    let sat = if st.saturation > 0.005 && target_saturation.is_finite() && target_saturation > 0.0 {
        100.0 * (1.0 + k * (target_saturation / st.saturation - 1.0))
    } else {
        100.0
    };
    let con = if st.contrast > 0.05 && target_contrast.is_finite() { 100.0 * k * (target_contrast - st.contrast) / st.contrast } else { 0.0 };
    (r1(sat.clamp(88.0, 112.0)), r1(con.clamp(-10.0, 10.0)))
}

/// What a correction made worse, comparing the picture `before` it and `after` it; `None`
/// when it did no harm.
pub fn drift(before: &Stats, after: &Stats) -> Option<String> {
    if before.grey.is_some() && after.grey.is_some() && after.cast > before.cast + 0.008 {
        return Some(format!("the greys came out more coloured ({:.3} → {:.3})", before.cast, after.cast));
    }
    if let (Some(a), Some(b)) = (before.skin_hue, after.skin_hue) {
        let d = (b - a).abs().min(360.0 - (b - a).abs());
        let inside = |h: f64| (SKIN_HUES.0 + 5.0..=SKIN_HUES.1 - 5.0).contains(&h);
        if d > 8.0 && !inside(b) {
            return Some(format!("skin tones moved {d:.0}° off the skin-tone line ({a:.0}° → {b:.0}°)"));
        }
    }
    if after.blown > before.blown + 0.02 {
        return Some(format!("highlights clipped ({:.0} % → {:.0} % white)", before.blown * 100.0, after.blown * 100.0));
    }
    if after.dark > before.dark + 0.05 {
        return Some(format!("shadows crushed ({:.0} % → {:.0} % black)", before.dark * 100.0, after.dark * 100.0));
    }
    None
}

/// `img` with a correction's white balance and exposure applied as Lumetri applies them (gains
/// in linear light): what a correction will do, for tests and previews.
pub fn apply_gains(img: &Image, c: &Correction) -> Image {
    let t = c.temperature / 100.0;
    let tint = c.tint / 100.0;
    let e = 2f64.powf(c.exposure);
    let g = [(1.0 + 0.35 * t) * e, (1.0 - 0.3 * tint) * e, (1.0 - 0.35 * t) * e].map(|v| v as f32);
    let mut out = img.clone();
    for p in out.px.as_chunks_mut::<4>().0 {
        for (v, k) in p.iter_mut().zip(g) {
            *v *= k;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A picture of a grey card (most of it), a skin patch and a bright patch, under a light of
    /// `light` (linear gains).
    fn scene(light: [f32; 3], gain: f32) -> Image {
        let (w, h) = (64usize, 36usize);
        let mut img = Image::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let base: [f32; 3] = if x < 16 && y < 12 {
                    // skin: a light brown
                    [0.55, 0.32, 0.22]
                } else if x > 56 {
                    [0.8, 0.8, 0.8]
                } else {
                    [0.18 + 0.002 * x as f32, 0.18 + 0.002 * x as f32, 0.18 + 0.002 * x as f32]
                };
                let i = (y * w + x) * 4;
                for c in 0..3 {
                    img.px[i + c] = base[c] * light[c] * gain;
                }
                img.px[i + 3] = 1.0;
            }
        }
        img
    }

    #[test]
    fn a_warm_cast_is_measured_and_balanced_back_to_grey() {
        let warm = scene([1.06, 1.0, 0.94], 1.0);
        let st = measure(&warm);
        assert!(st.grey.is_some() && st.grey_share > 0.5 && st.cast > 0.01, "{st:?}");
        let (full, said) = balance(&st, 1.0);
        assert!(full.temperature < -8.0 && said[0].contains("warm"), "{full:?} {said:?}");
        // applied, the greys are grey again
        let after = measure(&apply_gains(&warm, &full));
        assert!(after.cast < 0.012, "{:?}", after.cast);
        assert!(drift(&st, &after).is_none());
        // at 70 % strength, 70 % of the way
        let (part, _) = balance(&st, 0.7);
        assert!((part.temperature - 0.7 * full.temperature).abs() < 1.0, "{part:?} {full:?}");
        // a colourful picture is balanced half as far
        let colourful = Stats { saturation: COLOURFUL + 0.01, ..st.clone() };
        assert!((balance(&colourful, 1.0).0.temperature - 0.5 * full.temperature).abs() < 0.6);
        // a strong cast is only taken part of the way: capped
        let (strong, _) = balance(&measure(&scene([1.3, 1.0, 0.7], 1.0)), 1.0);
        assert_eq!(strong.temperature, -MAX_TEMPERATURE);
        // a neutral picture is left alone
        let (none, said) = balance(&measure(&scene([1.0, 1.0, 1.0], 1.0)), 1.0);
        assert_eq!((none.temperature, none.tint), (0.0, 0.0), "{said:?}");
    }

    #[test]
    fn exposure_is_aimed_at_the_middle_without_blowing_highlights() {
        let dark = scene([1.0, 1.0, 1.0], 0.25);
        let st = measure(&dark);
        assert!(st.median < LUMA_OK.0);
        let (c, _) = balance(&st, 1.0);
        assert!(c.exposure >= 0.25 && c.exposure <= MAX_STOPS, "{c:?}");
        let after = measure(&apply_gains(&dark, &c));
        assert!(after.blown <= st.blown + 0.02, "{after:?}");
        // well exposed: nothing
        assert_eq!(balance(&measure(&scene([1.0, 1.0, 1.0], 1.0)), 1.0).0.exposure, 0.0);
    }

    #[test]
    fn only_skin_and_colour_is_not_white_balanced() {
        let (w, h) = (16usize, 16usize);
        let mut img = Image::new(w, h);
        for p in img.px.as_chunks_mut::<4>().0 {
            *p = [0.55, 0.32, 0.22, 1.0];
        }
        let st = measure(&img);
        assert!(st.skin_share > 0.9 && st.grey.is_none(), "{st:?}");
        let (c, said) = balance(&st, 1.0);
        assert_eq!((c.temperature, c.tint), (0.0, 0.0));
        assert!(said[0].contains("left alone"));
    }

    #[test]
    fn harm_is_found_and_cameras_meet_in_the_middle() {
        let before = measure(&scene([1.0, 1.0, 1.0], 1.0));
        // pushed far too warm: greys coloured and skin off the line
        let after = measure(&apply_gains(&scene([1.0, 1.0, 1.0], 1.0), &Correction { temperature: 80.0, tint: -60.0, ..Correction::default() }));
        assert!(drift(&before, &after).is_some(), "{before:?} {after:?}");
        let blown = measure(&apply_gains(&scene([1.0, 1.0, 1.0], 1.0), &Correction { exposure: 2.0, ..Correction::default() }));
        assert!(drift(&before, &blown).is_some_and(|d| d.contains("clipped")));
        // a flat camera and a punchy one meet half way, within bounds
        let st = Stats { saturation: 0.05, contrast: 0.5, ..Stats::default() };
        let (sat, con) = match_to(&st, 0.08, 0.7, 0.5);
        assert!(sat > 100.0 && sat <= 112.0 && con > 0.0 && con <= 10.0, "{sat} {con}");
        assert_eq!(match_to(&Stats::default(), 0.08, 0.7, 0.5), (100.0, 0.0));
    }

    #[test]
    fn hostile_pictures_never_panic() {
        let mut img = Image::new(8, 8);
        for (i, v) in img.px.iter_mut().enumerate() {
            *v = [f32::NAN, f32::INFINITY, -1.0, 1e30][i % 4];
        }
        let st = measure(&img);
        let (c, _) = balance(&st, f64::NAN);
        assert!([c.temperature, c.tint, c.exposure, c.saturation, c.contrast].iter().all(|v| v.is_finite()));
        let _ = measure(&Image::new(0, 0));
        assert!(c.scaled(f64::INFINITY).temperature.is_finite());
    }
}
