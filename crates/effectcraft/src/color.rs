//! Colour between AutoCut and EffectCraft.
//!
//! AutoCut's compositor works in premultiplied **linear** RGBA f32 ([`filmcraft_frame::PixelData::RgbaF32`]);
//! EffectCraft's renderer takes footage and gives frames as premultiplied RGBA f32 **encoded**
//! with the sRGB curve (its unmanaged default, Working Gamma 2.2; a managed project converts from
//! and to sRGB at its edges). Straight 8-bit sRGB pictures (PNG stills, AutoCut's RGBA8 frames)
//! go across without any curve: only the alpha is multiplied in or divided out.
//!
//! Every value is sanitised on the way: NaN and infinities become 0, alpha is held to 0..1 and a
//! pixel without alpha has no colour.

use filmcraft_color::{linear_to_srgb, srgb_to_linear};
use filmcraft_frame::{PixelData, VideoFrame};

/// One pixel (premultiplied RGBA).
pub type Px = [f32; 4];

fn finite(v: f32) -> f32 {
    if v.is_finite() { v } else { 0.0 }
}

/// The straight colour and alpha of a premultiplied pixel (sanitised).
fn unpremultiply(p: Px) -> ([f32; 3], f32) {
    let a = finite(p[3]).clamp(0.0, 1.0);
    if a <= 0.0 {
        return ([0.0; 3], 0.0);
    }
    ([finite(p[0]) / a, finite(p[1]) / a, finite(p[2]) / a], a)
}

/// A premultiplied linear pixel (AutoCut) as EffectCraft takes it: premultiplied, sRGB-encoded.
pub fn linear_to_encoded(p: Px) -> Px {
    let (c, a) = unpremultiply(p);
    if a <= 0.0 {
        return [0.0; 4];
    }
    let e = c.map(|v| finite(linear_to_srgb(v)));
    [e[0] * a, e[1] * a, e[2] * a, a]
}

/// A premultiplied sRGB-encoded pixel (EffectCraft) as AutoCut works with it: premultiplied linear.
pub fn encoded_to_linear(p: Px) -> Px {
    let (c, a) = unpremultiply(p);
    if a <= 0.0 {
        return [0.0; 4];
    }
    let l = c.map(|v| finite(srgb_to_linear(v)));
    [l[0] * a, l[1] * a, l[2] * a, a]
}

/// A straight sRGB RGBA8 pixel as EffectCraft takes it (no curve: both are sRGB-encoded).
pub fn rgba8_to_encoded(px: [u8; 4]) -> Px {
    let a = f32::from(px[3]) / 255.0;
    let c = |v: u8| f32::from(v) / 255.0 * a;
    [c(px[0]), c(px[1]), c(px[2]), a]
}

fn level(v: f32) -> u8 {
    (finite(v).clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// An EffectCraft pixel as straight sRGB RGBA8 (what PNG and the ProRes encoder take), rounded.
pub fn encoded_to_rgba8(p: Px) -> [u8; 4] {
    let (c, a) = unpremultiply(p);
    if a <= 0.0 {
        return [0; 4];
    }
    [level(c[0]), level(c[1]), level(c[2]), level(a)]
}

/// A whole EffectCraft frame as straight sRGB RGBA8.
pub fn frame_to_rgba8(px: &[Px]) -> Vec<u8> {
    let mut out = Vec::with_capacity(px.len().saturating_mul(4));
    for p in px {
        out.extend_from_slice(&encoded_to_rgba8(*p));
    }
    out
}

/// A host frame as EffectCraft footage pixels (premultiplied, sRGB-encoded): `None` when it is
/// empty or larger than `max_pixels`. `opaque` ignores its alpha (Interpret Footage ▸ Ignore).
pub fn host_to_encoded(f: &VideoFrame, opaque: bool, max_pixels: u64) -> Option<(u32, u32, Vec<Px>)> {
    let (w, h) = (f.width, f.height);
    let n = u64::from(w).checked_mul(u64::from(h))?;
    if n == 0 || n > max_pixels {
        return None;
    }
    let n = usize::try_from(n).ok()?;
    let mut out: Vec<Px> = match &f.data {
        // straight 8-bit sRGB: the stored levels, no curve
        PixelData::Rgba8(d) => {
            let px = d.get(..n.checked_mul(4)?)?;
            px.as_chunks::<4>().0.iter().map(|p| rgba8_to_encoded(*p)).collect()
        }
        // anything else through AutoCut's own decode to premultiplied linear
        _ => {
            let lin = f.to_linear_f32();
            let px = lin.get(..n.checked_mul(4)?)?;
            px.as_chunks::<4>().0.iter().map(|p| linear_to_encoded(*p)).collect()
        }
    };
    if opaque {
        for p in &mut out {
            let (c, a) = unpremultiply(*p);
            *p = if a > 0.0 { [c[0], c[1], c[2], 1.0] } else { [0.0, 0.0, 0.0, 1.0] };
        }
    }
    (out.len() == n).then_some((w, h, out))
}

/// An EffectCraft frame as an AutoCut frame (premultiplied linear RGBA f32); `None` when the
/// pixel count does not match the size.
pub fn encoded_to_host(width: u32, height: u32, px: &[Px]) -> Option<VideoFrame> {
    let n = usize::try_from(u64::from(width).checked_mul(u64::from(height))?).ok()?;
    if n == 0 || px.len() != n {
        return None;
    }
    let mut data = Vec::with_capacity(n.checked_mul(4)?);
    for p in px {
        data.extend_from_slice(&encoded_to_linear(*p));
    }
    Some(VideoFrame::rgba_f32(width, height, data))
}
