//! Frames of project items as images, for agents and analysis: what is in this clip at 12 s?
//!
//! | command | does |
//! |---|---|
//! | `media.frame` | decode the frame of a media item at `seconds` (default: the middle), scaled so its long side is at most `maxSide`, as PNG (`pngBase64`, or written to `path`) |
//!
//! The frame is the source as decoded (display rotation applied by the decoders), before any
//! clip effects — what the camera saw.
//!
//! It is the frame *nearest* `seconds` (the decoders' lookup: files that store frame times
//! rounded would land a strict "frame on screen at" lookup one frame short), so a time exactly
//! between two frames can give either. To step through frames, ask at their starts (frame `n` at
//! `n / fps`): sampling at half-frame times repeats some frames and skips others. `seconds` in
//! the reply is the time asked for (past the end: the last frame's), not the frame's own time.

use serde_json::{Value, json};

use filmcraft_project::{ItemId, ItemKind};
use filmcraft_time::Tick;

use crate::commands::{CommandSpec, always, bad, f64_p, str_p, u64_p};
use crate::{Result, Session};

/// Largest image `maxSide` allows.
pub(crate) const MAX_SIDE: u32 = 4096;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "media.frame",
        label: "Media Frame",
        menu: &[],
        shortcut: None,
        params: r#"{"item":id,"seconds":f64?,"maxSide":u32=512,"path":str?}"#,
        enabled: always,
        run: frame,
        journal: false,
    }]
}

/// Area-average down-scale of straight RGBA8 so the long side is at most `max_side`.
pub fn downscale_rgba(rgba: &[u8], w: u32, h: u32, max_side: u32) -> (Vec<u8>, u32, u32) {
    let long = w.max(h);
    if long <= max_side || w == 0 || h == 0 || max_side == 0 {
        return (rgba.to_vec(), w, h);
    }
    let f = f64::from(max_side) / f64::from(long);
    let nw = ((f64::from(w) * f).round() as u32).max(1);
    let nh = ((f64::from(h) * f).round() as u32).max(1);
    let (w, h, nw_u, nh_u) = (w as usize, h as usize, nw as usize, nh as usize);
    let mut out = vec![0u8; nw_u * nh_u * 4];
    for y in 0..nh_u {
        let y0 = y * h / nh_u;
        let y1 = ((y + 1) * h / nh_u).max(y0 + 1).min(h);
        for x in 0..nw_u {
            let x0 = x * w / nw_u;
            let x1 = ((x + 1) * w / nw_u).max(x0 + 1).min(w);
            let mut acc = [0u32; 4];
            let mut n = 0u32;
            for sy in y0..y1 {
                for sx in x0..x1 {
                    let i = (sy * w + sx) * 4;
                    if let Some(px) = rgba.get(i..i + 4) {
                        for (a, v) in acc.iter_mut().zip(px) {
                            *a += u32::from(*v);
                        }
                        n += 1;
                    }
                }
            }
            let o = (y * nw_u + x) * 4;
            if let Some(dst) = out.get_mut(o..o + 4) {
                for (d, a) in dst.iter_mut().zip(acc) {
                    *d = (a / n.max(1)) as u8;
                }
            }
        }
    }
    (out, nw, nh)
}

/// A decoded frame: straight RGBA8, its size, and the media time it shows (seconds).
pub(crate) struct Decoded {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub seconds: f64,
}

/// The length of media item `item`, if it is video media.
pub(crate) fn video_length(s: &Session, cmd: &str, item: ItemId) -> Result<Tick> {
    let it = s.project.item(item).ok_or_else(|| bad(cmd, format!("no item {}", item.0)))?;
    let dur = match &it.kind {
        ItemKind::Media(m) => m.duration(),
        _ => return Err(bad(cmd, format!("item {} is not media", item.0))),
    };
    if !it.has_video() {
        return Err(bad(cmd, format!("item {} has no video", item.0)));
    }
    Ok(dur)
}

/// The frame of media item `item` nearest `secs` (default: the middle; past the end: the last
/// frame; see the module doc for times between frames), its long side at most `max_side`
/// (16..=[`MAX_SIDE`]).
pub(crate) fn decode(s: &Session, cmd: &str, item: ItemId, secs: Option<f64>, max_side: u32) -> Result<Decoded> {
    let dur = video_length(s, cmd, item)?;
    let secs = match secs {
        Some(x) if x.is_finite() && x >= 0.0 => x,
        Some(_) => return Err(bad(cmd, "`seconds` must be zero or more")),
        None => dur.seconds() / 2.0,
    };
    let max_side = max_side.clamp(16, MAX_SIDE);
    // (360° media as its stitched panorama: what the camera saw, all round)
    let src = crate::vr360::analysis_view(s, item).ok_or_else(|| bad(cmd, "the item's media is offline or unreadable"))?;
    let info = src.info().video.clone().ok_or_else(|| bad(cmd, "the item has no video"))?;
    // the last frame starts one frame before the end
    let last = (dur - info.frame_rate.frame_duration()).max(Tick::ZERO);
    let t = Tick::from_seconds_f64(secs).min(last);
    let scale = (f64::from(max_side) / f64::from(info.width.max(info.height).max(1))).clamp(0.05, 1.0) as f32;
    let vf = src.video_frame(filmcraft_media::FrameRequest { time: t, scale }).map_err(|e| bad(cmd, format!("cannot decode the frame: {e}")))?;
    let (rgba, w, h) = downscale_rgba(&vf.to_rgba8(), vf.width, vf.height, max_side);
    Ok(Decoded { rgba, width: w, height: h, seconds: t.seconds() })
}

fn frame(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "media.frame";
    let item = ItemId(u64_p(p, "item").ok_or_else(|| bad(cmd, "`item` is required"))?);
    let max_side = u64_p(p, "maxSide").unwrap_or(512).clamp(16, u64::from(MAX_SIDE)) as u32;
    let d = decode(s, cmd, item, f64_p(p, "seconds"), max_side)?;
    let (w, h, t) = (d.width, d.height, d.seconds);
    let png = filmcraft_export::encode_png(d.rgba, w, h).map_err(|e| bad(cmd, e.to_string()))?;
    let mut out = json!({"item": item.0, "seconds": t, "width": w, "height": h});
    match str_p(p, "path") {
        Some(path) => {
            s.services.write_file(path, &png).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
            out["path"] = json!(path);
        }
        None => out["pngBase64"] = json!(filmcraft_project::gtemplate::base64_encode(&png)),
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn downscale_keeps_aspect_and_averages() {
        // 4x2 image: left half black, right half white
        let mut img = vec![0u8; 4 * 2 * 4];
        for y in 0..2 {
            for x in 2..4 {
                let i = (y * 4 + x) * 4;
                img[i..i + 4].copy_from_slice(&[255, 255, 255, 255]);
            }
        }
        let (out, w, h) = downscale_rgba(&img, 4, 2, 2);
        assert_eq!((w, h), (2, 1));
        assert_eq!(&out[0..4], &[0, 0, 0, 0]);
        assert_eq!(&out[4..8], &[255, 255, 255, 255]);
        // already small: unchanged
        assert_eq!(downscale_rgba(&img, 4, 2, 512).1, 4);
        // hostile sizes don't panic
        let _ = downscale_rgba(&[], 0, 0, 10);
        let _ = downscale_rgba(&[1, 2, 3], 100, 100, 10);
    }

    #[test]
    fn frames_of_demo_media() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let r = s.execute("media.frame", json!({"item": 5, "seconds": 1.0, "maxSide": 160})).unwrap();
        assert!(r["width"].as_u64().unwrap() <= 160 && r["height"].as_u64().unwrap() <= 160, "{r}");
        let png = filmcraft_project::gtemplate::base64_decode(r["pngBase64"].as_str().unwrap()).unwrap();
        assert!(png.starts_with(b"\x89PNG"));
        // past the end clamps to the last frame; bad requests are errors
        assert!(s.execute("media.frame", json!({"item": 5, "seconds": 1e9})).is_ok());
        assert!(s.execute("media.frame", json!({"item": 5, "seconds": -1.0})).is_err());
        assert!(s.execute("media.frame", json!({"item": 999})).is_err());
        assert!(s.execute("media.frame", json!({})).is_err());
        // an audio-only item has no frame
        assert!(s.execute("media.frame", json!({"item": 11})).is_err());
    }
}
