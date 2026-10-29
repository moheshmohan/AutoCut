//! A logo over the bottom of a 360° clip's sphere, where the hand and stick are (the patch itself,
//! mirrored or smooth, is the media's VR Properties `nadir`: `filmcraft_render::vr::NadirPx`).
//!
//! | command | does |
//! |---|---|
//! | `clip.nadirLogo` | the user's `image` (a picture of the project) laid round the bottom of a 360° clip's sphere, `size` degrees (24) from straight down: a whole-sphere picture of it (transparent elsewhere) above the clip, seen through the clip's own virtual camera, so it stays on the ground as the view turns |

use filmcraft_project::{ClipId, ItemId};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, f64_p, has_seq, str_p, u64_p};
use crate::{EngineError, Result, Session};

const CMD: &str = "clip.nadirLogo";
/// The whole-sphere picture's width (2:1).
const WIDTH: usize = 2048;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Nadir Logo",
        menu: &["Clip", "Modify"],
        shortcut: None,
        params: r#"{"clip":id?,"image":id,"size":5..60=24,"folder":str?}"#,
        enabled: has_seq,
        run: nadir_logo,
        journal: true,
    }]
}

/// `logo` (RGBA8 `lw` × `lh`, straight alpha) laid round the bottom of a whole sphere, `size`
/// degrees from straight down, its top towards the front: a `w` × `w`/2 RGBA8 picture,
/// transparent elsewhere, the logo's edge softened by a pixel or two.
pub fn sphere_logo(logo: &[u8], lw: usize, lh: usize, size: f64, w: usize) -> Vec<u8> {
    let (w, h) = (w.clamp(16, 8192), (w.clamp(16, 8192) / 2).max(8));
    let mut out = vec![0u8; w * h * 4];
    if lw == 0 || lh == 0 || logo.len() < lw * lh * 4 {
        return out;
    }
    let r = size.clamp(1.0, 90.0).to_radians();
    let soft = 1.5 * std::f64::consts::PI / h as f64;
    for y in 0..h {
        // the angle from straight down
        let a = std::f64::consts::PI * (1.0 - (y as f64 + 0.5) / h as f64);
        if a > r + soft {
            continue;
        }
        for x in 0..w {
            let az = (x as f64 + 0.5) / w as f64 * std::f64::consts::TAU - std::f64::consts::PI;
            let k = a / r;
            // the logo square round the circle: its middle at straight down
            let (u, v) = (0.5 + 0.5 * k * az.sin(), 0.5 - 0.5 * k * az.cos());
            if !(0.0..1.0).contains(&u) || !(0.0..1.0).contains(&v) {
                continue;
            }
            let (px, py) = (((u * lw as f64) as usize).min(lw - 1), ((v * lh as f64) as usize).min(lh - 1));
            let Some(src) = logo.get((py * lw + px) * 4..(py * lw + px) * 4 + 4) else { continue };
            let edge = ((r + soft - a) / (2.0 * soft)).clamp(0.0, 1.0);
            let i = (y * w + x) * 4;
            if let Some(dst) = out.get_mut(i..i + 4) {
                dst[..3].copy_from_slice(&src[..3]);
                dst[3] = (f64::from(src[3]) * edge).round() as u8;
            }
        }
    }
    out
}

fn nadir_logo(s: &mut Session, p: &Value) -> Result<Value> {
    let clips = crate::vr360::clips_360(s, p, CMD)?;
    let it = clips.first().cloned().ok_or_else(|| bad(CMD, "no 360° clip given or selected"))?;
    if it.effect(filmcraft_project::vr::REFRAME).is_none() {
        return Err(bad(CMD, "the clip is not seen through a virtual camera (clip.reframe360 adds one)"));
    }
    if (it.speed - 1.0).abs() > 1e-6 || it.reverse || it.frame_hold.is_some() {
        return Err(bad(CMD, "the clip plays at another speed, backwards or held: lay the logo before changing its time"));
    }
    let image = u64_p(p, "image").map(ItemId).ok_or_else(|| bad(CMD, "need `image` (a picture of the project)"))?;
    let size = match f64_p(p, "size") {
        None => 24.0,
        Some(v) if v.is_finite() && (5.0..=60.0).contains(&v) => v,
        Some(_) => return Err(bad(CMD, "`size` is 5 to 60 degrees")),
    };
    let src = s.source(image).ok_or_else(|| bad(CMD, format!("no picture for item {}", image.0)))?;
    let f =
        src.video_frame(filmcraft_media::FrameRequest { time: Tick::ZERO, scale: 1.0 }).map_err(|e| bad(CMD, format!("the picture does not decode: {e}")))?;
    let rgba = sphere_logo(&f.to_rgba8(), f.width as usize, f.height as usize, size, WIDTH);
    let dir = match str_p(p, "folder") {
        Some(d) => std::path::PathBuf::from(crate::export_tools::expand_home(d)),
        None => crate::autosave::default_data_dir().map(|d| d.join("nadir-logos")).ok_or_else(|| bad(CMD, "no data directory for the logo"))?,
    };
    std::fs::create_dir_all(&dir).map_err(|e| bad(CMD, format!("{}: {e}", dir.display())))?;
    let path = dir.join(format!("nadir-{}-{}.png", image.0, size.round()));
    let png = filmcraft_export::encode_png(rgba, WIDTH as u32, (WIDTH / 2) as u32).map_err(|e| bad(CMD, e.to_string()))?;
    let path_s = path.to_string_lossy().into_owned();
    s.services.write_file(&path_s, &png).map_err(|e| bad(CMD, format!("{path_s}: {e}")))?;
    let n0 = s.history.undo.len();
    let r = s.execute("file.import", json!({"paths": [path_s], "reuse": true}))?;
    let item = r["items"][0].as_u64().map(ItemId).ok_or_else(|| bad(CMD, "the logo did not import"))?;
    s.execute("media.vrProperties", json!({"items": [item.0], "projection": "equirectangular"}))?;
    let seq_id = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let track = s.active_sequence().and_then(|q| q.video_tracks.iter().position(|t| t.items.iter().any(|i| i.id == it.id))).unwrap_or(0);
    let base = it.clone();
    let clip = s.edit("Nadir Logo", |pr, st| {
        let mut c = base.clone();
        c.id = ClipId(pr.alloc_id());
        c.item = item;
        c.name = format!("{} (nadir logo)", base.name);
        c.link = None;
        c.group = None;
        c.markers.clear();
        c.source_in = Tick::ZERO;
        // the sphere through the clip's own camera; no masks of the clip's
        c.effects.retain(|e| e.effect == "motion" || e.effect == "opacity" || e.effect == filmcraft_project::vr::REFRAME);
        if let Some(op) = c.effect_mut("opacity") {
            op.masks.clear();
        }
        crate::cutout::shift_keys(&mut c, base.source_in);
        let id = c.id;
        let mut tracks = vec![pr.alloc_id()];
        let q = pr.sequence_mut(seq_id).ok_or(EngineError::NoSequence)?;
        crate::cutout::place_above(q, track, vec![c], &mut || tracks.pop().unwrap_or(0));
        q.check().map_err(EngineError::Other)?;
        st.selection = vec![id];
        Ok(id)
    })?;
    crate::clip_ops::collapse_history(s, n0, "Nadir Logo");
    Ok(json!({"clip": it.id.0, "logo": clip.0, "item": item.0, "picture": path_s}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_logo_is_laid_round_the_bottom_of_the_sphere() {
        // a red square with a white top half
        let (lw, lh) = (32usize, 32usize);
        let logo: Vec<u8> = (0..lw * lh).flat_map(|i| if i / lw < lh / 2 { [255, 255, 255, 255] } else { [255, 0, 0, 255] }).collect();
        let sphere = sphere_logo(&logo, lw, lh, 20.0, 360);
        let (w, h) = (360usize, 180usize);
        let at = |x: usize, y: usize| &sphere[(y * w + x) * 4..(y * w + x) * 4 + 4];
        assert_eq!(at(10, 10)[3], 0, "the sky is left alone");
        assert_eq!(at(180, 90)[3], 0, "the horizon too");
        assert!(at(180, h - 1)[3] > 200, "straight down: the logo");
        // forward (the middle column) towards the logo's top: white; behind: red
        assert_eq!(&at(180, h - 12)[..3], &[255, 255, 255]);
        assert_eq!(&at(1, h - 12)[..3], &[255, 0, 0]);
        // hostile pictures
        assert!(sphere_logo(&[], 0, 0, 20.0, 64).iter().all(|v| *v == 0));
        assert!(sphere_logo(&logo, lw, lh, f64::NAN, 64).len() == 64 * 32 * 4);
    }
}
