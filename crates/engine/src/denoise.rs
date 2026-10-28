//! Cleaner night shots: noise reduction across neighbouring frames (`filmcraft_render::denoise`:
//! each frame merged with its neighbours, each first moved to line up with it and taken only where
//! it agrees, so grain averages away and what moves does not smear; the merge on the GPU).
//!
//! | command | does |
//! |---|---|
//! | `media.denoise` | noise reduction on `items` (default: the selected footage) at `strength` 0..1 (0.5) over `frames` either side 1..3 (2), or off (`off: true`); with `auto: true` only on the dark, noisy ones (noise ≥ 1.5 levels and light under 0.3, measured at the middle frame), the others left as they are; one undo step. With `auto` or `measure: true` the answer gives each item's `noise` (σ, 8-bit levels) and `light` (mean luma 0..1) |

use filmcraft_media::FrameRequest;
use filmcraft_project::{Denoise, ItemId, ItemKind};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, u64_p};
use crate::{Result, Session};

const CMD: &str = "media.denoise";
/// How hard, by default.
pub const STRENGTH: f32 = 0.5;
/// Frames either side, by default.
pub const FRAMES: u8 = 2;
/// `auto`: noisier than this (σ, 8-bit levels)…
pub const NOISY: f32 = 1.5;
/// …and darker than this (mean luma).
pub const DARK: f32 = 0.3;
/// Most items one call measures.
const MAX_ITEMS: usize = 200;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: CMD,
        label: "Reduce Noise",
        menu: &["Clip", "Modify"],
        shortcut: None,
        params: r#"{"items":[id]?,"strength":0..1=0.5,"frames":1..3=2,"off":bool=false,"auto":bool=false,"measure":bool=false}"#,
        enabled: always,
        run: denoise,
        journal: true,
    }]
}

/// An item's noise (σ, 8-bit levels) and light (mean luma 0..1) at its middle frame, as its
/// full-resolution source gives it (denoised when it is on; a proxy's compression would hide
/// the grain).
pub fn measure(s: &Session, item: ItemId) -> Option<(f32, f32)> {
    let src = s.media.full_res_source(&s.project, item, &*s.services)?;
    let info = src.info();
    if !info.has_video() {
        return None;
    }
    let f = src.video_frame(FrameRequest { time: Tick(info.duration.0 / 2), scale: 1.0 }).ok()?;
    let (p, _) = filmcraft_render::denoise::planes(&f)?;
    let (noise, light) = filmcraft_render::denoise::grain(&p[0]);
    Some((noise * 255.0, light))
}

fn denoise(s: &mut Session, p: &Value) -> Result<Value> {
    let items: Vec<ItemId> = crate::color::footage_targets(s, p)
        .into_iter()
        .filter(|i| s.project.item(*i).and_then(|it| it.as_media()).is_some_and(|m| m.info.has_video()))
        .take(MAX_ITEMS)
        .collect();
    if items.is_empty() {
        return Err(bad(CMD, "no video selected"));
    }
    let strength = match f64_p(p, "strength") {
        None => STRENGTH,
        Some(v) if v.is_finite() && (0.0..=1.0).contains(&v) => v as f32,
        Some(_) => return Err(bad(CMD, "`strength` is 0 to 1")),
    };
    let frames = match u64_p(p, "frames") {
        None if p.get("frames").is_some_and(|v| !v.is_null()) => return Err(bad(CMD, "`frames` is 1 to 3")),
        None => FRAMES,
        Some(v) if (1..=filmcraft_render::denoise::MAX_RADIUS as u64).contains(&v) => v as u8,
        Some(_) => return Err(bad(CMD, "`frames` is 1 to 3")),
    };
    let off = bool_p(p, "off").unwrap_or(false);
    let auto = bool_p(p, "auto").unwrap_or(false) && !off;
    let measured: Vec<(ItemId, Option<(f32, f32)>)> = if auto || bool_p(p, "measure").unwrap_or(false) {
        items.iter().map(|i| (*i, measure(s, *i))).collect()
    } else {
        items.iter().map(|i| (*i, None)).collect()
    };
    let chosen: Vec<ItemId> = measured.iter().filter(|(_, m)| !auto || m.is_some_and(|(n, l)| n >= NOISY && l < DARK)).map(|(i, _)| *i).collect();
    let setting = (!off).then_some(Denoise { strength, frames });
    if !chosen.is_empty() {
        s.edit(if off { "Reduce Noise Off" } else { "Reduce Noise" }, |pr, _| {
            for i in &chosen {
                if let Some(ItemKind::Media(m)) = pr.item_mut(*i).map(|it| &mut it.kind) {
                    m.interpret.denoise = setting;
                }
            }
            Ok(())
        })?;
    }
    let r1 = |x: f32| (f64::from(x) * 10.0).round() / 10.0;
    let report: Vec<Value> = measured
        .iter()
        .map(|(i, m)| {
            let on = s.project.item(*i).and_then(|it| it.as_media()).is_some_and(|m| m.interpret.denoise.is_some());
            let mut v = json!({"item": i.0, "denoise": on});
            if let Some((n, l)) = m {
                v["noise"] = json!(r1(*n));
                v["light"] = json!((f64::from(*l) * 1000.0).round() / 1000.0);
            }
            v
        })
        .collect();
    Ok(json!({
        "items": report,
        "changed": chosen.iter().map(|i| i.0).collect::<Vec<_>>(),
        "strength": strength,
        "frames": frames,
        "gpu": filmcraft_render::denoise::on_gpu(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media_test_util::{make_movie, tmp_dir};
    use filmcraft_media::DemoScene;

    #[test]
    fn noise_reduction_is_set_undone_and_used_by_the_source() {
        let dir = tmp_dir("denoise");
        let path = dir.join("night.mp4");
        make_movie(&path, DemoScene::Aurora, 160, 96, 24);
        let mut s = Session::default();
        let r = s.execute("file.import", json!({"paths": [path.to_string_lossy()]})).unwrap();
        let id = r["items"][0].as_u64().unwrap();
        let before = s.source(ItemId(id)).unwrap().video_frame(FrameRequest { time: Tick::ZERO, scale: 1.0 }).unwrap();
        let r = s.execute(CMD, json!({"items": [id], "strength": 0.8, "frames": 1, "measure": true})).unwrap();
        assert_eq!(r["changed"], json!([id]), "{r}");
        assert!(r["items"][0]["noise"].as_f64().is_some(), "{r}");
        let m = |s: &Session| s.project.item(ItemId(id)).and_then(|it| it.as_media()).and_then(|m| m.interpret.denoise);
        assert_eq!(m(&s), Some(Denoise { strength: 0.8, frames: 1 }));
        // the source now merges each frame with its neighbours: same size, a picture still
        let after = s.source(ItemId(id)).unwrap().video_frame(FrameRequest { time: Tick::ZERO, scale: 1.0 }).unwrap();
        assert_eq!((after.width, after.height), (before.width, before.height));
        s.execute("edit.undo", json!({})).unwrap();
        assert_eq!(m(&s), None);
        // auto: a clean, lit picture is left alone
        let r = s.execute(CMD, json!({"items": [id], "auto": true})).unwrap();
        assert_eq!(r["changed"], json!([]), "{r}");
        assert_eq!(m(&s), None);
        s.execute(CMD, json!({"items": [id]})).unwrap();
        s.execute(CMD, json!({"items": [id], "off": true})).unwrap();
        assert_eq!(m(&s), None);
        for bad in [json!({"items": [id], "strength": 2}), json!({"items": [id], "frames": 9}), json!({"items": [id], "frames": "x"}), json!({"items": []})] {
            assert!(s.execute(CMD, bad.clone()).is_err(), "{bad}");
        }
    }
}
