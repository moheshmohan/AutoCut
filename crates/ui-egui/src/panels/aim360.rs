//! Aiming a 360° clip's camera by hand on the Program monitor (View ▸ Aim 360° Camera,
//! `view.aim360`): while it is on, a drag on the picture turns the view of the topmost 360° clip
//! under the playhead — the picture follows the pointer, as in a 360° player — and while the
//! sequence plays every turn is recorded as the camera's yaw and pitch keyframes (`clip.aim360`,
//! one drag one undo step), smoothed when the button is let go.
//!
//! Automation id: `program.aim360` (the picture while aiming).

use egui::{Align2, Color32, Pos2, Rect, Sense, Stroke, vec2};
use filmcraft_project::ClipId;
use filmcraft_project::vr::REFRAME;
use serde_json::json;

use crate::FilmcraftApp;
use crate::theme::Tokens;

/// A drag under way: the clip, where it started and the view then, and the samples so far.
#[derive(Clone, Debug)]
struct Aim {
    clip: ClipId,
    start: Pos2,
    yaw: f64,
    pitch: f64,
    fov: f64,
    samples: Vec<[f64; 3]>,
}

/// The topmost 360° clip seen through a camera under the playhead (on a track that shows), and
/// its yaw, pitch and field of view there.
fn target(app: &FilmcraftApp) -> Option<(ClipId, f64, f64, f64)> {
    let seq = app.session.active_sequence()?;
    let ph = app.session.playhead();
    seq.video_tracks.iter().rev().filter(|t| t.enabled).find_map(|t| {
        t.items.iter().find_map(|it| {
            let e = it.effect(REFRAME).filter(|e| e.enabled && it.enabled && it.range().contains(ph))?;
            let mt = it.effect_time_at(ph);
            let fin = |v: f64, or: f64| if v.is_finite() { v } else { or };
            Some((it.id, fin(e.f64_at("yaw", mt), 0.0), fin(e.f64_at("pitch", mt), 0.0), fin(e.f64_at("fov", mt), 90.0).clamp(10.0, 170.0)))
        })
    })
}

/// The overlay while aiming is on: `true` when it owns the picture (the other overlays then stay
/// off it).
pub fn monitor_overlay(app: &mut FilmcraftApp, ui: &mut egui::Ui, pic: Rect) -> bool {
    if !app.ui.program.aim360 {
        return false;
    }
    let id = egui::Id::new("program-aim360");
    let resp = ui.interact(pic, id, Sense::drag());
    app.auto.add("program.aim360", pic, tl!("Drag to aim the 360° camera"));
    let t = app.tokens;
    let Some((clip, yaw, pitch, fov)) = target(app) else {
        ui.painter().text(pic.center(), Align2::CENTER_CENTER, tl!("No 360° clip under the playhead"), Tokens::ui(12.0), t.text_dim);
        return true;
    };
    // what a drag does, and the middle of the view
    ui.painter().text(pic.min + vec2(8.0, 6.0), Align2::LEFT_TOP, tl!("Drag to aim the 360° camera"), Tokens::ui(11.0), Color32::from_white_alpha(200));
    let c = pic.center();
    let stroke = Stroke::new(1.5, Color32::from_white_alpha(170));
    ui.painter().line_segment([c - vec2(10.0, 0.0), c + vec2(10.0, 0.0)], stroke);
    ui.painter().line_segment([c - vec2(0.0, 10.0), c + vec2(0.0, 10.0)], stroke);
    let key = id.with("drag");
    if resp.drag_started() {
        let start = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos()).unwrap_or(c);
        ui.data_mut(|d| d.insert_temp(key, Aim { clip, start, yaw, pitch, fov, samples: Vec::new() }));
    }
    let aim: Option<Aim> = ui.data(|d| d.get_temp(key));
    let Some(mut a) = aim else { return true };
    let now = app.session.playhead().seconds();
    if resp.dragged() {
        let p = resp.interact_pointer_pos().unwrap_or(a.start);
        // the picture follows the pointer: a drag of the picture's width turns by its field of view
        let w = pic.width().max(1.0) as f64;
        let deg = a.fov / w;
        let y = a.yaw - f64::from(p.x - a.start.x) * deg;
        let pi = (a.pitch + f64::from(p.y - a.start.y) * deg).clamp(-90.0, 90.0);
        let moved = a.samples.last().is_none_or(|l| l[0] != now || (l[1] - y).abs() > 1e-3 || (l[2] - pi).abs() > 1e-3);
        if moved {
            a.samples.push([now, y, pi]);
            let first = a.samples.len() == 1;
            let r = app.session.execute("clip.aim360", json!({"clip": a.clip.0, "samples": a.samples, "smooth": 0, "merge": true, "begin": first}));
            if let Err(e) = r {
                app.ui.status = e.to_string();
            }
        }
        ui.data_mut(|d| d.insert_temp(key, a.clone()));
    }
    if resp.drag_stopped() {
        if a.samples.len() > 2 {
            let r =
                app.session.execute("clip.aim360", json!({"clip": a.clip.0, "samples": a.samples, "smooth": filmcraft_engine::aim360::SMOOTH, "merge": true}));
            if let Err(e) = r {
                app.ui.status = e.to_string();
            }
        }
        ui.data_mut(|d| d.remove::<Aim>(key));
    }
    true
}
