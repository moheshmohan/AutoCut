//! Motion tracking on the Program monitor: the Track tool (CapCut-style).
//!
//! The Program monitor's transport has a Track Object button (`track.tool`). With it on, dragging
//! on the picture draws a box around the object at the playhead; Track backward / both ways /
//! forward runs `track.create` on the clip under it (`subject` or `points`, picked on the tool's
//! bar) as a background job, its progress shown on the bar and on the clip in the timeline. The
//! track's path is drawn on the picture (dim where the object was lost) with the object's box at
//! the playhead; Attach adds what follows it — a text callout, the selected clip as a sticker, a
//! blur, a mosaic, a spotlight (`track.add`), an effect's point or a mask (`track.attach`). Escape
//! or Done closes the tool.
//!
//! Automation ids: `program.transport.track.tool` (the button), `program.track.picture` (where the
//! box is drawn), `program.track.box`, `program.track.method.subject`, `program.track.method.points`,
//! `program.track.backward`, `program.track.both`, `program.track.forward`, `program.track.progress`,
//! `program.track.cancel`, `program.track.path`, `program.track.object`, `program.track.select`
//! (the shown track) and its rows `program.track.select.<id>`, `program.track.attach` and its rows
//! `program.track.attach.{text,sticker,blur,mosaic,spotlight}`,
//! `program.track.attach.point.<effect>.<param>`, `program.track.attach.mask.<effect>.<mask>`,
//! `program.track.done`; in the timeline `timeline.clip.<id>.tracking` while a job tracks the clip.

use egui::{Color32, Pos2, Rect, Sense, Stroke, StrokeKind, pos2, vec2};
use filmcraft_geom::{Affine, Vec2};
use filmcraft_project::{ClipId, MotionTrackId, ParamKind, TrackKind};
use serde_json::{Value, json};

use crate::FilmcraftApp;
use crate::state::TrackToolState;

/// The colour of the box and the path.
const TRACK_COLOR: Color32 = Color32::from_rgb(0xff, 0xc8, 0x2e);
const LOST_COLOR: Color32 = Color32::from_rgb(0xe0, 0x50, 0x50);

/// Whether the clip shows a picture to track.
fn trackable(app: &FilmcraftApp, it: &filmcraft_project::TrackItem) -> bool {
    match app.session.project.item(it.item).map(|i| &i.kind) {
        Some(filmcraft_project::ItemKind::Media(m)) => m.info.has_video(),
        Some(filmcraft_project::ItemKind::Subclip { .. }) => app.session.project.resolve_media(it.item).is_some_and(|(_, m, _)| m.info.has_video()),
        _ => false,
    }
}

/// The clip the tool tracks: the first selected video clip with a picture, else the topmost one
/// under the playhead.
fn pick_clip(app: &FilmcraftApp) -> Option<ClipId> {
    let q = app.session.active_sequence()?;
    let ph = app.session.playhead();
    let selected = app
        .session
        .state
        .selection
        .iter()
        .filter_map(|c| q.find_item(*c))
        .find(|(t, it)| q.track(*t).is_some_and(|t| t.kind == TrackKind::Video) && trackable(app, it));
    if let Some((_, it)) = selected {
        return Some(it.id);
    }
    q.video_tracks.iter().rev().filter_map(|t| t.item_at(ph)).find(|it| trackable(app, it)).map(|it| it.id)
}

/// The Track Object button: open the tool on the clip to track, or close it.
pub fn toggle(app: &mut FilmcraftApp) -> Result<(), String> {
    if app.ui.track_tool.take().is_some() {
        return Ok(());
    }
    let clip = pick_clip(app).ok_or_else(|| tl!("Select a video clip to track").to_string())?;
    let track = app.session.project.motion_tracks.values().filter(|t| t.clip == clip).map(|t| t.id.0).max();
    app.ui.track_tool = Some(TrackToolState { clip: clip.0, rect: None, method: "subject".into(), job: None, track });
    Ok(())
}

/// The progress (0..1) of a tracking job on `clip`, drawn as a bar along the bottom of its body
/// in the timeline.
pub fn clip_progress(app: &mut FilmcraftApp, painter: &egui::Painter, body: Rect, clip: ClipId) {
    let jobs = app.session.motion_tracks.jobs_on(clip);
    let Some(job) = jobs.first().and_then(|id| app.session.jobs.iter().find(|j| j.id == *id)) else { return };
    let f = job.progress.fraction().clamp(0.0, 1.0);
    let bar = Rect::from_min_max(pos2(body.min.x, body.max.y - 4.0), body.max);
    painter.rect_filled(bar, 0.0, Color32::from_black_alpha(140));
    painter.rect_filled(Rect::from_min_max(bar.min, pos2(bar.min.x + bar.width() * f, bar.max.y)), 0.0, TRACK_COLOR);
    app.auto.add(&format!("timeline.clip.{}.tracking", clip.0), bar, &tlf!("Tracking… {n}%", n = format!("{:.0}", f * 100.0)));
}

fn to_screen(pic: Rect, frame: (u32, u32)) -> Affine {
    Affine::translate(f64::from(pic.min.x), f64::from(pic.min.y))
        .then_apply(&Affine::scale(f64::from(pic.width()) / f64::from(frame.0.max(1)), f64::from(pic.height()) / f64::from(frame.1.max(1))))
}

fn sp(m: &Affine, v: Vec2) -> Pos2 {
    let q = m.apply(v);
    pos2(q.x as f32, q.y as f32)
}

fn run(app: &mut FilmcraftApp, cmd: &str, p: Value) -> Option<Value> {
    match app.session.execute(cmd, p) {
        Ok(v) => {
            app.ui.status.clear();
            Some(v)
        }
        Err(e) => {
            app.ui.status = e.to_string();
            None
        }
    }
}

/// The tool on the Program monitor picture `pic` of a `frame`-sized sequence.
pub fn monitor_overlay(app: &mut FilmcraftApp, ui: &mut egui::Ui, pic: Rect, frame: (u32, u32)) {
    let Some(mut st) = app.ui.track_tool.clone() else { return };
    let clip = ClipId(st.clip);
    let Some(it) = app.session.active_sequence().and_then(|q| q.find_item(clip)).map(|(_, i)| i.clone()) else {
        app.ui.track_tool = None;
        return;
    };
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        app.ui.track_tool = None;
        return;
    }
    let ph = app.session.playhead();
    let m = to_screen(pic, frame);
    let painter = ui.painter().with_clip_rect(pic);
    let (fw, fh) = (f64::from(frame.0.max(1)), f64::from(frame.1.max(1)));
    // ---- a finished job: show its track
    if let Some(job) = st.job {
        let running = app.session.jobs.iter().find(|j| j.id == job).is_some_and(|j| !j.progress.finished.load(std::sync::atomic::Ordering::Relaxed));
        if !running && let Some(o) = filmcraft_engine::motion_track::outcome_json(&app.session, job) {
            st.job = None;
            match (o.get("id").and_then(Value::as_u64), o.get("error").and_then(Value::as_str)) {
                (Some(id), _) => {
                    st.track = Some(id);
                    st.rect = None;
                }
                (None, Some(e)) => app.ui.status = e.to_string(),
                _ => {}
            }
        } else if !running && app.session.jobs.iter().all(|j| j.id != job) {
            st.job = None;
        }
    }
    // ---- drawing the box
    let draw = ui.interact(pic, egui::Id::new("track-tool-draw"), Sense::click_and_drag());
    app.auto.add("program.track.picture", pic, tl!("Draw a box around the object to track"));
    if draw.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }
    let start_id = egui::Id::new("track-tool-start");
    if draw.drag_started_by(egui::PointerButton::Primary)
        && let Some(p) = ui.input(|i| i.pointer.press_origin()).or(draw.interact_pointer_pos())
    {
        ui.data_mut(|d| d.insert_temp(start_id, p));
    }
    if draw.dragged_by(egui::PointerButton::Primary)
        && let (Some(a), Some(b)) = (ui.data(|d| d.get_temp::<Pos2>(start_id)), draw.interact_pointer_pos())
        && let Some(inv) = m.inverse()
    {
        let (pa, pb) = (inv.apply(Vec2::new(f64::from(a.x), f64::from(a.y))), inv.apply(Vec2::new(f64::from(b.x), f64::from(b.y))));
        let (x0, x1) = ((pa.x.min(pb.x) / fw).clamp(0.0, 1.0), (pa.x.max(pb.x) / fw).clamp(0.0, 1.0));
        let (y0, y1) = ((pa.y.min(pb.y) / fh).clamp(0.0, 1.0), (pa.y.max(pb.y) / fh).clamp(0.0, 1.0));
        if x1 - x0 > 0.004 && y1 - y0 > 0.004 {
            st.rect = Some([x0, y0, x1 - x0, y1 - y0]);
        }
    }
    if let Some([x, y, w, h]) = st.rect {
        let r = Rect::from_two_pos(sp(&m, Vec2::new(x * fw, y * fh)), sp(&m, Vec2::new((x + w) * fw, (y + h) * fh)));
        painter.rect_stroke(r, 0.0, Stroke::new(1.5, TRACK_COLOR), StrokeKind::Middle);
        for c in [r.left_top(), r.right_top(), r.right_bottom(), r.left_bottom()] {
            painter.rect_filled(Rect::from_center_size(c, vec2(5.0, 5.0)), 0.0, TRACK_COLOR);
        }
        app.auto.add("program.track.box", r, tl!("Box to track"));
    }
    // ---- the shown track: its path and the object at the playhead
    let track = st.track.and_then(|id| app.session.project.motion_tracks.get(&MotionTrackId(id)).cloned());
    if st.track.is_some() && track.is_none() {
        st.track = None;
    }
    if let Some(t) = &track {
        let path = filmcraft_engine::motion_track::shown_path(&app.session.project, t);
        let pts: Vec<(Pos2, bool)> = path.iter().map(|(_, o)| (sp(&m, o.pos), o.lost)).collect();
        for w in pts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let c = if a.1 || b.1 { LOST_COLOR.gamma_multiply(0.7) } else { TRACK_COLOR.gamma_multiply(0.85) };
            painter.line_segment([a.0, b.0], Stroke::new(1.5, c));
        }
        if let Some(bounds) = (!pts.is_empty()).then(|| Rect::from_points(&pts.iter().map(|p| p.0).collect::<Vec<_>>())) {
            app.auto.add("program.track.path", bounds.expand(2.0).intersect(pic), &t.name);
        }
        if let Some(o) = filmcraft_engine::motion_track::shown_at(&app.session.project, t, ph) {
            let half = o.size * 0.5;
            let turn = Affine::translate(o.pos.x, o.pos.y).then_apply(&Affine::rotate_deg(o.rot));
            let q: Vec<Pos2> = [Vec2::new(-half.x, -half.y), Vec2::new(half.x, -half.y), Vec2::new(half.x, half.y), Vec2::new(-half.x, half.y)]
                .into_iter()
                .map(|v| sp(&m, turn.apply(v)))
                .collect();
            let col = if o.lost { LOST_COLOR } else { TRACK_COLOR };
            painter.add(egui::Shape::closed_line(q.clone(), Stroke::new(2.0, col)));
            let c = sp(&m, o.pos);
            painter.circle_filled(c, 3.5, col);
            app.auto.add("program.track.object", Rect::from_points(&q), if o.lost { tl!("Object lost here") } else { tl!("Tracked object") });
        }
    }
    // ---- the bars: the method and the tracking; the track shown, Attach and Done below
    let width = (pic.width() - 12.0).max(40.0);
    let bar = Rect::from_min_size(pic.min + vec2(6.0, 6.0), vec2(width, 26.0));
    let bar2 = bar.translate(vec2(0.0, 30.0));
    painter.rect_filled(bar, 4.0, Color32::from_black_alpha(170));
    let running = st.job.and_then(|j| app.session.jobs.iter().find(|x| x.id == j)).map(|j| j.progress.fraction());
    let mut close = false;
    let mut actions: Vec<(String, Value)> = Vec::new();
    ui.scope_builder(egui::UiBuilder::new().max_rect(bar.shrink2(vec2(4.0, 2.0))), |ui| {
        ui.horizontal_centered(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for (key, label) in [("subject", tl!("Subject")), ("points", tl!("Points"))] {
                let r = ui.selectable_label(st.method == key, label);
                app.auto.add_response(&format!("program.track.method.{key}"), &r, label);
                if r.clicked() {
                    st.method = key.into();
                }
            }
            ui.separator();
            let can_track = st.rect.is_some() && running.is_none();
            for (key, icon, label) in
                [("backward", "◀", tl!("Track backward")), ("both", "◀ ▶", tl!("Track both ways")), ("forward", "▶", tl!("Track forward"))]
            {
                let r = ui.add_enabled(can_track, egui::Button::new(icon).min_size(vec2(28.0, 0.0))).on_hover_text(label);
                r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, can_track, label));
                app.auto.add_response(&format!("program.track.{key}"), &r, label);
                if r.clicked()
                    && let Some(b) = st.rect
                {
                    actions.push((
                        "track.create".into(),
                        json!({"clip": clip.0, "box": b, "space": "frame", "at": ph.seconds(), "direction": key, "method": st.method}),
                    ));
                }
            }
            if let Some(f) = running {
                let label = tlf!("Tracking… {n}%", n = format!("{:.0}", f * 100.0));
                let r = ui.add(egui::ProgressBar::new(f).desired_width(110.0).text(label.clone()));
                app.auto.add_response("program.track.progress", &r, &label);
                let c = ui.small_button(tl!("Cancel"));
                app.auto.add_response("program.track.cancel", &c, tl!("Cancel"));
                if c.clicked()
                    && let Some(j) = st.job
                {
                    actions.push(("jobs.cancel".into(), json!({"job": j})));
                }
            }
        });
    });
    // the tracks of this clip, to show and attach to
    let tracks: Vec<(u64, String)> = app.session.project.motion_tracks.values().filter(|t| t.clip == clip).map(|t| (t.id.0, t.name.clone())).collect();
    painter.rect_filled(bar2, 4.0, Color32::from_black_alpha(170));
    ui.scope_builder(egui::UiBuilder::new().max_rect(bar2.shrink2(vec2(4.0, 2.0))), |ui| {
        ui.horizontal_centered(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            if !tracks.is_empty() {
                let shown = st.track.and_then(|id| tracks.iter().find(|t| t.0 == id)).map_or_else(|| tl!("Motion track").to_string(), |t| short(&t.1, 28));
                let r = ui.menu_button(shown, |ui| {
                    for (id, name) in &tracks {
                        let row = ui.selectable_label(st.track == Some(*id), name);
                        app.auto.add_response(&format!("program.track.select.{id}"), &row, name);
                        if row.clicked() {
                            st.track = Some(*id);
                            ui.close();
                        }
                    }
                });
                app.auto.add_response("program.track.select", &r.response, tl!("Motion track"));
            }
            if let Some(t) = &track {
                attach_menu(app, ui, t, &it, &mut actions);
            }
            let d = ui.small_button(tl!("Done"));
            app.auto.add_response("program.track.done", &d, tl!("Done"));
            if d.clicked() {
                close = true;
            }
        });
    });
    if !(ph >= it.start && ph < it.end()) {
        painter.text(
            bar2.left_bottom() + vec2(2.0, 6.0),
            egui::Align2::LEFT_TOP,
            tl!("Move the playhead over the clip to track"),
            egui::FontId::proportional(11.0),
            TRACK_COLOR,
        );
    }
    for (cmd, p) in actions {
        if let Some(v) = run(app, &cmd, p)
            && cmd == "track.create"
        {
            st.job = v.get("job").and_then(Value::as_u64);
        }
    }
    app.ui.track_tool = if close { None } else { Some(st) };
}

/// At most `n` characters of `s` (an ellipsis when cut).
fn short(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_string() } else { format!("{}…", s.chars().take(n.saturating_sub(1)).collect::<String>()) }
}

/// The Attach menu: what can follow `t` (on clip `it`).
fn attach_menu(
    app: &mut FilmcraftApp,
    ui: &mut egui::Ui,
    t: &filmcraft_project::MotionTrack,
    it: &filmcraft_project::TrackItem,
    actions: &mut Vec<(String, Value)>,
) {
    let id = t.id.0;
    let follow = if t.method == filmcraft_project::TrackingMethod::Points { "positionScaleRotation" } else { "positionScale" };
    // a clip selected on another video track: the sticker
    let sticker = app.session.active_sequence().and_then(|q| {
        app.session
            .state
            .selection
            .iter()
            .copied()
            .filter(|c| *c != it.id)
            .find(|c| q.find_item(*c).and_then(|(tr, _)| q.track(tr)).is_some_and(|tr| tr.kind == TrackKind::Video))
    });
    let points: Vec<(usize, String, String)> = it
        .effects
        .iter()
        .enumerate()
        .filter(|(_, e)| !matches!(e.effect.as_str(), "motion" | "transform") && !filmcraft_project::graphic::is_layer(e))
        .flat_map(|(i, e)| {
            let name = e.def().map_or(e.effect.as_str(), |d| d.name).to_string();
            e.def()
                .map(|d| {
                    d.params
                        .iter()
                        .filter(|p| p.kind == ParamKind::Point && p.animatable)
                        .map(|p| (i, p.id.to_string(), format!("{}: {}", crate::i18n::t(&name), crate::i18n::t(p.label))))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        })
        .collect();
    let masks: Vec<(usize, usize, String)> =
        it.effects.iter().enumerate().flat_map(|(i, e)| e.masks.iter().enumerate().map(move |(k, m)| (i, k, m.name.clone()))).collect();
    let r = ui.menu_button(tl!("Attach"), |ui| {
        let mut row = |ui: &mut egui::Ui, key: &str, label: &str, enabled: bool| -> bool {
            let r = ui.add_enabled(enabled, egui::Button::new(label));
            app.auto.add_response(&format!("program.track.attach.{key}"), &r, label);
            r.clicked()
        };
        if row(ui, "text", tl!("Text callout"), true) {
            actions.push(("track.add".into(), json!({"track": id, "what": "label", "text": tl!("Label")})));
            ui.close();
        }
        if row(ui, "sticker", tl!("Selected clip (sticker)"), sticker.is_some()) {
            if let Some(c) = sticker {
                actions.push(("track.attach".into(), json!({"track": id, "target": {"clip": c.0}, "follow": follow, "offset": "keep"})));
            }
            ui.close();
        }
        if row(ui, "blur", tl!("Blur"), true) {
            actions.push(("track.add".into(), json!({"track": id, "what": "blur"})));
            ui.close();
        }
        if row(ui, "mosaic", tl!("Mosaic"), true) {
            actions.push(("track.add".into(), json!({"track": id, "what": "mosaic"})));
            ui.close();
        }
        if row(ui, "spotlight", tl!("Spotlight"), true) {
            actions.push(("track.add".into(), json!({"track": id, "what": "spotlight"})));
            ui.close();
        }
        if !points.is_empty() || !masks.is_empty() {
            ui.separator();
        }
        for (e, p, label) in &points {
            if row(ui, &format!("point.{e}.{p}"), label, true) {
                actions.push(("track.attach".into(), json!({"track": id, "target": {"clip": it.id.0, "effect": e, "param": p}})));
                ui.close();
            }
        }
        for (e, k, name) in &masks {
            if row(ui, &format!("mask.{e}.{k}"), name, true) {
                actions.push(("track.attach".into(), json!({"track": id, "target": {"clip": it.id.0, "effect": e, "mask": k}, "follow": follow})));
                ui.close();
            }
        }
    });
    app.auto.add_response("program.track.attach", &r.response, tl!("Attach"));
}
