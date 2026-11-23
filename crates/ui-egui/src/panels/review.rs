//! The Program monitor's Comment tool (PX8): a review comment pinned to a frame, for an agent to
//! look at and fix (`review.comment`, `review.inbox`, `review.look`, `review.done`; the inbox is
//! `build/review/inbox.jsonl` beside the saved project).
//!
//! The transport's Comment on Frame button turns the tool on (the project must be saved: the
//! inbox lives beside it). With it on, a click on the picture points at a spot and a drag draws a
//! box; the clip under it and the playhead's frame go with the comment. The bar along the top of
//! the picture takes the text; Enter or Save runs `review.comment` with author "user", and the
//! tool stays on for the next one. Moving to another frame drops the box (the text stays).
//! Escape drops the box and the text, or closes the tool when there are none; Done closes it.
//! Previous and Next move the playhead to the open comments before and after it. The open
//! comments of the frame shown are drawn on the picture while the tool is on, and every open
//! comment is marked on the monitor's scrub bar, the tool on or off (read every few seconds).
//!
//! Automation ids: `program.transport.review.tool` (the button), `program.review.picture` (where
//! a click or a drag points), `program.review.box` (the box or point drawn), `program.review.text`,
//! `program.review.save`, `program.review.clear`, `program.review.done`, `program.review.prev`,
//! `program.review.next`, `program.review.comment.<id>` (an open comment on the frame shown) and
//! `program.review.mark.<id>` (its mark on the scrub bar).

use egui::{Color32, Pos2, Rect, Sense, Shape, Stroke, StrokeKind, pos2, vec2};
use filmcraft_geom::{Affine, Vec2};
use filmcraft_project::ItemKind;
use filmcraft_time::Tick;
use serde_json::json;

use crate::FilmcraftApp;
use crate::state::{ReviewMark, ReviewToolState};

/// The colour of the comment being written and of the marks.
const COMMENT_COLOR: Color32 = Color32::from_rgb(0xff, 0xc4, 0x28);
/// Saved comments on the frame shown.
const SAVED_COLOR: Color32 = Color32::from_rgb(0x7f, 0xd1, 0xff);
/// How often the open comments are read again, seconds.
const READ_EVERY: f64 = 3.0;
/// Most comments marked.
const MAX_MARKS: usize = 500;

/// The Comment on Frame button: turn the tool on, or off.
pub fn toggle(app: &mut FilmcraftApp) -> Result<(), String> {
    if app.ui.review_tool.take().is_some() {
        return Ok(());
    }
    if app.session.active_sequence().is_none() {
        return Err(tl!("Open a sequence to comment on").to_string());
    }
    if filmcraft_engine::review::inbox_path(&app.session).is_none() {
        return Err(tl!("Save the project first: review comments are kept in its build folder").to_string());
    }
    app.stop();
    // both tools own the picture
    app.ui.track_tool = None;
    app.ui.review_tool = Some(ReviewToolState::default());
    app.ui.review_marks.read_at = None;
    Ok(())
}

/// Read the open comments again when they are older than [`READ_EVERY`] (or were never read).
/// `now` is the UI clock, seconds.
pub fn refresh(app: &mut FilmcraftApp, now: f64) {
    if app.ui.review_marks.read_at.is_some_and(|t| now >= t && now - t < READ_EVERY) {
        return;
    }
    app.ui.review_marks.read_at = Some(now);
    app.ui.review_marks.comments = filmcraft_engine::review::open_comments(&app.session)
        .into_iter()
        .take(MAX_MARKS)
        .map(|c| ReviewMark { id: c.id, sequence: c.sequence, time: c.time, area: c.area, text: c.text })
        .collect();
}

/// The open comments of the active sequence, by time.
fn marks(app: &FilmcraftApp) -> Vec<ReviewMark> {
    let Some(seq) = app.session.state.active_sequence else { return Vec::new() };
    let mut v: Vec<ReviewMark> = app.ui.review_marks.comments.iter().filter(|m| m.sequence == seq.0).cloned().collect();
    v.sort_by_key(|m| m.time);
    v
}

/// At most `n` characters of `s` on one line.
fn short(s: &str, n: usize) -> String {
    let line = s.lines().next().unwrap_or_default();
    if line.chars().count() <= n && !s.contains('\n') { line.to_string() } else { format!("{}…", line.chars().take(n.saturating_sub(1)).collect::<String>()) }
}

/// The open comments as marks on the Program monitor's scrub bar `bar`, which shows `shown`.
pub fn scrub_marks(app: &mut FilmcraftApp, ui: &egui::Ui, bar: Rect, shown: (Tick, Tick)) {
    let dur = (shown.1 - shown.0).0.max(1) as f64;
    let p = ui.painter();
    for m in marks(app).into_iter().filter(|m| m.time >= shown.0.0 && m.time <= shown.1.0) {
        let x = bar.min.x + (((m.time - shown.0.0) as f64 / dur) as f32).clamp(0.0, 1.0) * bar.width();
        // a little square bubble with a tail, at the top of the bar
        let body = Rect::from_min_size(pos2(x - 4.0, bar.min.y + 1.0), vec2(8.0, 5.5));
        p.rect_filled(body, 1.0, COMMENT_COLOR);
        p.add(Shape::convex_polygon(vec![pos2(x - 2.5, body.max.y), pos2(x + 0.5, body.max.y), pos2(x - 2.5, body.max.y + 2.5)], COMMENT_COLOR, Stroke::NONE));
        app.auto.add(&format!("program.review.mark.{}", m.id), body.expand(2.0), &format!("{}: {}", m.id, short(&m.text, 60)));
    }
}

/// The clip a point on the picture is on: a graphic's layer under it, else the top-most picture
/// under it (a graphic fills the frame, so only its layers count).
fn clip_at(app: &FilmcraftApp, pic: Rect, frame: (u32, u32), p: Pos2) -> Option<u64> {
    if let Some(v) = crate::panels::graphics::visible_layers(app, pic, frame).iter().find(|v| v.hit(p)) {
        return Some(v.clip.0);
    }
    let seq = app.session.active_sequence()?;
    let t = app.session.playhead();
    let project = &app.session.project;
    let view = crate::panels::graphics::frame_to_screen(pic, frame);
    seq.video_tracks
        .iter()
        .rev()
        .filter(|tr| tr.enabled)
        .filter_map(|tr| tr.item_at(t))
        .filter(|it| it.enabled && !matches!(project.item(it.item).map(|i| &i.kind), Some(ItemKind::Graphic { .. })))
        .find_map(|it| {
            let size = filmcraft_render::source_size(project, it.item).unwrap_or(frame);
            let m = view.then_apply(&filmcraft_render::motion_matrix(seq, it, size, filmcraft_render::source_par(project, it.item), it.source_time_at(t)));
            let q = m.inverse()?.apply(Vec2::new(f64::from(p.x), f64::from(p.y)));
            (q.x >= 0.0 && q.y >= 0.0 && q.x <= f64::from(size.0) && q.y <= f64::from(size.1)).then_some(it.id.0)
        })
}

fn sp(m: &Affine, x: f64, y: f64) -> Pos2 {
    let q = m.apply(Vec2::new(x, y));
    pos2(q.x as f32, q.y as f32)
}

/// A box (or a point, when it has no size) in sequence pixels, drawn on the picture: its screen
/// rect.
fn draw_area(painter: &egui::Painter, m: &Affine, b: [f64; 4], color: Color32, width: f32) -> Rect {
    let r = Rect::from_two_pos(sp(m, b[0], b[1]), sp(m, b[0] + b[2], b[1] + b[3]));
    if r.width() < 2.0 && r.height() < 2.0 {
        let c = r.center();
        painter.circle_stroke(c, 9.0, Stroke::new(width + 2.0, Color32::from_black_alpha(160)));
        painter.circle_stroke(c, 9.0, Stroke::new(width, color));
        painter.line_segment([c - vec2(5.0, 0.0), c + vec2(5.0, 0.0)], Stroke::new(1.5, color));
        painter.line_segment([c - vec2(0.0, 5.0), c + vec2(0.0, 5.0)], Stroke::new(1.5, color));
        return Rect::from_center_size(c, vec2(20.0, 20.0));
    }
    painter.rect_stroke(r, 0.0, Stroke::new(width + 2.0, Color32::from_black_alpha(160)), StrokeKind::Middle);
    painter.rect_stroke(r, 0.0, Stroke::new(width, color), StrokeKind::Middle);
    r
}

enum Action {
    Save,
    Seek(i64),
}

/// The tool on the Program monitor picture `pic` of a `frame`-sized sequence.
pub fn monitor_overlay(app: &mut FilmcraftApp, ui: &mut egui::Ui, pic: Rect, frame: (u32, u32)) {
    let Some(mut st) = app.ui.review_tool.clone() else { return };
    let Some(rate) = app.session.active_sequence().map(|q| q.settings.frame_rate.sane()) else {
        app.ui.review_tool = None;
        return;
    };
    let ph = app.session.playhead();
    let text_id = egui::Id::new("review-tool-text");
    if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
        if st.area.is_none() && st.text.trim().is_empty() {
            app.ui.review_tool = None;
            return;
        }
        st = ReviewToolState::default();
    }
    // a box drawn on another frame is not this frame's
    if st.time.is_some_and(|t| rate.frame_at(Tick(t)) != rate.frame_at(ph)) {
        st.area = None;
        st.clip = None;
        st.time = None;
    }
    let m = crate::panels::graphics::frame_to_screen(pic, frame);
    let painter = ui.painter().with_clip_rect(pic);
    let (fw, fh) = (f64::from(frame.0.max(1)), f64::from(frame.1.max(1)));
    let to_seq = |p: Pos2| -> Option<(f64, f64)> {
        let v = m.inverse()?.apply(Vec2::new(f64::from(p.x), f64::from(p.y)));
        Some(((v.x.clamp(0.0, fw) * 10.0).round() / 10.0, (v.y.clamp(0.0, fh) * 10.0).round() / 10.0))
    };
    // ---- pointing: a click is a point, a drag a box
    let draw = ui.interact(pic, egui::Id::new("review-tool-draw"), Sense::click_and_drag());
    app.auto.add("program.review.picture", pic, tl!("Click or drag a box on the picture to comment on it"));
    if draw.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::Crosshair);
    }
    let start_id = egui::Id::new("review-tool-start");
    if draw.drag_started_by(egui::PointerButton::Primary)
        && let Some(p) = ui.input(|i| i.pointer.press_origin()).or(draw.interact_pointer_pos())
    {
        ui.data_mut(|d| d.insert_temp(start_id, p));
    }
    if draw.dragged_by(egui::PointerButton::Primary)
        && let (Some(a), Some(b)) = (ui.data(|d| d.get_temp::<Pos2>(start_id)), draw.interact_pointer_pos())
        && let (Some(pa), Some(pb)) = (to_seq(a), to_seq(b))
    {
        let (x0, y0) = (pa.0.min(pb.0), pa.1.min(pb.1));
        let (w, h) = ((pa.0 - pb.0).abs(), (pa.1 - pb.1).abs());
        if w >= 2.0 || h >= 2.0 {
            st.area = Some([x0, y0, w, h]);
            st.time = Some(ph.0);
            st.clip = clip_at(app, pic, frame, pos2((a.x + b.x) / 2.0, (a.y + b.y) / 2.0));
        }
    }
    if draw.drag_stopped() && st.area.is_some() {
        st.focus = true;
    }
    if draw.clicked()
        && let Some(p) = draw.interact_pointer_pos()
        && let Some((x, y)) = to_seq(p)
    {
        st.area = Some([x, y, 0.0, 0.0]);
        st.time = Some(ph.0);
        st.clip = clip_at(app, pic, frame, p);
        st.focus = true;
    }
    // ---- the open comments of this frame, then the one being written
    let here: Vec<ReviewMark> = marks(app).into_iter().filter(|c| rate.frame_at(Tick(c.time)) == rate.frame_at(ph)).collect();
    let mut stack = pic.left_bottom() + vec2(8.0, -8.0);
    for c in &here {
        let label = format!("{}  {}", c.id, short(&c.text, 48));
        let r = match c.area {
            Some(b) => {
                let r = draw_area(&painter, &m, b, SAVED_COLOR, 1.5);
                painter.text(r.left_top() + vec2(0.0, -3.0), egui::Align2::LEFT_BOTTOM, &label, egui::FontId::proportional(11.0), SAVED_COLOR);
                r
            }
            None => {
                let g = painter.text(stack, egui::Align2::LEFT_BOTTOM, &label, egui::FontId::proportional(11.0), SAVED_COLOR);
                stack.y -= 15.0;
                g
            }
        };
        app.auto.add(&format!("program.review.comment.{}", c.id), r, &format!("{}: {}", c.id, short(&c.text, 60)));
    }
    if let Some(b) = st.area {
        let r = draw_area(&painter, &m, b, COMMENT_COLOR, 2.0);
        app.auto.add("program.review.box", r, tl!("What the comment points at"));
    }
    // ---- the bar: previous / next, the text, Save, Clear, Done
    let width = (pic.width() - 12.0).max(60.0);
    let bar = Rect::from_min_size(pic.min + vec2(6.0, 6.0), vec2(width, 28.0));
    painter.rect_filled(bar, 4.0, Color32::from_black_alpha(185));
    let all = marks(app);
    let prev = all.iter().rev().find(|c| c.time < ph.0).map(|c| c.time);
    let next = all.iter().find(|c| rate.frame_at(Tick(c.time)) > rate.frame_at(ph)).map(|c| c.time);
    let mut actions = Vec::new();
    let mut close = false;
    ui.scope_builder(egui::UiBuilder::new().max_rect(bar.shrink2(vec2(4.0, 2.0))), |ui| {
        ui.horizontal_centered(|ui| {
            ui.spacing_mut().item_spacing.x = 4.0;
            for (key, glyph, label, to) in [("prev", "◀", tl!("Previous comment"), prev), ("next", "▶", tl!("Next comment"), next)] {
                let r = ui.add_enabled(to.is_some(), egui::Button::new(glyph).min_size(vec2(24.0, 0.0))).on_hover_text(label);
                r.widget_info(|| egui::WidgetInfo::labeled(egui::WidgetType::Button, to.is_some(), label));
                app.auto.add_response(&format!("program.review.{key}"), &r, label);
                if r.clicked()
                    && let Some(t) = to
                {
                    actions.push(Action::Seek(t));
                }
            }
            let room = (ui.available_width() - 160.0).max(80.0);
            let edit = egui::TextEdit::singleline(&mut st.text)
                .id(text_id)
                .hint_text(tl!("Comment on this frame…"))
                .char_limit(filmcraft_engine::review::MAX_TEXT)
                .desired_width(room);
            let r = ui.add(edit);
            app.auto.add_response("program.review.text", &r, tl!("Comment"));
            if std::mem::take(&mut st.focus) {
                r.request_focus();
            }
            let entered = r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            let can = !st.text.trim().is_empty();
            let save = ui.add_enabled(can, egui::Button::new(tl!("Save")));
            app.auto.add_response("program.review.save", &save, tl!("Save"));
            if can && (save.clicked() || entered) {
                actions.push(Action::Save);
            }
            let dirty = st.area.is_some() || !st.text.is_empty();
            let clear = ui.add_enabled(dirty, egui::Button::new(tl!("Clear")).small());
            app.auto.add_response("program.review.clear", &clear, tl!("Clear"));
            if clear.clicked() {
                st = ReviewToolState::default();
            }
            let done = ui.small_button(tl!("Done"));
            app.auto.add_response("program.review.done", &done, tl!("Done"));
            if done.clicked() {
                close = true;
            }
        });
    });
    // what the comment will point at
    let what = match (st.area, st.clip) {
        (None, _) => tl!("Click a spot or drag a box on the picture; or just type to comment on the whole frame").to_string(),
        (Some(b), clip) => {
            let name = clip
                .and_then(|c| app.session.active_sequence()?.find_item(filmcraft_project::ClipId(c)).map(|(_, it)| it.name.clone()))
                .unwrap_or_else(|| tl!("no clip").to_string());
            if b[2] < 1.0 && b[3] < 1.0 {
                tlf!("A point on {name}", name = name)
            } else {
                tlf!("A {w} × {h} box on {name}", w = b[2].round(), h = b[3].round(), name = name)
            }
        }
    };
    let open = all.len();
    let line = if open > 0 { format!("{what} · {}", tlf!("{n} open", n = open)) } else { what };
    painter.text(bar.left_bottom() + vec2(2.0, 6.0), egui::Align2::LEFT_TOP, line, egui::FontId::proportional(11.0), COMMENT_COLOR);
    for a in actions {
        match a {
            Action::Seek(t) => {
                app.stop();
                app.session.set_playhead(Tick(t));
                st.area = None;
                st.clip = None;
                st.time = None;
            }
            Action::Save => save(app, &mut st),
        }
    }
    app.ui.review_tool = if close { None } else { Some(st) };
}

/// `review.comment` with what the tool holds; on success the tool is cleared for the next one.
fn save(app: &mut FilmcraftApp, st: &mut ReviewToolState) {
    let time = st.time.unwrap_or(app.session.playhead().0);
    let mut p = json!({"time": time, "text": st.text, "author": "user"});
    if let Some(b) = st.area {
        p["box"] = json!(b);
    }
    if let Some(c) = st.clip {
        p["clip"] = json!(c);
    }
    match app.session.execute("review.comment", p) {
        Ok(v) => {
            let id = v["comment"]["id"].as_str().unwrap_or_default().to_string();
            app.ui.status = tlf!("Comment {id} saved to the review inbox", id = id);
            *st = ReviewToolState::default();
            app.ui.review_marks.read_at = None;
        }
        Err(e) => app.ui.status = e.to_string(),
    }
}
