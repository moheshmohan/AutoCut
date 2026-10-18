//! Export mode's Destinations (roadmap B5): the rows of the left column — Media File, YouTube,
//! YouTube Short, Instagram Reel, each with a generic glyph (no platform logo) and its connect
//! state — and, for a platform row, its form in the middle column: the account (Connect), the
//! metadata, the file to publish (the last export unless another is given), the pre-flight
//! checklist (`publish.preflight`: each failure with its fix, and Export a Fixed Copy,
//! `publish.fixCopy`), and Publish (`publish.run`, a job with progress and cancel).
//!
//! Everything the form holds is in [`PublishForm`] (`ui.inspect` / `ui.set {"export": {"publish":
//! {…}}}`); every action runs an engine command.
//!
//! Automation ids: `export.destination.<mediaFile|youtube|youtubeShort|instagramReel>` (label
//! "<Label>: <state>"), `export.publish.connect`, `export.publish.channel` (and
//! `….option.<i>`), `export.publish.title`, `export.publish.description`, `export.publish.tags`,
//! `export.publish.privacy`, `export.publish.caption`, `export.publish.captionCount`,
//! `export.publish.shareToFeed`, `export.publish.coverSeconds`, `export.publish.path`,
//! `export.publish.useExport`, `export.publish.check`, `export.publish.summary`,
//! `export.publish.check.<check id>` (label "<status>: <label> — <value>"), `export.publish.fix`,
//! `export.publish.run` (its label is the reason when disabled), `export.publish.status`,
//! `export.publish.open`.

use egui::{Align2, Rect, Sense, pos2, vec2};
use filmcraft_engine::publish::Destination;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::FilmcraftApp;
use crate::icons::Icon;
use crate::theme::Tokens;

/// The platform form's state.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct PublishForm {
    pub title: String,
    pub description: String,
    /// Comma-separated.
    pub tags: String,
    /// `private` | `unlisted` | `public`.
    pub privacy: String,
    /// The YouTube channel (id; empty: the default) and its label.
    pub channel: String,
    pub channel_label: String,
    pub caption: String,
    pub share_to_feed: bool,
    pub cover_seconds: f64,
    /// The file to publish (empty: the last export).
    pub path: String,
    /// The last pre-flight (`publish.preflight`'s answer) and what it was for
    /// (`<destination>|<path>`).
    pub preflight: Option<Value>,
    pub preflight_for: String,
    /// The publish job started here, its destination and file.
    pub job: Option<u64>,
    pub job_for: String,
    /// The fixed-copy job, and the connect job.
    pub fix_job: Option<u64>,
    pub connect_job: Option<u64>,
}

impl Default for PublishForm {
    fn default() -> Self {
        Self {
            title: String::new(),
            description: String::new(),
            tags: String::new(),
            privacy: "private".into(),
            channel: String::new(),
            channel_label: String::new(),
            caption: String::new(),
            share_to_feed: true,
            cover_seconds: 0.0,
            path: String::new(),
            preflight: None,
            preflight_for: String::new(),
            job: None,
            job_for: String::new(),
            fix_job: None,
            connect_job: None,
        }
    }
}

fn icon(d: Destination) -> Icon {
    match d {
        Destination::MediaFile => Icon::DestFile,
        Destination::Youtube => Icon::DestVideo,
        Destination::YoutubeShort => Icon::DestShort,
        Destination::InstagramReel => Icon::DestCamera,
    }
}

/// The destination picked (Media File unless a platform row was).
pub fn current(app: &FilmcraftApp) -> Destination {
    Destination::parse(&app.ui.export.destination).unwrap_or(Destination::MediaFile)
}

/// The rows, from `top` down the left column; returns where they end.
pub fn rows(app: &mut FilmcraftApp, ui: &egui::Ui, rect: Rect, top: f32) -> f32 {
    let t = app.tokens;
    let picked = current(app);
    let mut y = top;
    let mut clicked = None;
    for d in Destination::ALL {
        let r = Rect::from_min_size(pos2(rect.min.x + 8.0, y), vec2(rect.width() - 16.0, 26.0));
        let resp = ui.interact(r, egui::Id::new(("export-destination", d.id())), Sense::click());
        let p = ui.painter();
        if d == picked {
            p.rect_filled(r, 4.0, t.row_selected);
        } else if resp.hovered() {
            p.rect_filled(r, 4.0, t.hover);
        }
        crate::icons::paint(
            p,
            Rect::from_center_size(pos2(r.min.x + 14.0, r.center().y), vec2(16.0, 16.0)),
            icon(d),
            if d == picked { t.icon_active } else { t.icon },
        );
        p.text(pos2(r.min.x + 28.0, r.center().y), Align2::LEFT_CENTER, crate::i18n::t(d.label()), Tokens::ui(12.5), t.text);
        let state = filmcraft_engine::publish::connect_state(&app.session, d);
        let (word, col) = match (state["needed"].as_bool(), state["ready"].as_bool(), state["available"].as_bool()) {
            (Some(false), _, _) => (String::new(), t.text_dim),
            (_, Some(true), _) => (tl!("Connected").to_string(), t.render_green),
            (_, _, Some(false)) => (tl!("Not available").to_string(), t.text_faint),
            _ => (tl!("Not connected").to_string(), t.text_faint),
        };
        if !word.is_empty() {
            p.text(pos2(r.max.x - 8.0, r.center().y), Align2::RIGHT_CENTER, &word, Tokens::ui(11.0), col);
        }
        let label = if word.is_empty() { d.label().to_string() } else { format!("{}: {word}", d.label()) };
        app.auto.add(&format!("export.destination.{}", d.id()), r, &label);
        if resp.clicked() {
            clicked = Some(d);
        }
        y = r.max.y + 2.0;
    }
    if let Some(d) = clicked {
        app.ui.export.destination = d.id().into();
    }
    y
}

/// The file the form publishes: the one given, else the last export.
fn file_of(app: &FilmcraftApp) -> Option<String> {
    let given = app.ui.export.publish.path.trim();
    if !given.is_empty() {
        return Some(given.to_string());
    }
    filmcraft_engine::photos::last_export(&app.session)
}

fn job_of(app: &FilmcraftApp, id: Option<u64>) -> Option<filmcraft_engine::Job> {
    id.and_then(|id| app.session.jobs.iter().find(|j| j.id == id)).cloned()
}

fn running(j: &Option<filmcraft_engine::Job>) -> bool {
    j.as_ref().is_some_and(|j| !j.progress.finished.load(std::sync::atomic::Ordering::Relaxed))
}

fn error_of(j: &filmcraft_engine::Job) -> Option<String> {
    j.progress.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
}

/// The params `publish.run` gets for the form.
pub fn run_params(form: &PublishForm, d: Destination, path: &str) -> Value {
    let mut p = json!({"destination": d.id(), "path": path});
    match d {
        Destination::InstagramReel => {
            if !form.caption.trim().is_empty() {
                p["caption"] = json!(form.caption);
            }
            p["shareToFeed"] = json!(form.share_to_feed);
            if form.cover_seconds > 0.0 {
                p["coverSeconds"] = json!(form.cover_seconds);
            }
        }
        _ => {
            if !form.title.trim().is_empty() {
                p["title"] = json!(form.title.trim());
            }
            if !form.description.trim().is_empty() {
                p["description"] = json!(form.description);
            }
            let tags: Vec<&str> = form.tags.split(',').map(str::trim).filter(|t| !t.is_empty()).collect();
            if !tags.is_empty() {
                p["tags"] = json!(tags);
            }
            p["privacy"] = json!(if form.privacy.is_empty() { "private" } else { form.privacy.as_str() });
            if !form.channel.is_empty() {
                p["channel"] = json!(form.channel);
            }
        }
    }
    p
}

/// The form of a platform destination, in the middle column.
pub fn form(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    use std::sync::atomic::Ordering;
    let t = app.tokens;
    let d = current(app);
    let state = filmcraft_engine::publish::connect_state(&app.session, d);
    let ready = state["ready"].as_bool() == Some(true);
    let file = file_of(app);
    let key = format!("{}|{}", d.id(), file.clone().unwrap_or_default());
    let fresh = app.ui.export.publish.preflight.as_ref().filter(|_| app.ui.export.publish.preflight_for == key).cloned();
    let job = job_of(app, app.ui.export.publish.job);
    let fix_job = job_of(app, app.ui.export.publish.fix_job);
    let connect_job = job_of(app, app.ui.export.publish.connect_job);
    let (busy, fixing, connecting) = (running(&job), running(&fix_job), running(&connect_job));
    // a finished fixed copy becomes the file, with its own pre-flight
    if let Some(j) = &fix_job
        && !fixing
        && error_of(j).is_none()
    {
        let out = filmcraft_engine::publish::outcome_json(&app.session, j.id);
        if let Some(path) = out["path"].as_str() {
            app.ui.export.publish.path = path.to_string();
            app.ui.export.publish.preflight_for = format!("{}|{path}", d.id());
            app.ui.export.publish.preflight = Some(out["preflight"].clone()).filter(|v| !v.is_null());
            app.ui.export.publish.fix_job = None;
        }
    }
    let mut action: Option<(&'static str, Value)> = None;
    let mut run_check = false;
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(rect.shrink2(vec2(14.0, 10.0))).id_salt("export-destination-form"));
    egui::ScrollArea::vertical().id_salt("export-destination-scroll").auto_shrink([false, false]).show(&mut child, |ui| {
        ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::hover());
            crate::icons::paint(ui.painter(), r, icon(d), t.icon_active);
            ui.label(egui::RichText::new(crate::i18n::t(d.label())).strong().size(14.0));
        });
        ui.add_space(6.0);
        // the account
        ui.horizontal_wrapped(|ui| {
            let account = if ready { tl!("Connected").to_string() } else { state["why"].as_str().unwrap_or_default().to_string() };
            let r = ui.add(
                egui::Label::new(egui::RichText::new(&account).size(11.5).color(if ready { t.render_green } else { t.text_dim }))
                    .wrap_mode(egui::TextWrapMode::Wrap),
            );
            app.auto.add("export.publish.account", r.rect, &account);
        });
        // YouTube connects one channel per sign-in (another can always be added); Instagram one account
        let youtube = state["connect"].as_str() == Some("youtube.connect");
        if let Some(cmd) = state["connect"].as_str().filter(|_| youtube || !ready) {
            let can = state["canConnect"].as_bool() == Some(true) && !connecting;
            let why = state["connectWhy"].as_str().unwrap_or_default().to_string();
            let b = ui.add_enabled(can, egui::Button::new(if youtube { tl!("Connect a channel") } else { tl!("Connect") }));
            let b = if why.is_empty() { b } else { b.on_disabled_hover_text(&why) };
            app.auto.add_response("export.publish.connect", &b, if why.is_empty() { "Connect" } else { &why });
            if b.clicked() {
                action = Some(if cmd == "youtube.connect" { ("youtube.connect", json!({})) } else { ("instagram.connect", json!({})) });
            }
            if let Some(c) = &connect_job {
                let s = if connecting {
                    c.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone()
                } else {
                    error_of(c).unwrap_or_default()
                };
                if !s.is_empty() {
                    ui.label(egui::RichText::new(s).size(11.0).color(if connecting { t.accent } else { t.danger }));
                }
            }
        }
        ui.add_space(8.0);
        // the metadata
        let form = &mut app.ui.export.publish;
        match d {
            Destination::InstagramReel => {
                ui.label(egui::RichText::new(tl!("Caption")).color(t.text_dim).size(12.0));
                let r = ui.add(egui::TextEdit::multiline(&mut form.caption).desired_rows(4).desired_width(f32::INFINITY));
                app.auto.add("export.publish.caption", r.rect, "Caption");
                let (tags, mentions) = filmcraft_engine::instagram::caption_counts(&form.caption);
                let n = form.caption.chars().count();
                let over = n > filmcraft_engine::instagram::MAX_CAPTION_CHARS
                    || tags > filmcraft_engine::instagram::MAX_HASHTAGS
                    || mentions > filmcraft_engine::instagram::MAX_MENTIONS;
                let count = format!("{n}/2200 · #{tags}/30 · @{mentions}/20");
                let r = ui.label(egui::RichText::new(&count).size(11.0).color(if over { t.danger } else { t.text_faint }));
                app.auto.add("export.publish.captionCount", r.rect, &count);
                let r = ui.checkbox(&mut form.share_to_feed, tl!("Also show it in the profile grid"));
                app.auto.add("export.publish.shareToFeed", r.rect, "Share to feed");
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(tl!("Cover frame at")).color(t.text_dim).size(12.0));
                    let r = ui.add(egui::DragValue::new(&mut form.cover_seconds).range(0.0..=900.0).speed(0.1).suffix(" s"));
                    app.auto.add("export.publish.coverSeconds", r.rect, "Cover frame at");
                });
            }
            _ => {
                ui.label(egui::RichText::new(tl!("Title")).color(t.text_dim).size(12.0));
                let r = ui.add(egui::TextEdit::singleline(&mut form.title).hint_text(tl!("The file's name")).desired_width(f32::INFINITY));
                app.auto.add("export.publish.title", r.rect, "Title");
                ui.label(egui::RichText::new(tl!("Description")).color(t.text_dim).size(12.0));
                let r = ui.add(egui::TextEdit::multiline(&mut form.description).desired_rows(3).desired_width(f32::INFINITY));
                app.auto.add("export.publish.description", r.rect, "Description");
                ui.label(egui::RichText::new(tl!("Tags")).color(t.text_dim).size(12.0));
                let r = ui.add(egui::TextEdit::singleline(&mut form.tags).hint_text("robots, expo").desired_width(f32::INFINITY));
                app.auto.add("export.publish.tags", r.rect, "Tags");
                ui.horizontal(|ui| {
                    ui.label(egui::RichText::new(tl!("Privacy")).color(t.text_dim).size(12.0));
                    let r = egui::ComboBox::from_id_salt("export.publish.privacy").selected_text(crate::i18n::t(&form.privacy)).show_ui(ui, |ui| {
                        for p in ["private", "unlisted", "public"] {
                            ui.selectable_value(&mut form.privacy, p.to_string(), crate::i18n::t(p));
                        }
                    });
                    app.auto.add("export.publish.privacy", r.response.rect, &form.privacy);
                });
                // the channel: the connected ones, read only while the list is open
                if let Some(publisher) = app.session.youtube.publisher.clone() {
                    let selected = if form.channel.is_empty() { tl!("Default channel").to_string() } else { form.channel_label.clone() };
                    let mut picked: Option<(String, String)> = None;
                    let r = egui::ComboBox::from_id_salt("export.publish.channel").selected_text(&selected).width(220.0).show_ui(ui, |ui| {
                        if ui.selectable_label(form.channel.is_empty(), tl!("Default channel")).clicked() {
                            picked = Some((String::new(), String::new()));
                        }
                        let list = publisher.channels().unwrap_or(Value::Null);
                        for (i, c) in list["channels"].as_array().into_iter().flatten().enumerate() {
                            let id = c["id"].as_str().unwrap_or_default().to_string();
                            let label = c["title"].as_str().unwrap_or_default().to_string();
                            let resp = ui.selectable_label(form.channel == id, &label);
                            app.auto.add(&format!("export.publish.channel.option.{i}"), resp.rect, &label);
                            if resp.clicked() {
                                picked = Some((id, label));
                            }
                        }
                    });
                    app.auto.add("export.publish.channel", r.response.rect, &selected);
                    if let Some((id, label)) = picked {
                        form.channel = id;
                        form.channel_label = label;
                    }
                }
            }
        }
        ui.add_space(8.0);
        // the file
        ui.label(egui::RichText::new(tl!("File to publish")).color(t.text_dim).size(12.0));
        let last = filmcraft_engine::photos::last_export(&app.session);
        let form = &mut app.ui.export.publish;
        let r = ui.add(
            egui::TextEdit::singleline(&mut form.path)
                .hint_text(last.clone().unwrap_or_else(|| tl!("Export first, or type the path of a video").to_string()))
                .desired_width(f32::INFINITY),
        );
        app.auto.add("export.publish.path", r.rect, file.as_deref().unwrap_or_default());
        if let Some(last) = &last
            && !form.path.is_empty()
            && form.path != *last
        {
            let b = ui.small_button(tl!("Use the last export"));
            app.auto.add_response("export.publish.useExport", &b, last);
            if b.clicked() {
                form.path.clear();
            }
        }
        ui.add_space(8.0);
        // the pre-flight
        ui.horizontal(|ui| {
            let b = ui.add_enabled(file.is_some() && !fixing, egui::Button::new(tl!("Check")));
            app.auto.add_response("export.publish.check", &b, "Check");
            run_check = b.clicked();
            // offered only when a copy can mend every failure (not a length or a shape)
            if let Some(pre) = &fresh
                && pre["fixable"].is_string()
                && pre["unfixable"].as_array().is_none_or(Vec::is_empty)
            {
                let b = ui.add_enabled(!fixing && !busy, egui::Button::new(tl!("Export a Fixed Copy")));
                app.auto.add_response("export.publish.fix", &b, "Export a Fixed Copy");
                if b.clicked()
                    && let Some(path) = &file
                {
                    action = Some(("publish.fixCopy", json!({"destination": d.id(), "path": path})));
                }
            }
        });
        if let Some(j) = &fix_job {
            if fixing {
                ui.add(egui::ProgressBar::new(j.progress.fraction().clamp(0.0, 1.0)).desired_height(6.0));
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
            }
            let s = if fixing { j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone() } else { error_of(j).unwrap_or_default() };
            if !s.is_empty() {
                ui.label(egui::RichText::new(s).size(11.0).color(if fixing { t.accent } else { t.danger }));
            }
        }
        if let Some(pre) = &fresh {
            checklist(app, ui, pre);
        }
        ui.add_space(10.0);
        // publish
        let why_not = if d.publish_command().is_none() {
            Some("the Media File is the export itself".to_string())
        } else if !ready {
            Some(state["why"].as_str().unwrap_or("not connected").to_string())
        } else if file.is_none() {
            Some(tl!("Export first, or type the path of a video").to_string())
        } else if fresh.is_none() {
            Some(tl!("Check the file first").to_string())
        } else if fresh.as_ref().is_some_and(|p| p["ok"] != true) {
            Some(tl!("Fix what the check found first").to_string())
        } else if busy {
            Some(tl!("Publishing…").to_string())
        } else {
            None
        };
        let label = match d {
            Destination::InstagramReel => tl!("Publish Reel"),
            _ => tl!("Upload to YouTube"),
        };
        let b = ui.add_enabled(why_not.is_none(), egui::Button::new(label).min_size(vec2(140.0, 28.0)));
        let b = match &why_not {
            Some(w) => b.on_disabled_hover_text(w.as_str()),
            None => b,
        };
        app.auto.add_response("export.publish.run", &b, why_not.as_deref().unwrap_or(label));
        if b.clicked()
            && let Some(path) = &file
        {
            action = Some(("publish.run", run_params(&app.ui.export.publish, d, path)));
        }
        // its progress and outcome
        if let Some(j) = job.as_ref().filter(|_| app.ui.export.publish.job_for == key) {
            let status = j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
            let (text, col, link) = if busy {
                ui.add(egui::ProgressBar::new(j.progress.fraction().clamp(0.0, 1.0)).desired_height(6.0));
                ui.ctx().request_repaint_after(std::time::Duration::from_millis(200));
                (status, t.accent, None)
            } else if let Some(e) = error_of(j) {
                (e, t.danger, None)
            } else {
                let out = filmcraft_engine::youtube::outcome_json(&app.session, j.id);
                let out = if out.is_null() { filmcraft_engine::instagram::outcome_json(&app.session, j.id) } else { out };
                let link = out["url"]
                    .as_str()
                    .or(out["permalink"].as_str())
                    .filter(|u| u.starts_with("https://www.youtube.com/") || u.starts_with("https://www.instagram.com/"))
                    .map(str::to_string);
                (status, t.render_green, link)
            };
            if !text.is_empty() {
                let r = ui.add(egui::Label::new(egui::RichText::new(&text).size(11.0).color(col)).wrap_mode(egui::TextWrapMode::Wrap));
                app.auto.add("export.publish.status", r.rect, &text);
            }
            if let Some(url) = link {
                let r = ui.link(tl!("Open it"));
                app.auto.add_response("export.publish.open", &r, &url);
                if r.clicked() {
                    let ctx = ui.ctx().clone();
                    crate::links::open(app, &ctx, &url);
                }
            }
        }
    });
    if run_check && let Some(path) = &file {
        match app.session.execute("publish.preflight", json!({"destination": d.id(), "path": path})) {
            Ok(v) => {
                app.ui.export.publish.preflight = Some(v);
                app.ui.export.publish.preflight_for = key.clone();
            }
            Err(e) => {
                app.ui.export.publish.preflight = None;
                app.ui.status = e.to_string();
            }
        }
    }
    if let Some((cmd, p)) = action {
        match app.session.execute(cmd, p) {
            Ok(v) => {
                let job = v["job"].as_u64();
                match cmd {
                    "publish.fixCopy" => app.ui.export.publish.fix_job = job,
                    "publish.run" => {
                        app.ui.export.publish.job = job;
                        app.ui.export.publish.job_for = key;
                    }
                    _ => {
                        app.ui.export.publish.connect_job = job;
                        app.ui.status = tl!("Finish the sign-in in your browser").into();
                    }
                }
            }
            Err(e) => app.ui.status = e.to_string(),
        }
    }
    if busy || fixing || connecting || job.as_ref().is_some_and(|j| !j.progress.finished.load(Ordering::Relaxed)) {
        ui.ctx().request_repaint_after(std::time::Duration::from_millis(250));
    }
}

/// The pre-flight's checks: a coloured dot, what the file has, what the destination takes, the fix.
fn checklist(app: &mut FilmcraftApp, ui: &mut egui::Ui, pre: &Value) {
    let t = app.tokens;
    let (errors, warnings) = (pre["errors"].as_u64().unwrap_or(0), pre["warnings"].as_u64().unwrap_or(0));
    let (summary, col) = if errors > 0 {
        (tlf!("{n} problem(s) to fix before publishing", n = errors), t.danger)
    } else if warnings > 0 {
        (tlf!("Ready to publish, with {n} warning(s)", n = warnings), t.render_yellow)
    } else {
        (tl!("Ready to publish").to_string(), t.render_green)
    };
    ui.add_space(4.0);
    let r = ui.label(egui::RichText::new(&summary).size(12.0).color(col));
    app.auto.add("export.publish.summary", r.rect, &summary);
    for c in pre["checks"].as_array().into_iter().flatten().take(64) {
        let status = c["status"].as_str().unwrap_or("info");
        let dot = match status {
            "pass" => t.render_green,
            "warn" => t.render_yellow,
            "fail" => t.danger,
            _ => t.text_faint,
        };
        let label = c["label"].as_str().unwrap_or_default();
        let value = c["value"].as_str().unwrap_or_default();
        let line = format!("{label}: {value}");
        let resp = ui
            .horizontal_wrapped(|ui| {
                let (r, _) = ui.allocate_exact_size(vec2(10.0, 14.0), Sense::hover());
                ui.painter().circle_filled(r.center(), 3.5, dot);
                ui.label(egui::RichText::new(&line).size(11.5).color(t.text));
                if status != "pass"
                    && let Some(need) = c["need"].as_str()
                {
                    ui.label(egui::RichText::new(format!("({need})")).size(11.0).color(t.text_dim));
                }
            })
            .response;
        let id = c["id"].as_str().unwrap_or("check");
        app.auto.add(&format!("export.publish.check.{id}"), resp.rect, &format!("{status}: {label} — {value}"));
        if status != "pass"
            && let Some(fix) = c["fix"].as_str()
        {
            ui.add(egui::Label::new(egui::RichText::new(fix).size(11.0).color(t.text_dim)).wrap_mode(egui::TextWrapMode::Wrap));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_form_becomes_publish_params() {
        let mut f = PublishForm { caption: "Robots #ai".into(), cover_seconds: 2.0, share_to_feed: false, ..Default::default() };
        let p = run_params(&f, Destination::InstagramReel, "/x/a.mp4");
        assert_eq!(p, json!({"destination": "instagramReel", "path": "/x/a.mp4", "caption": "Robots #ai", "shareToFeed": false, "coverSeconds": 2.0}));
        f.title = "  Every robot ".into();
        f.tags = "robots, , expo ".into();
        f.channel = "UCx".into();
        let p = run_params(&f, Destination::YoutubeShort, "/x/a.mp4");
        assert_eq!(
            p,
            json!({"destination": "youtubeShort", "path": "/x/a.mp4", "title": "Every robot", "tags": ["robots", "expo"], "privacy": "private", "channel": "UCx"})
        );
    }
}
