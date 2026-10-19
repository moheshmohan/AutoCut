//! Recipes panel (Window ▸ Recipes): the built-in and saved recipes — parameterised chains of
//! engine commands and AutoCut intents. Pick one, set its parameters, Run (a background job with
//! progress and Cancel; each step checked before it runs, a failing step stops the run and is
//! shown). "Save this edit as a recipe": Record, edit as usual, Stop and save under a name. A
//! user recipe's steps can be edited as JSON.
//!
//! The draft is UI state: `ui.set {recipes: {selected, params, recordName, stepsJson, editOpen}}`.
//! Automation ids: `recipes.item.<id>`, `recipes.param.<id>`, `recipes.run`, `recipes.cancel`,
//! `recipes.progress`, `recipes.step.<n>`, `recipes.delete`, `recipes.record.start`,
//! `recipes.record.name`, `recipes.record.stop`, `recipes.record.cancel`, `recipes.edit`,
//! `recipes.edit.steps`, `recipes.edit.save`, `recipes.refresh`.

use std::collections::BTreeMap;
use std::sync::atomic::Ordering;

use egui::{Rect, pos2, vec2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use filmcraft_engine::recipes::{self as rc, ParamKind, Recipe};

use crate::FilmcraftApp;

const PAD: f32 = 12.0;
/// Longest text a field takes.
const MAX_TEXT: usize = 4096;
/// Largest steps JSON the editor takes.
const MAX_STEPS_JSON: usize = 256 * 1024;

/// The panel's state (UI state, serde so agents can read and set it).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct RecipesDraft {
    /// The recipe open (its id).
    pub selected: Option<String>,
    /// Its parameter values, by parameter id (the defaults until changed).
    pub params: BTreeMap<String, Value>,
    /// The name a recording is saved under.
    pub record_name: String,
    /// The steps editor (a user recipe's steps as JSON).
    pub steps_json: String,
    pub edit_open: bool,
    /// The run job shown.
    pub job: Option<u64>,
}

/// Apply `ui.set {recipes: patch}`.
pub fn patch(app: &mut FilmcraftApp, patch: &Value) -> Result<(), String> {
    let mut cur = serde_json::to_value(&app.ui.recipes).map_err(|e| e.to_string())?;
    crate::control::merge(&mut cur, patch);
    let mut d: RecipesDraft = serde_json::from_value(cur).map_err(|e| format!("bad `recipes`: {e}"))?;
    if d.params.len() > rc::MAX_PARAMS {
        return Err(format!("at most {} parameters", rc::MAX_PARAMS));
    }
    for v in d.params.values_mut() {
        if let Value::String(s) = v
            && s.chars().count() > MAX_TEXT
        {
            *s = s.chars().take(MAX_TEXT).collect();
        }
    }
    if d.record_name.chars().count() > 200 {
        d.record_name = d.record_name.chars().take(200).collect();
    }
    if d.steps_json.len() > MAX_STEPS_JSON {
        return Err(format!("the steps are larger than {MAX_STEPS_JSON} bytes"));
    }
    if d.selected != app.ui.recipes.selected && patch.get("params").is_none() {
        d.params.clear();
    }
    app.ui.recipes = d;
    Ok(())
}

enum Action {
    Select(String),
    Run,
    Cancel,
    Delete,
    RecordStart,
    RecordStop,
    RecordCancel,
    EditOpen,
    EditSave,
}

fn add(auto: &mut crate::automation::Registry, ui: &egui::Ui, id: &str, r: Rect, label: &str) {
    let shown = r.intersect(ui.clip_rect());
    if shown.is_positive() {
        auto.add(id, shown, label);
    }
}

fn small(ui: &mut egui::Ui, t: &crate::theme::Tokens, text: &str) {
    ui.label(egui::RichText::new(text).color(t.text_dim).size(11.0));
}

/// The run job's progress and status, and how it ended.
struct JobView {
    running: bool,
    fraction: f32,
    status: String,
    error: Option<String>,
}

fn job_view(app: &FilmcraftApp, id: u64) -> Option<JobView> {
    let j = app.session.jobs.iter().find(|j| j.id == id)?;
    let total = j.progress.total.load(Ordering::Relaxed);
    Some(JobView {
        running: !j.progress.finished.load(Ordering::Relaxed),
        fraction: if total > 0 { (j.progress.done.load(Ordering::Relaxed) as f32 / total as f32).clamp(0.0, 1.0) } else { 0.0 },
        status: j.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone(),
        error: j.progress.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone(),
    })
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.panel_bg);
    let ctx = ui.ctx().clone();
    // the library, read again when a recipe is saved or deleted (or Refresh)
    #[derive(Clone, Default)]
    struct Lib {
        entries: std::sync::Arc<Vec<rc::Entry>>,
        invalid: usize,
        revision: u64,
        read: bool,
    }
    let lib_id = egui::Id::new("recipes-panel-library");
    let mut cached = ctx.data(|d| d.get_temp::<Lib>(lib_id)).unwrap_or_default();
    if !cached.read || cached.revision != rc::library_revision() {
        let (entries, bad) = rc::library(&app.session);
        cached = Lib { entries: std::sync::Arc::new(entries), invalid: bad.len(), revision: rc::library_revision(), read: true };
        ctx.data_mut(|d| d.insert_temp(lib_id, cached.clone()));
    }
    let lib = cached.entries.clone();
    let invalid = cached.invalid;
    let recording = app.session.recipes.recording.as_ref().map(|r| r.steps.len());
    let mut actions: Vec<Action> = Vec::new();
    let body = Rect::from_min_max(pos2(rect.min.x + PAD, rect.min.y + 8.0), pos2(rect.max.x - PAD, rect.max.y - 8.0));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("recipes-panel"));
    let ui = &mut child;
    egui::ScrollArea::vertical().id_salt("recipes-scroll").auto_shrink([false, false]).show(ui, |ui| {
        ui.set_max_width(ui.available_width() - 6.0);
        let w = ui.available_width();
        // Save this edit as a recipe
        ui.label(egui::RichText::new(tl!("Save this edit as a recipe")).color(t.text).size(12.0).strong());
        match recording {
            None => {
                small(ui, &t, tl!("Record, edit as usual, then stop: the edits become a recipe you can run again."));
                let b = ui.button(tl!("● Record"));
                add(&mut app.auto, ui, "recipes.record.start", b.rect, tl!("Record"));
                if b.clicked() {
                    actions.push(Action::RecordStart);
                }
            }
            Some(n) => {
                small(ui, &t, &tlf!("Recording: {n} edits so far", n));
                ui.horizontal(|ui| {
                    let d = &mut app.ui.recipes;
                    let r = ui
                        .add(egui::TextEdit::singleline(&mut d.record_name).hint_text(tl!("Recipe name")).desired_width((w - 190.0).max(60.0)).char_limit(200));
                    add(&mut app.auto, ui, "recipes.record.name", r.rect, tl!("Recipe name"));
                    let b = ui.add_enabled(!d.record_name.trim().is_empty(), egui::Button::new(tl!("Stop and Save")));
                    add(&mut app.auto, ui, "recipes.record.stop", b.rect, tl!("Stop and Save"));
                    if b.clicked() {
                        actions.push(Action::RecordStop);
                    }
                    let b = ui.small_button(tl!("Discard"));
                    add(&mut app.auto, ui, "recipes.record.cancel", b.rect, tl!("Discard"));
                    if b.clicked() {
                        actions.push(Action::RecordCancel);
                    }
                });
            }
        }
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(egui::RichText::new(tl!("Recipes")).color(t.text).size(12.0).strong());
            let b = ui.small_button(tl!("Refresh"));
            add(&mut app.auto, ui, "recipes.refresh", b.rect, tl!("Refresh"));
            if b.clicked() {
                rc::bump_library();
            }
            if invalid > 0 {
                small(ui, &t, &tlf!("({n} files could not be read: recipe.list says why)", n = invalid));
            }
        });
        for e in lib.iter() {
            let on = app.ui.recipes.selected.as_deref() == Some(e.recipe.id.as_str());
            let label = if e.source == "user" { format!("{} · {}", e.recipe.name, tl!("yours")) } else { e.recipe.name.clone() };
            let r = ui.selectable_label(on, label);
            add(&mut app.auto, ui, &format!("recipes.item.{}", e.recipe.id), r.rect, &e.recipe.name);
            if r.on_hover_text(&e.recipe.description).clicked() {
                actions.push(Action::Select(e.recipe.id.clone()));
            }
        }
        // the recipe open
        let Some(sel) = app.ui.recipes.selected.clone() else { return };
        let Some(e) = lib.iter().find(|e| e.recipe.id == sel) else {
            small(ui, &t, &tlf!("The recipe {id} is gone", id = sel));
            return;
        };
        let r: &Recipe = &e.recipe;
        ui.add_space(8.0);
        ui.label(egui::RichText::new(&r.name).color(t.text).size(14.0).strong());
        small(ui, &t, &r.description);
        ui.add_space(4.0);
        for (i, s) in r.steps.iter().enumerate() {
            let kind = match &s.action {
                rc::Action::Command(c) => c.clone(),
                rc::Action::Intent(x) => tlf!("AutoCut: {intent}", intent = x),
            };
            let when = s.when.as_ref().map(|w| format!(" {}", tlf!("(when {param} is on)", param = w))).unwrap_or_default();
            ui.label(egui::RichText::new(format!("{}. {} — {kind}{when}", i + 1, s.label)).color(t.text_dim).size(11.0));
        }
        if !r.params.is_empty() {
            ui.add_space(6.0);
            ui.label(egui::RichText::new(tl!("Parameters")).color(t.text).size(12.0).strong());
        }
        let d = &mut app.ui.recipes;
        for p in &r.params {
            let v = d.params.entry(p.id.clone()).or_insert_with(|| p.default.clone());
            ui.horizontal(|ui| {
                ui.add_sized(vec2(170.0, 18.0), egui::Label::new(egui::RichText::new(&p.name).color(t.text_dim).size(11.5)));
                let id = format!("recipes.param.{}", p.id);
                let resp = match p.kind {
                    ParamKind::Bool => {
                        let mut b = v.as_bool().unwrap_or(false);
                        let r = ui.checkbox(&mut b, "");
                        *v = json!(b);
                        r
                    }
                    ParamKind::Number => {
                        let mut x = v.as_f64().unwrap_or(0.0);
                        let (lo, hi) = (p.min.unwrap_or(-1e9), p.max.unwrap_or(1e9));
                        let r = ui.add(egui::DragValue::new(&mut x).range(lo.min(hi)..=hi.max(lo)).speed(0.1));
                        *v = json!(x);
                        r
                    }
                    ParamKind::Choice => {
                        let mut cur = v.as_str().unwrap_or_default().to_string();
                        let r = egui::ComboBox::from_id_salt(("recipe-param", &p.id)).selected_text(cur.clone()).show_ui(ui, |ui| {
                            for c in &p.choices {
                                let o = ui.selectable_label(cur == *c, c);
                                app.auto.add(&format!("recipes.param.{}.{c}", p.id), o.rect, c);
                                if o.clicked() {
                                    cur = c.clone();
                                }
                            }
                        });
                        *v = json!(cur);
                        r.response
                    }
                    ParamKind::Json => {
                        let mut s = if v.is_string() { v.as_str().unwrap_or_default().to_string() } else { v.to_string() };
                        let r = ui.add(egui::TextEdit::singleline(&mut s).desired_width((w - 190.0).max(60.0)).char_limit(MAX_TEXT).code_editor());
                        if r.changed() {
                            *v = serde_json::from_str(&s).unwrap_or(Value::String(s));
                        }
                        r
                    }
                    ParamKind::Text | ParamKind::Path | ParamKind::Template => {
                        let mut s = v.as_str().unwrap_or_default().to_string();
                        let hint = match p.kind {
                            ParamKind::Path => tl!("a file path"),
                            ParamKind::Template => tl!("a template id (Templates panel)"),
                            _ => "",
                        };
                        let r = ui.add(egui::TextEdit::singleline(&mut s).hint_text(hint).desired_width((w - 190.0).max(60.0)).char_limit(MAX_TEXT));
                        *v = json!(s);
                        r
                    }
                };
                add(&mut app.auto, ui, &id, resp.rect, &p.name);
                if !p.help.is_empty() {
                    resp.on_hover_text(&p.help);
                }
            });
        }
        ui.add_space(8.0);
        let view = d.job.and_then(|j| job_view(app, j));
        let running = view.as_ref().is_some_and(|v| v.running);
        ui.horizontal(|ui| {
            let b = ui.add_enabled(!running && recording.is_none(), egui::Button::new(egui::RichText::new(tl!("Run")).strong()));
            add(&mut app.auto, ui, "recipes.run", b.rect, tl!("Run"));
            if b.clicked() {
                actions.push(Action::Run);
            }
            if running {
                let b = ui.button(tl!("Cancel"));
                add(&mut app.auto, ui, "recipes.cancel", b.rect, tl!("Cancel"));
                if b.clicked() {
                    actions.push(Action::Cancel);
                }
            }
            if e.source == "user" {
                let b = ui.small_button(tl!("Edit steps"));
                add(&mut app.auto, ui, "recipes.edit", b.rect, tl!("Edit steps"));
                if b.clicked() {
                    actions.push(Action::EditOpen);
                }
                let b = ui.add_enabled(!running, egui::Button::new(tl!("Delete")).small());
                add(&mut app.auto, ui, "recipes.delete", b.rect, tl!("Delete"));
                if b.clicked() {
                    actions.push(Action::Delete);
                }
            }
        });
        if let Some(v) = &view {
            let r = ui.add(egui::ProgressBar::new(v.fraction).desired_width(w - 18.0).fill(t.accent).show_percentage());
            add(&mut app.auto, ui, "recipes.progress", r.rect, &v.status);
            match &v.error {
                Some(e) => {
                    ui.label(egui::RichText::new(e).color(t.danger).size(11.5));
                }
                None => small(ui, &t, &v.status),
            }
            if running {
                ctx.request_repaint_after(std::time::Duration::from_millis(100));
            }
        }
        // how the last run's steps went
        if let Some(job) = app.ui.recipes.job {
            let out = rc::outcome_json(&app.session, job);
            if let Some(steps) = out["steps"].as_array() {
                for s in steps {
                    let n = s["step"].as_u64().unwrap_or(0);
                    let mark = if s["skipped"] == true {
                        "–"
                    } else if s["ok"] == true {
                        "✓"
                    } else {
                        "✗"
                    };
                    let line = match s["error"].as_str() {
                        Some(e) => format!("{mark} {n}. {} — {e}", s["label"].as_str().unwrap_or_default()),
                        None => format!("{mark} {n}. {}", s["label"].as_str().unwrap_or_default()),
                    };
                    let r = ui.label(egui::RichText::new(line).color(if s["ok"] == false { t.danger } else { t.text_dim }).size(11.0));
                    add(&mut app.auto, ui, &format!("recipes.step.{n}"), r.rect, s["label"].as_str().unwrap_or_default());
                }
            }
        }
        if app.ui.recipes.edit_open && e.source == "user" {
            ui.add_space(8.0);
            ui.label(egui::RichText::new(tl!("Steps (JSON)")).color(t.text).size(12.0).strong());
            let d = &mut app.ui.recipes;
            let r = ui.add(egui::TextEdit::multiline(&mut d.steps_json).code_editor().desired_rows(10).desired_width(w - 10.0));
            add(&mut app.auto, ui, "recipes.edit.steps", r.rect, tl!("Steps (JSON)"));
            let b = ui.button(tl!("Save Steps"));
            add(&mut app.auto, ui, "recipes.edit.save", b.rect, tl!("Save Steps"));
            if b.clicked() {
                actions.push(Action::EditSave);
            }
        }
    });
    for a in actions {
        run(app, a);
    }
}

fn run(app: &mut FilmcraftApp, a: Action) {
    match a {
        Action::Select(id) => {
            let d = &mut app.ui.recipes;
            if d.selected.as_deref() != Some(id.as_str()) {
                d.params.clear();
                d.edit_open = false;
            }
            d.selected = Some(id);
        }
        Action::Run => {
            let Some(id) = app.ui.recipes.selected.clone() else { return };
            let params = Value::Object(app.ui.recipes.params.iter().map(|(k, v)| (k.clone(), v.clone())).collect());
            match app.session.execute("recipe.run", json!({"name": id, "params": params})) {
                Ok(r) => {
                    app.ui.recipes.job = r["job"].as_u64();
                    app.ui.status = tl!("Running the recipe…").into();
                }
                Err(e) => app.ui.status = tlf!("Recipe: {e}", e),
            }
        }
        Action::Cancel => {
            if let Some(job) = app.ui.recipes.job {
                let _ = app.session.execute("jobs.cancel", json!({"job": job}));
            }
        }
        Action::Delete => {
            let Some(id) = app.ui.recipes.selected.clone() else { return };
            match app.session.execute("recipe.delete", json!({"name": id})) {
                Ok(_) => {
                    app.ui.recipes.selected = None;
                    app.ui.recipes.params.clear();
                    app.ui.status = tl!("Recipe deleted").into();
                }
                Err(e) => app.ui.status = e.to_string(),
            }
        }
        Action::RecordStart => match app.session.execute("recipe.record", json!({"action": "start"})) {
            Ok(_) => app.ui.status = tl!("Recording: edit as usual, then Stop and Save in Recipes").into(),
            Err(e) => app.ui.status = e.to_string(),
        },
        Action::RecordStop => {
            let name = app.ui.recipes.record_name.trim().to_string();
            match app.session.execute("recipe.record", json!({"action": "stop", "name": name})) {
                Ok(r) => {
                    let n = r["steps"].as_array().map_or(0, Vec::len);
                    if let Some(id) = r["recipe"]["id"].as_str() {
                        app.ui.recipes.selected = Some(id.to_string());
                        app.ui.recipes.params.clear();
                        app.ui.status = tlf!("Saved the recipe “{name}” ({n} steps)", name, n);
                    } else {
                        app.ui.status = tl!("Nothing was recorded").into();
                    }
                    app.ui.recipes.record_name.clear();
                }
                Err(e) => app.ui.status = e.to_string(),
            }
        }
        Action::RecordCancel => {
            let _ = app.session.execute("recipe.record", json!({"action": "cancel"}));
            app.ui.status = tl!("Recording discarded").into();
        }
        Action::EditOpen => {
            let Some(id) = app.ui.recipes.selected.clone() else { return };
            if let Ok(e) = rc::find(&app.session, &id, "recipes") {
                let steps = e.recipe.to_json()["steps"].clone();
                app.ui.recipes.steps_json = serde_json::to_string_pretty(&steps).unwrap_or_default();
                app.ui.recipes.edit_open = true;
            }
        }
        Action::EditSave => {
            let Some(id) = app.ui.recipes.selected.clone() else { return };
            let Ok(e) = rc::find(&app.session, &id, "recipes") else { return };
            let steps: Value = match serde_json::from_str(&app.ui.recipes.steps_json) {
                Ok(v) => v,
                Err(err) => {
                    app.ui.status = tlf!("The steps are not JSON: {e}", e = err);
                    return;
                }
            };
            let r = e.recipe.to_json();
            let p = json!({"name": e.recipe.name, "id": e.recipe.id, "description": e.recipe.description, "params": r["params"], "steps": steps, "overwrite": true});
            match app.session.execute("recipe.save", p) {
                Ok(_) => {
                    app.ui.recipes.edit_open = false;
                    app.ui.status = tl!("Recipe saved").into();
                }
                Err(e) => app.ui.status = tlf!("Recipe: {e}", e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafts_round_trip_as_ui_state() {
        let d = RecipesDraft { selected: Some("builtin:vertical-short".into()), params: [("seconds".to_string(), json!(30))].into(), ..Default::default() };
        let v = serde_json::to_value(&d).unwrap();
        assert_eq!(v["selected"], "builtin:vertical-short");
        assert_eq!(serde_json::from_value::<RecipesDraft>(v).unwrap(), d);
    }
}
