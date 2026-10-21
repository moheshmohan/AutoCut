//! The crash-recovery prompt and the Revert confirmation (Settings ▸ Auto Save lives in
//! `panels::settings`).
//!
//! When the project was saved again after the auto-save (by FilmCraft or another program), or that
//! can't be ruled out (`primaryAction` of `file.recoveryList`), the recovery prompt warns and makes
//! keeping the saved project the primary choice; Recover stays available as "Recover Anyway".
//!
//! Automation ids: `recovery.item.<n>`, `recovery.recover`, `recovery.open` (Open Saved Project),
//! `recovery.discard`, `recovery.later`, `recovery.warning` (the warning, when shown);
//! `revert.yes`, `revert.no`.

use egui::{Color32, CornerRadius, Frame, Margin, RichText, Stroke, Ui, vec2};
use serde_json::{Value, json};

use crate::theme::Tokens;
use crate::{Dialog, FilmcraftApp};

/// Dialog-local state.
#[derive(Default)]
pub struct FileDialogState {
    pub recovery_choice: usize,
}

/// The recovery prompt's primary button (`primaryAction` of a `file.recoveryList` entry).
#[derive(Clone, Copy, PartialEq)]
enum Primary {
    Recover,
    OpenSaved,
    Later,
}

impl Primary {
    /// Anything unexpected keeps what is on disk.
    fn of(candidate: &Value) -> Self {
        match candidate["primaryAction"].as_str() {
            Some("recover") => Self::Recover,
            Some("openSaved") if candidate["projectPath"].is_string() => Self::OpenSaved,
            _ => Self::Later,
        }
    }
}

fn modal_frame(t: &Tokens) -> Frame {
    Frame::new().fill(t.panel_bg).stroke(Stroke::new(1.0, t.separator)).corner_radius(CornerRadius::same(10)).inner_margin(Margin::same(0))
}

fn button(app: &mut FilmcraftApp, ui: &mut Ui, id: &str, label: &str, primary: bool) -> bool {
    let t = app.tokens;
    let text = RichText::new(label).size(13.0).color(if primary { Color32::WHITE } else { t.text });
    let b = egui::Button::new(text)
        .min_size(vec2(88.0, 30.0))
        .corner_radius(CornerRadius::same(15))
        .fill(if primary { t.accent } else { Color32::TRANSPARENT })
        .stroke(if primary { Stroke::NONE } else { Stroke::new(1.5, t.text_faint) });
    let r = ui.add(b).on_hover_cursor(egui::CursorIcon::PointingHand);
    app.auto.add(id, r.rect, label);
    r.clicked()
}

/// The warning above the recovery question (title, detail), from what recovering would replace
/// (`projectFile` of a candidate, `autosave::ProjectFileState`).
fn recovery_warning(c: &Value, primary: Primary) -> Option<(&'static str, &'static str)> {
    let w = match (c["projectFile"].as_str(), primary) {
        (Some("newer"), _) => (
            tl!("The saved project is newer than this auto-save."),
            tl!("It was saved after the auto-save, by FilmCraft or another program. Recovering would replace it with older changes."),
        ),
        (Some("missing"), _) => (
            tl!("The project file can't be found where it was saved."),
            tl!("If it was moved or renamed, open it from its new location and check it before recovering."),
        ),
        (_, Primary::Recover) => return None,
        _ => (tl!("The saved project could not be checked."), tl!("It may have been saved after this auto-save. Open it and check it before recovering.")),
    };
    Some(w)
}

pub fn show_recovery(app: &mut FilmcraftApp, ctx: &egui::Context) -> bool {
    let t = app.tokens;
    let list = app.session.execute("file.recoveryList", json!({})).unwrap_or_default();
    let items = list.as_array().cloned().unwrap_or_default();
    if items.is_empty() {
        return false;
    }
    let mut choice = app.file_dialogs.recovery_choice.min(items.len() - 1);
    let mut open = true;
    let c = items.get(choice).cloned().unwrap_or_default();
    let primary = Primary::of(&c);
    egui::Modal::new(egui::Id::new("recovery-modal")).frame(modal_frame(&t)).show(ctx, |ui| {
        ui.set_width(if primary == Primary::Recover { 520.0 } else { 580.0 });
        Frame::new().inner_margin(Margin { left: 28, right: 28, top: 24, bottom: 20 }).show(ui, |ui| {
            ui.label(RichText::new(tl!("Recover Unsaved Changes")).size(18.0).strong().color(t.text));
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(6.0);
            let name = c["projectName"].as_str().unwrap_or(tl!("Untitled"));
            let when = c["savedAt"].as_str().unwrap_or("");
            let why = if c["cleanExit"].as_bool() == Some(true) {
                tlf!("FilmCraft was closed while “{name}” had unsaved changes.", name)
            } else {
                tlf!("FilmCraft quit unexpectedly while “{name}” had unsaved changes.", name)
            };
            ui.label(RichText::new(why).size(13.5).color(t.text));
            if let Some((title, detail)) = recovery_warning(&c, primary) {
                ui.add_space(8.0);
                let r = Frame::new()
                    .fill(t.danger.gamma_multiply(0.14))
                    .stroke(Stroke::new(1.0, t.danger))
                    .corner_radius(CornerRadius::same(6))
                    .inner_margin(Margin::symmetric(12, 8))
                    .show(ui, |ui| {
                        ui.set_width(ui.available_width());
                        ui.label(RichText::new(title).size(13.5).strong().color(t.text));
                        ui.label(RichText::new(detail).size(12.5).color(t.text));
                        if let Some(saved) = c["projectModifiedAt"].as_str() {
                            ui.add_space(2.0);
                            let times = tlf!("Auto-save from {snapshot}; project file last saved {saved}.", snapshot = when, saved);
                            ui.label(RichText::new(times).size(12.0).color(t.text_dim));
                        }
                    });
                app.auto.add("recovery.warning", r.response.rect, &format!("{title} {detail}"));
            }
            ui.add_space(4.0);
            if primary == Primary::Recover {
                ui.label(RichText::new(tlf!("Recover unsaved changes from {when}?", when)).size(13.5).color(t.text));
            } else {
                if primary == Primary::OpenSaved {
                    ui.label(RichText::new(tl!("Open the saved project instead?")).size(13.5).color(t.text));
                }
                let later = tlf!("You can still recover the changes from {when} later with File ▸ Recover Unsaved Changes…", when);
                ui.label(RichText::new(later).size(12.5).color(t.text_dim));
            }
            if let Some(p) = c["projectPath"].as_str() {
                ui.label(RichText::new(p).size(11.5).color(t.text_dim));
            } else {
                ui.label(RichText::new(tl!("The project had not been saved yet.")).size(11.5).color(t.text_dim));
            }
            if items.len() > 1 {
                ui.add_space(8.0);
                ui.label(RichText::new(tlf!("{n} sessions have unsaved changes:", n = items.len())).size(12.0).color(t.text_dim));
                for (i, it) in items.iter().enumerate() {
                    let label = format!("{} — {}", it["projectName"].as_str().unwrap_or(tl!("Untitled")), it["savedAt"].as_str().unwrap_or(""));
                    let r = ui.radio(choice == i, RichText::new(&label).size(12.5));
                    app.auto.add(&format!("recovery.item.{i}"), r.rect, &label);
                    if r.clicked() {
                        choice = i;
                    }
                }
            }
            ui.add_space(18.0);
            let id = c["id"].as_str().unwrap_or("").to_string();
            ui.horizontal(|ui| {
                if button(app, ui, "recovery.discard", tl!("Discard"), false) {
                    if let Err(e) = app.session.execute("file.discardRecovery", json!({ "id": id })) {
                        app.ui.status = e.to_string();
                    }
                    choice = 0;
                }
                // right to left: the primary button is the rightmost
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let (recover, later) = match primary {
                        Primary::Recover => {
                            let recover = button(app, ui, "recovery.recover", tl!("Recover"), true);
                            (recover, button(app, ui, "recovery.later", tl!("Not Now"), false))
                        }
                        Primary::OpenSaved => {
                            // keeps the saved file; the snapshot stays for File ▸ Recover Unsaved Changes…
                            if button(app, ui, "recovery.open", tl!("Open Saved Project"), true) {
                                let path = c["projectPath"].as_str().unwrap_or("");
                                match crate::menus::invoke(app, ctx, "file.open", json!({ "path": path })) {
                                    Ok(_) => open = false,
                                    Err(e) => app.ui.status = e,
                                }
                            }
                            let recover = button(app, ui, "recovery.recover", tl!("Recover Anyway"), false);
                            (recover, button(app, ui, "recovery.later", tl!("Not Now"), false))
                        }
                        Primary::Later => {
                            let later = button(app, ui, "recovery.later", tl!("Not Now"), true);
                            (button(app, ui, "recovery.recover", tl!("Recover Anyway"), false), later)
                        }
                    };
                    if recover {
                        match app.session.execute("file.recover", json!({ "id": id })) {
                            Ok(_) => open = false,
                            Err(e) => app.ui.status = e.to_string(),
                        }
                    }
                    if later {
                        open = false;
                    }
                });
            });
        });
    });
    app.file_dialogs.recovery_choice = choice;
    open && !app.session.recovery_candidates().is_empty()
}

pub fn show_revert(app: &mut FilmcraftApp, ctx: &egui::Context) -> bool {
    let t = app.tokens;
    let file = app.session.path.as_deref().and_then(|p| std::path::Path::new(p).file_name()).map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let mut open = true;
    let r = egui::Modal::new(egui::Id::new("revert-modal")).frame(modal_frame(&t)).show(ctx, |ui| {
        ui.set_width(460.0);
        Frame::new().inner_margin(Margin { left: 28, right: 28, top: 24, bottom: 20 }).show(ui, |ui| {
            ui.label(RichText::new(tl!("Revert")).size(18.0).strong().color(t.text));
            ui.add_space(6.0);
            ui.separator();
            ui.add_space(6.0);
            ui.label(RichText::new(tlf!("Are you sure you want to discard your changes to '{file}'?", file)).size(13.5).color(t.text));
            ui.add_space(18.0);
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if button(app, ui, "revert.yes", tl!("Yes"), true) {
                    if let Err(e) = app.session.execute("file.revert", json!({})) {
                        app.ui.status = e.to_string();
                    }
                    open = false;
                }
                if button(app, ui, "revert.no", tl!("No"), false) {
                    open = false;
                }
            });
        });
    });
    // (the modal's own close also fires on Escape: not while a drag is being cancelled, #580)
    open && !(r.should_close() && !ctx.input(|i| i.pointer.any_down()))
}

/// Draw the dialog if it is one of ours; returns Some(still open).
pub fn show(app: &mut FilmcraftApp, ctx: &egui::Context, d: Dialog) -> Option<bool> {
    match d {
        Dialog::Preferences => Some(crate::panels::settings::show(app, ctx)),
        Dialog::Recovery => Some(show_recovery(app, ctx)),
        Dialog::RevertConfirm => Some(show_revert(app, ctx)),
        _ => None,
    }
}
