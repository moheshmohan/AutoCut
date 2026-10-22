//! Sequence ▸ Quick Short… (and the Templates panel's Quick Short… button): a vertical Short or
//! Reel from a template AutoCut picks and fills from the clips (`autocut.quick`, a background job
//! with the usual progress and cancel). The dialog chooses the source (the Project panel's
//! selection, or a folder), the template (Auto, or one of the starters made to fill themselves),
//! the length, the platform and layout, the topic and the look.
//!
//! The draft is UI state (`ui.set {quickShort: {...}}` merges into it and opens the dialog; `null`
//! closes it). Automation ids: `quickShort.source.selection`,
//! `quickShort.source.folder`, `quickShort.folder`, `quickShort.browse`, `quickShort.template`,
//! `quickShort.template.<auto | template id>`, `quickShort.seconds`, `quickShort.platform`,
//! `quickShort.platform.<reels | shorts | tiktok>`, `quickShort.aspect`, `quickShort.aspect.<9:16 | 1:1 | 4:5>`,
//! `quickShort.topic`, `quickShort.look`, `quickShort.look.<template | tech | vhs | clean>`,
//! `quickShort.info`, `quickShort.cancel`, `quickShort.go`.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::FilmcraftApp;
use crate::native_dialogs::FileDialog;

/// The platforms: (value, label). Product names, not translated.
const PLATFORMS: [(&str, &str); 3] = [("reels", "Instagram Reels"), ("shorts", "YouTube Shorts"), ("tiktok", "TikTok")];
/// The layouts.
const ASPECTS: [&str; 3] = ["9:16", "1:1", "4:5"];

/// The Quick Short dialog's choices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct QuickShortDraft {
    /// `selection` (the Project panel's) or `folder`.
    pub source: String,
    pub folder: String,
    /// `auto` or a template id.
    pub template: String,
    pub seconds: f64,
    /// `reels`, `shorts` or `tiktok`.
    pub platform: String,
    /// `9:16`, `1:1` or `4:5`.
    pub aspect: String,
    pub topic: String,
    /// Empty (the template's own), `tech`, `vhs` or `clean`.
    pub look: String,
    /// Why Make Short did not start.
    pub error: String,
}

impl Default for QuickShortDraft {
    fn default() -> Self {
        QuickShortDraft {
            source: "selection".into(),
            folder: String::new(),
            template: "auto".into(),
            seconds: 30.0,
            platform: "reels".into(),
            aspect: "9:16".into(),
            topic: String::new(),
            look: String::new(),
            error: String::new(),
        }
    }
}

/// `autocut.quick`'s parameters for the draft (`selection`: the Project panel's selected items).
pub fn params(d: &QuickShortDraft, selection: &[u64]) -> Result<Value, String> {
    let mut p = json!({
        "template": if d.template.trim().is_empty() { "auto" } else { d.template.trim() },
        "seconds": if d.seconds.is_finite() { d.seconds.clamp(10.0, 90.0).round() } else { 30.0 },
        "platform": d.platform,
        "aspect": d.aspect,
    });
    if d.source == "folder" {
        let f = d.folder.trim();
        if f.is_empty() {
            return Err(tl!("Choose a folder first").into());
        }
        p["paths"] = json!([f]);
    } else if !selection.is_empty() {
        p["items"] = json!(selection);
    }
    let topic: String = d.topic.trim().chars().take(400).collect();
    if !topic.is_empty() {
        p["topic"] = json!(topic);
    }
    if !d.look.is_empty() {
        p["look"] = json!(d.look);
    }
    Ok(p)
}

/// `ui.set {quickShort}`: merge `patch` into the draft (opening the dialog), or close it with
/// `null`; a value that does not fit is refused and nothing changes.
pub fn patch(app: &mut FilmcraftApp, patch: &Value) -> Result<(), String> {
    if patch.is_null() {
        app.ui.quick_short = None;
        return Ok(());
    }
    if !patch.is_object() {
        return Err("`quickShort` is an object of the dialog's choices, or null".into());
    }
    let mut cur = serde_json::to_value(app.ui.quick_short.clone().unwrap_or_default()).unwrap_or_default();
    if let (Some(c), Some(p)) = (cur.as_object_mut(), patch.as_object()) {
        for (k, v) in p {
            c.insert(k.clone(), v.clone());
        }
    }
    let d: QuickShortDraft = serde_json::from_value(cur).map_err(|e| format!("bad `quickShort`: {e}"))?;
    app.ui.quick_short = Some(d);
    Ok(())
}

/// Open the dialog (the menu entry, the Templates panel's button).
pub fn open(app: &mut FilmcraftApp) -> Result<Value, String> {
    filmcraft_engine::find_command("autocut.quick").map_or(Ok(()), |c| (c.enabled)(&app.session))?;
    let source = if app.session.state.project_selection.is_empty() { "folder" } else { "selection" };
    app.ui.quick_short = Some(QuickShortDraft { source: source.into(), ..app.ui.quick_short.clone().unwrap_or_default() });
    Ok(json!({"dialog": "quickShort"}))
}

/// The starters made to fill themselves: (id, name).
fn starters() -> Vec<(String, String)> {
    filmcraft_templates::builtin::AI_READY_IDS
        .iter()
        .filter_map(|id| filmcraft_templates::builtin::find(id).map(|p| (p.manifest.id.clone(), p.manifest.name.clone())))
        .collect()
}

fn look_label(l: &str) -> &'static str {
    match l {
        "tech" => tl!("Tech"),
        "vhs" => "VHS",
        "clean" => tl!("Clean"),
        _ => tl!("The template's"),
    }
}

pub fn show(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let Some(mut d) = app.ui.quick_short.clone() else { return };
    let selected: Vec<u64> = app.session.state.project_selection.iter().map(|i| i.0).collect();
    let mut close = false;
    let mut go = false;
    let mut browse = false;
    let mut elems: Vec<(String, egui::Rect, String)> = Vec::new();
    let starters = starters();
    egui::Window::new(tl!("Quick Short"))
        .id(egui::Id::new("Quick Short"))
        .collapsible(false)
        .resizable(false)
        .pivot(egui::Align2::CENTER_CENTER)
        .default_pos(crate::widgets::dialog_center(ctx))
        .show(ctx, |ui| {
            ui.set_min_width(420.0);
            ui.weak(tl!("Make a vertical Short or Reel: AutoCut picks a template and fills it from your clips."));
            ui.add_space(6.0);
            egui::Grid::new("quick-short").num_columns(2).spacing([12.0, 8.0]).show(ui, |ui| {
                ui.label(tl!("Source:"));
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        let r = ui.radio_value(&mut d.source, "selection".to_string(), tl!("Selected clips"));
                        elems.push(("quickShort.source.selection".into(), r.rect, tl!("Selected clips").into()));
                        let r = ui.radio_value(&mut d.source, "folder".to_string(), tl!("Folder"));
                        elems.push(("quickShort.source.folder".into(), r.rect, tl!("Folder").into()));
                    });
                    if d.source == "folder" {
                        ui.horizontal(|ui| {
                            let r = ui.add(
                                egui::TextEdit::singleline(&mut d.folder)
                                    .hint_text(tl!("Choose a folder of videos and photos"))
                                    .desired_width(260.0)
                                    .char_limit(4096),
                            );
                            elems.push(("quickShort.folder".into(), r.rect, d.folder.clone()));
                            let b = ui.button(tl!("Browse…"));
                            elems.push(("quickShort.browse".into(), b.rect, tl!("Browse…").into()));
                            if b.clicked() {
                                browse = true;
                            }
                        });
                    } else {
                        let info = if selected.is_empty() {
                            tl!("Nothing is selected in the Project panel: every video and photo of the project is used.").to_string()
                        } else {
                            tlf!("{n} clip(s) selected in the Project panel", n = selected.len())
                        };
                        let r = ui.weak(&info);
                        elems.push(("quickShort.info".into(), r.rect, info));
                    }
                });
                ui.end_row();
                ui.label(tl!("Template:"));
                let current = if d.template == "auto" {
                    tl!("Auto").to_string()
                } else {
                    starters.iter().find(|(id, _)| *id == d.template).map_or_else(|| d.template.clone(), |(_, n)| n.clone())
                };
                let cb = egui::ComboBox::from_id_salt("quick-short-template").selected_text(current.clone()).width(260.0).show_ui(ui, |ui| {
                    let r = ui.selectable_label(d.template == "auto", tl!("Auto"));
                    elems.push(("quickShort.template.auto".into(), r.rect, tl!("Auto").into()));
                    if r.clicked() {
                        d.template = "auto".into();
                    }
                    for (id, name) in &starters {
                        let r = ui.selectable_label(d.template == *id, name);
                        elems.push((format!("quickShort.template.{id}"), r.rect, name.clone()));
                        if r.clicked() {
                            d.template = id.clone();
                        }
                    }
                });
                elems.push(("quickShort.template".into(), cb.response.rect, current));
                ui.end_row();
                ui.label(tl!("Length:"));
                let r = ui.add(egui::Slider::new(&mut d.seconds, 10.0..=90.0).step_by(1.0).suffix(" s"));
                elems.push(("quickShort.seconds".into(), r.rect, format!("{:.0} s", d.seconds)));
                ui.end_row();
                ui.label(tl!("Platform:"));
                let current = PLATFORMS.iter().find(|(v, _)| *v == d.platform).map_or("Instagram Reels", |(_, l)| l);
                let cb = egui::ComboBox::from_id_salt("quick-short-platform").selected_text(current).width(180.0).show_ui(ui, |ui| {
                    for (v, l) in PLATFORMS {
                        let r = ui.selectable_label(d.platform == v, l);
                        elems.push((format!("quickShort.platform.{v}"), r.rect, l.into()));
                        if r.clicked() {
                            d.platform = v.into();
                        }
                    }
                });
                elems.push(("quickShort.platform".into(), cb.response.rect, current.into()));
                ui.end_row();
                ui.label(tl!("Layout:"));
                let cb = egui::ComboBox::from_id_salt("quick-short-aspect").selected_text(d.aspect.clone()).width(100.0).show_ui(ui, |ui| {
                    for a in ASPECTS {
                        let r = ui.selectable_label(d.aspect == a, a);
                        elems.push((format!("quickShort.aspect.{a}"), r.rect, a.into()));
                        if r.clicked() {
                            d.aspect = a.into();
                        }
                    }
                });
                elems.push(("quickShort.aspect".into(), cb.response.rect, d.aspect.clone()));
                ui.end_row();
                ui.label(tl!("Topic:"));
                let r = ui.add(egui::TextEdit::singleline(&mut d.topic).hint_text(tl!("What it is about (optional)")).desired_width(260.0).char_limit(400));
                elems.push(("quickShort.topic".into(), r.rect, d.topic.clone()));
                ui.end_row();
                ui.label(tl!("Look:"));
                let current = look_label(&d.look);
                let cb = egui::ComboBox::from_id_salt("quick-short-look").selected_text(current).width(180.0).show_ui(ui, |ui| {
                    for (v, id) in [("", "template"), ("tech", "tech"), ("vhs", "vhs"), ("clean", "clean")] {
                        let l = look_label(v);
                        let r = ui.selectable_label(d.look == v, l);
                        elems.push((format!("quickShort.look.{id}"), r.rect, l.into()));
                        if r.clicked() {
                            d.look = v.into();
                        }
                    }
                });
                elems.push(("quickShort.look".into(), cb.response.rect, current.into()));
                ui.end_row();
            });
            if !d.error.is_empty() {
                ui.colored_label(app.tokens.danger, &d.error);
            }
            ui.add_space(8.0);
            ui.horizontal(|ui| {
                let c = ui.button(tl!("Cancel"));
                elems.push(("quickShort.cancel".into(), c.rect, tl!("Cancel").into()));
                if c.clicked() {
                    close = true;
                }
                let o = ui.add(egui::Button::new(egui::RichText::new(tl!("Make Short")).color(egui::Color32::WHITE)).fill(app.tokens.accent));
                elems.push(("quickShort.go".into(), o.rect, tl!("Make Short").into()));
                if o.clicked() {
                    go = true;
                }
            });
        });
    for (id, r, l) in elems {
        app.auto.add(&id, r, &l);
    }
    if crate::widgets::escape_closes(ctx) {
        close = true;
    }
    if go {
        match params(&d, &selected).and_then(|p| app.session.execute("autocut.quick", p).map_err(|e| e.to_string())) {
            Ok(_) => {
                app.ui.status = tl!("Making a quick short…").into();
                close = true;
            }
            Err(e) => d.error = e,
        }
    }
    app.ui.quick_short = if close { None } else { Some(d) };
    if browse {
        app.pick_ui(FileDialog::Folder { at: None }, |app, paths| {
            if let (Some(p), Some(d)) = (paths.into_iter().next(), app.ui.quick_short.as_mut()) {
                d.folder = p;
                d.source = "folder".into();
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_draft_becomes_the_job_parameters() {
        let d = QuickShortDraft::default();
        assert_eq!(params(&d, &[]).unwrap(), json!({"template": "auto", "seconds": 30.0, "platform": "reels", "aspect": "9:16"}));
        assert_eq!(params(&d, &[3, 4]).unwrap()["items"], json!([3, 4]));
        let d = QuickShortDraft {
            source: "folder".into(),
            folder: " /clips ".into(),
            template: "builtin:tech-reveal".into(),
            seconds: f64::NAN,
            topic: "robots".into(),
            look: "vhs".into(),
            ..Default::default()
        };
        let p = params(&d, &[3]).unwrap();
        assert_eq!((p["paths"].clone(), p["items"].clone(), p["seconds"].as_f64()), (json!(["/clips"]), Value::Null, Some(30.0)));
        assert_eq!((p["template"].as_str(), p["topic"].as_str(), p["look"].as_str()), (Some("builtin:tech-reveal"), Some("robots"), Some("vhs")));
        assert!(params(&QuickShortDraft { source: "folder".into(), ..Default::default() }, &[]).is_err(), "a folder is needed");
        assert_eq!(params(&QuickShortDraft { seconds: 1e9, ..Default::default() }, &[]).unwrap()["seconds"], 90.0);
        // the starters it offers exist
        assert_eq!(starters().len(), filmcraft_templates::builtin::AI_READY_IDS.len());
    }
}
