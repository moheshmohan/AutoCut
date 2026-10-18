//! Templates panel: the video template browser. A grid of the built-in starters and the user's
//! saved templates, each with a thumbnail rendered from its own placeholder sequence
//! (`template.thumbnail`), filtered by layout (16:9, 9:16, 1:1, 4:5), category and how many
//! media it takes. Picking one opens it: choose the layout, fill its slots from the project's
//! media (the Google Photos picks already imported can be shown alone), type its text — or
//! Auto-fill from the bin — then Preview (frames of the filled template, the project untouched)
//! and Build (`template.apply`: one undo step; the new sequence opens in the Timeline to tweak).
//!
//! The draft is UI state: `ui.set {templates: {...}}` patches it (`fills` are `template.apply`'s
//! `slots`: item ids for media, strings for text). Automation ids: `templates.search`,
//! `templates.aspect`, `templates.category`, `templates.slotCount`, `templates.refresh`,
//! `templates.card.<template id>`, `templates.back`, `templates.variant`,
//! `templates.variant.<id>`, `templates.googlePhotos`, `templates.autofill`, `templates.clear`,
//! `templates.slot.<slot>.add`, `templates.slot.<slot>.option.<item id>`,
//! `templates.slot.<slot>.<i>.remove`, `templates.slot.<slot>.text.<i>`, `templates.name`,
//! `templates.preview`, `templates.previewFrame.<k>`, `templates.build`, `templates.openTimeline`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use egui::{Rect, Sense, pos2, vec2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use filmcraft_engine::video_template_frames::{self as frames, Candidate, THUMB_SIDE};
use filmcraft_engine::video_templates as vt;
use filmcraft_templates::manifest::{Manifest, SlotKind};

use crate::FilmcraftApp;
use crate::dock::PanelKind;

const PAD: f32 = 12.0;
const CARD_W: f32 = 176.0;
const THUMB_H: f32 = 108.0;
const CARD_H: f32 = THUMB_H + 54.0;
const FOOTER_H: f32 = 44.0;
/// New thumbnails drawn per frame (each renders the template's sequence once).
const THUMBS_PER_FRAME: usize = 2;
/// Preview frames.
const PREVIEW_FRAMES: usize = 6;
/// Values a slot holds at most in the draft (a template's slot takes at most 256).
const MAX_VALUES: usize = 256;
/// Longest text the fields take.
const MAX_TEXT: usize = 1000;

/// The aspect filter's choices: (value, label).
const ASPECTS: [(&str, &str); 5] = [("", "Any layout"), ("16:9", "16:9"), ("9:16", "9:16"), ("1:1", "1:1"), ("4:5", "4:5")];
/// The media-count filter's choices: (value, label, min, max).
const SLOT_COUNTS: [(&str, &str, usize, usize); 5] = [
    ("", "Any number of clips", 0, usize::MAX),
    ("1", "One clip or photo", 0, 1),
    ("2-3", "2 or 3 clips", 2, 3),
    ("4-6", "4 to 6 clips", 4, 6),
    ("7+", "7 or more clips", 7, usize::MAX),
];

/// The browser's state (UI state, serde so agents can read and set it).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct TemplatesDraft {
    /// Search in names, descriptions and tags.
    pub query: String,
    /// Only templates with this layout (`9:16`); empty: all.
    pub aspect: String,
    /// Only this category; empty: all.
    pub category: String,
    /// How many clips or photos it takes: "", "1", "2-3", "4-6", "7+".
    pub slots: String,
    /// The template being filled (its id); none: the grid.
    pub selected: Option<String>,
    /// Its layout.
    pub variant: String,
    /// Slot → values, as `template.apply`'s `slots`: item ids for media, strings for text.
    pub fills: BTreeMap<String, Vec<Value>>,
    /// Offer only Google Photos picks in the slot menus.
    pub google_photos: bool,
    /// Name of the sequence Build makes (empty: the template's name and layout).
    pub name: String,
    /// The sequence the last Build made.
    pub built: Option<u64>,
}

impl TemplatesDraft {
    /// `template.apply` / `template.preview` parameters for the selected template.
    pub fn apply_params(&self) -> Option<Value> {
        let id = self.selected.as_ref()?;
        let mut p = json!({"template": id, "slots": self.fills});
        if !self.variant.is_empty() {
            p["variant"] = json!(self.variant);
        }
        if !self.name.trim().is_empty() {
            p["name"] = json!(self.name.trim());
        }
        Some(p)
    }
}

/// Apply `ui.set {templates: patch}`: fields merged into the draft (`fills` replaced per slot).
pub fn patch(app: &mut FilmcraftApp, patch: &Value) -> Result<(), String> {
    let mut cur = serde_json::to_value(&app.ui.templates).map_err(|e| e.to_string())?;
    crate::control::merge(&mut cur, patch);
    let mut d: TemplatesDraft = serde_json::from_value(cur).map_err(|e| format!("bad `templates`: {e}"))?;
    for v in d.fills.values_mut() {
        v.truncate(MAX_VALUES);
        for x in v.iter_mut() {
            if let Value::String(s) = x
                && s.len() > MAX_TEXT
            {
                *s = s.chars().take(MAX_TEXT).collect();
            }
        }
    }
    if d.selected != app.ui.templates.selected {
        // a template chosen by an agent: its default layout unless one is given
        if let Some(id) = &d.selected
            && patch.get("variant").is_none()
        {
            d.variant = vt::find_template(&app.session, id, "templates").map(|e| e.package.manifest.default_variant).unwrap_or_default();
        }
    }
    app.ui.templates = d;
    Ok(())
}

/// One template in the grid.
#[derive(Clone, Debug)]
struct Card {
    id: String,
    name: String,
    category: String,
    description: String,
    tags: Vec<String>,
    seconds: f64,
    media_count: usize,
    aspects: Vec<String>,
    default_variant: String,
    captions: bool,
    user: bool,
}

/// The library, read once (Refresh reads it again).
#[derive(Clone, Default)]
struct Library {
    cards: Arc<Vec<Card>>,
    invalid: usize,
    read: bool,
}

/// Textures of thumbnails (`id|variant`) and of the current preview.
#[derive(Clone, Default)]
struct Textures {
    thumbs: HashMap<String, egui::TextureHandle>,
    failed: HashMap<String, String>,
    preview_key: String,
    preview: Vec<(egui::TextureHandle, f64)>,
}

/// The template being filled, as read.
#[derive(Clone)]
struct Selected {
    id: String,
    entry: Arc<vt::Entry>,
}

fn sel_id() -> egui::Id {
    egui::Id::new("templates-panel-selected")
}

fn lib_id() -> egui::Id {
    egui::Id::new("templates-panel-library")
}

fn tex_id() -> egui::Id {
    egui::Id::new("templates-panel-textures")
}

fn read_library(app: &FilmcraftApp) -> Library {
    let (lib, invalid) = vt::library(&app.session);
    let cards = lib
        .iter()
        .map(|e| {
            let m = &e.package.manifest;
            Card {
                id: m.id.clone(),
                name: m.name.clone(),
                category: m.category.clone(),
                description: m.description.clone(),
                tags: m.tags.clone(),
                seconds: m.duration.seconds(),
                media_count: m.media_count(),
                aspects: m.variants.iter().map(|v| v.id.clone()).collect(),
                default_variant: m.default_variant.clone(),
                captions: m.captions_slot().is_some(),
                user: e.source != "builtin",
            }
        })
        .collect();
    Library { cards: Arc::new(cards), invalid: invalid.len(), read: true }
}

/// The cards the filters let through.
fn filtered<'a>(cards: &'a [Card], d: &TemplatesDraft) -> Vec<&'a Card> {
    let q = d.query.trim().to_lowercase();
    let (min, max) = SLOT_COUNTS.iter().find(|(v, ..)| *v == d.slots).map_or((0, usize::MAX), |(_, _, a, b)| (*a, *b));
    cards
        .iter()
        .filter(|c| d.aspect.is_empty() || c.aspects.contains(&d.aspect))
        .filter(|c| d.category.is_empty() || c.category.eq_ignore_ascii_case(&d.category))
        .filter(|c| c.media_count >= min && c.media_count <= max)
        .filter(|c| {
            q.is_empty()
                || c.name.to_lowercase().contains(&q)
                || c.description.to_lowercase().contains(&q)
                || c.tags.iter().any(|t| t.to_lowercase().contains(&q))
        })
        .collect()
}

/// The layout a card shows: the aspect filter's when it has it, else its own.
fn card_variant(c: &Card, d: &TemplatesDraft) -> String {
    if !d.aspect.is_empty() && c.aspects.contains(&d.aspect) { d.aspect.clone() } else { c.default_variant.clone() }
}

/// The candidates a slot of `kind` offers.
fn offer(all: &[Candidate], kind: SlotKind, google_photos: bool) -> Vec<Candidate> {
    all.iter()
        .filter(|c| !google_photos || c.google_photos)
        .filter(|c| match kind {
            SlotKind::Music => c.has_audio,
            SlotKind::Photo => c.kind == "photo" || c.kind == "video",
            SlotKind::Video | SlotKind::Logo => c.kind == "video" || c.kind == "photo",
            SlotKind::Text | SlotKind::Captions => false,
        })
        .cloned()
        .collect()
}

/// Fill every empty picture slot from the project's media, in order: photos for photo slots
/// (then clips), clips for video slots, each medium used once while there are enough.
fn autofill(m: &Manifest, d: &mut TemplatesDraft, all: &[Candidate]) -> usize {
    let mut used: Vec<u64> = d.fills.values().flatten().filter_map(Value::as_u64).collect();
    let mut filled = 0;
    let mut slots: Vec<_> = m.slots.iter().filter(|s| s.kind.is_visual()).collect();
    slots.sort_by_key(|s| s.order);
    for s in slots {
        if d.fills.get(&s.id).is_some_and(|v| !v.is_empty()) {
            continue;
        }
        let mut pool = offer(all, s.kind, d.google_photos);
        let prefer = if s.kind == SlotKind::Video { "video" } else { "photo" };
        pool.sort_by_key(|c| c.kind != prefer);
        let fresh: Vec<u64> = pool.iter().map(|c| c.item.0).filter(|i| !used.contains(i)).collect();
        let source = if fresh.is_empty() { pool.iter().map(|c| c.item.0).collect() } else { fresh };
        let take: Vec<Value> = source.iter().take(s.count as usize).map(|i| json!(i)).collect();
        if take.is_empty() {
            continue;
        }
        used.extend(take.iter().filter_map(Value::as_u64));
        filled += 1;
        d.fills.insert(s.id.clone(), take);
    }
    filled
}

enum Action {
    Refresh,
    Select(String),
    Back,
    AutoFill,
    Clear,
    Preview,
    Build,
    OpenTimeline,
}

/// Register widget `id` by the part of `r` the scroll area shows.
fn add(auto: &mut crate::automation::Registry, ui: &egui::Ui, id: &str, r: Rect, label: &str) {
    let shown = r.intersect(ui.clip_rect());
    if shown.is_positive() {
        auto.add(id, shown, label);
    }
}

fn small(ui: &mut egui::Ui, t: &crate::theme::Tokens, text: &str) {
    ui.label(egui::RichText::new(text).color(t.text_dim).size(11.0));
}

/// A thumbnail texture for `id` in `variant`, rendering at most `budget` new ones this frame.
fn thumb(session: &filmcraft_engine::Session, ctx: &egui::Context, id: &str, variant: &str, budget: &mut usize) -> Option<egui::TextureHandle> {
    let key = format!("{id}|{variant}");
    let have = ctx.data(|d| d.get_temp::<Textures>(tex_id())).unwrap_or_default();
    if let Some(t) = have.thumbs.get(&key) {
        return Some(t.clone());
    }
    if have.failed.contains_key(&key) || *budget == 0 {
        return None;
    }
    *budget -= 1;
    // (the texture is made outside `data_mut`, which holds the context's lock)
    match frames::thumbnail(session, id, Some(variant), None, THUMB_SIDE) {
        Ok(f) => {
            let img = egui::ColorImage::from_rgba_unmultiplied([f.width as usize, f.height as usize], &f.rgba);
            let h = ctx.load_texture(format!("template-thumb-{key}"), img, egui::TextureOptions::LINEAR);
            ctx.data_mut(|d| d.get_temp_mut_or_default::<Textures>(tex_id()).thumbs.insert(key, h.clone()));
            Some(h)
        }
        Err(e) => {
            ctx.data_mut(|d| d.get_temp_mut_or_default::<Textures>(tex_id()).failed.insert(key, e.to_string()));
            None
        }
    }
}

fn draw_image(ui: &egui::Ui, r: Rect, tex: Option<&egui::TextureHandle>) {
    ui.painter().rect_filled(r, 3.0, egui::Color32::from_rgb(14, 14, 16));
    if let Some(t) = tex {
        let sz = t.size_vec2();
        ui.painter().image(t.id(), crate::panels::monitor::fit(r, sz.x, sz.y), Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), egui::Color32::WHITE);
    }
}

pub fn show(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect) {
    let t = app.tokens;
    ui.painter().rect_filled(rect, 0.0, t.panel_bg);
    let ctx = ui.ctx().clone();
    let mut lib = ctx.data(|d| d.get_temp::<Library>(lib_id())).unwrap_or_default();
    if !lib.read {
        lib = read_library(app);
        ctx.data_mut(|d| d.insert_temp(lib_id(), lib.clone()));
    }
    let mut actions: Vec<Action> = Vec::new();
    let mut budget = THUMBS_PER_FRAME;
    let mut missing = false;
    let body = Rect::from_min_max(pos2(rect.min.x + PAD, rect.min.y + 8.0), pos2(rect.max.x - PAD, rect.max.y - FOOTER_H));
    match app.ui.templates.selected.clone() {
        None => {
            // the filters
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("templates-grid"));
            let ui = &mut child;
            let w = ui.available_width();
            ui.horizontal(|ui| {
                let d = &mut app.ui.templates;
                let r = ui.add(egui::TextEdit::singleline(&mut d.query).hint_text("Search templates").desired_width((w - 90.0).max(60.0)).char_limit(120));
                add(&mut app.auto, ui, "templates.search", r.rect, "Search templates");
                let b = ui.small_button("Refresh");
                add(&mut app.auto, ui, "templates.refresh", b.rect, "Refresh");
                if b.on_hover_text("Read the template folder again").clicked() {
                    actions.push(Action::Refresh);
                }
            });
            ui.horizontal_wrapped(|ui| {
                let d = &mut app.ui.templates;
                let label = ASPECTS.iter().find(|(v, _)| *v == d.aspect).map_or("Any layout", |(_, l)| l).to_string();
                let cb = egui::ComboBox::from_id_salt("templates-aspect").selected_text(label).width(110.0).show_ui(ui, |ui| {
                    for (v, l) in ASPECTS {
                        let r = ui.selectable_label(d.aspect == v, l);
                        app.auto.add(&format!("templates.aspect.{}", if v.is_empty() { "any" } else { v }), r.rect, l);
                        if r.clicked() {
                            d.aspect = v.to_string();
                        }
                    }
                });
                add(&mut app.auto, ui, "templates.aspect", cb.response.rect, "Layout");
                let cats: Vec<String> = {
                    let mut c: Vec<String> = lib.cards.iter().map(|c| c.category.clone()).collect();
                    c.sort();
                    c.dedup();
                    c
                };
                let label = if d.category.is_empty() { "All categories".to_string() } else { d.category.clone() };
                let cb = egui::ComboBox::from_id_salt("templates-category").selected_text(label).width(130.0).show_ui(ui, |ui| {
                    let r = ui.selectable_label(d.category.is_empty(), "All categories");
                    app.auto.add("templates.category.all", r.rect, "All categories");
                    if r.clicked() {
                        d.category.clear();
                    }
                    for c in &cats {
                        let r = ui.selectable_label(d.category == *c, c);
                        app.auto.add(&format!("templates.category.{c}"), r.rect, c);
                        if r.clicked() {
                            d.category = c.clone();
                        }
                    }
                });
                add(&mut app.auto, ui, "templates.category", cb.response.rect, "Category");
                let label = SLOT_COUNTS.iter().find(|(v, ..)| *v == d.slots).map_or("Any number of clips", |(_, l, ..)| l).to_string();
                let cb = egui::ComboBox::from_id_salt("templates-slots").selected_text(label).width(150.0).show_ui(ui, |ui| {
                    for (v, l, ..) in SLOT_COUNTS {
                        let r = ui.selectable_label(d.slots == v, l);
                        app.auto.add(&format!("templates.slotCount.{}", if v.is_empty() { "any" } else { v }), r.rect, l);
                        if r.clicked() {
                            d.slots = v.to_string();
                        }
                    }
                });
                add(&mut app.auto, ui, "templates.slotCount", cb.response.rect, "Media slots");
            });
            let shown = filtered(&lib.cards, &app.ui.templates);
            let what = match (shown.len(), lib.invalid) {
                (n, 0) => format!("{n} of {} templates", lib.cards.len()),
                (n, bad) => format!("{n} of {} templates ({bad} in the folder could not be read: template.list says why)", lib.cards.len()),
            };
            small(ui, &t, &what);
            ui.add_space(4.0);
            let cards: Vec<Card> = shown.into_iter().cloned().collect();
            egui::ScrollArea::vertical().id_salt("templates-cards").auto_shrink([false, false]).show(ui, |ui| {
                let w = ui.available_width();
                let cols = (((w + 8.0) / (CARD_W + 8.0)).floor() as usize).max(1);
                let rows = cards.len().div_ceil(cols);
                let (area, _) = ui.allocate_exact_size(vec2(w, rows as f32 * (CARD_H + 8.0)), Sense::hover());
                for (k, c) in cards.iter().enumerate() {
                    let (col, row) = (k % cols, k / cols);
                    let r = Rect::from_min_size(pos2(area.min.x + col as f32 * (CARD_W + 8.0), area.min.y + row as f32 * (CARD_H + 8.0)), vec2(CARD_W, CARD_H));
                    if !ui.is_rect_visible(r) {
                        continue;
                    }
                    let variant = card_variant(c, &app.ui.templates);
                    let tex = thumb(&app.session, &ctx, &c.id, &variant, &mut budget);
                    missing |= tex.is_none();
                    let resp = ui.interact(r, egui::Id::new(("template-card", &c.id)), Sense::click());
                    let fill = if resp.hovered() { t.hover } else { t.field_bg };
                    ui.painter().rect_filled(r, 5.0, fill);
                    draw_image(ui, Rect::from_min_size(r.min + vec2(6.0, 6.0), vec2(CARD_W - 12.0, THUMB_H - 6.0)), tex.as_ref());
                    let name_at = pos2(r.min.x + 8.0, r.min.y + THUMB_H + 6.0);
                    ui.painter().text(name_at, egui::Align2::LEFT_TOP, &c.name, egui::FontId::proportional(12.5), t.text);
                    let mut line = format!("{:.0} s · {} clips · {}", c.seconds, c.media_count, c.aspects.join(" "));
                    if c.captions {
                        line.push_str(" · CC");
                    }
                    ui.painter().text(name_at + vec2(0.0, 18.0), egui::Align2::LEFT_TOP, line, egui::FontId::proportional(10.5), t.text_dim);
                    let tag = if c.user { format!("{} · yours", c.category) } else { c.category.clone() };
                    ui.painter().text(name_at + vec2(0.0, 32.0), egui::Align2::LEFT_TOP, tag, egui::FontId::proportional(10.0), t.text_faint);
                    add(&mut app.auto, ui, &format!("templates.card.{}", c.id), r, &c.name);
                    if resp.on_hover_text(&c.description).clicked() {
                        actions.push(Action::Select(c.id.clone()));
                    }
                }
            });
        }
        Some(id) => {
            // the template, read once while it is open (a user template is a folder on disk)
            let cached = ctx.data(|d| d.get_temp::<Selected>(sel_id())).filter(|s| s.id == id);
            let entry = match cached {
                Some(s) => s.entry,
                None => match vt::find_template(&app.session, &id, "templates") {
                    Ok(e) => {
                        let e = Arc::new(e);
                        ctx.data_mut(|d| d.insert_temp(sel_id(), Selected { id: id.clone(), entry: e.clone() }));
                        e
                    }
                    Err(_) => {
                        small(ui, &t, &format!("The template {id} is gone"));
                        app.ui.templates.selected = None;
                        return;
                    }
                },
            };
            let m = entry.package.manifest.clone();
            if m.variant(Some(&app.ui.templates.variant)).is_none() {
                app.ui.templates.variant = m.default_variant.clone();
            }
            let variant = app.ui.templates.variant.clone();
            let tex = thumb(&app.session, &ctx, &m.id, &variant, &mut budget);
            missing |= tex.is_none();
            let all = frames::candidates(&app.session, None);
            let placeholders: HashMap<String, Vec<String>> =
                m.slots.iter().filter(|s| s.kind == SlotKind::Text).map(|s| (s.id.clone(), entry.package.placeholder_texts(&s.id))).collect();
            let preview_key = format!("{}|{}|{}", m.id, variant, serde_json::to_string(&app.ui.templates.fills).unwrap_or_default());
            let textures = ctx.data(|d| d.get_temp::<Textures>(tex_id())).unwrap_or_default();
            let preview: Vec<(egui::TextureHandle, f64)> = if textures.preview_key == preview_key { textures.preview.clone() } else { Vec::new() };
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("templates-detail"));
            let ui = &mut child;
            egui::ScrollArea::vertical().id_salt("templates-detail-scroll").auto_shrink([false, false]).show(ui, |ui| {
                ui.set_max_width(ui.available_width() - 6.0);
                let w = ui.available_width();
                ui.horizontal(|ui| {
                    let b = ui.small_button("‹ All templates");
                    add(&mut app.auto, ui, "templates.back", b.rect, "All templates");
                    if b.clicked() {
                        actions.push(Action::Back);
                    }
                    ui.label(egui::RichText::new(&m.name).color(t.text).size(14.0).strong());
                });
                small(ui, &t, &format!("{} · {:.1} s · {}", m.category, m.duration.seconds(), m.description));
                ui.add_space(6.0);
                // the layout and its thumbnail
                ui.horizontal(|ui| {
                    let d = &mut app.ui.templates;
                    let current = m.variant(Some(&d.variant)).map_or(d.variant.clone(), |v| format!("{} ({}×{})", v.name, v.width, v.height));
                    let cb = egui::ComboBox::from_id_salt("templates-variant").selected_text(current).width((w - 20.0).min(320.0)).show_ui(ui, |ui| {
                        for v in &m.variants {
                            let r = ui.selectable_label(d.variant == v.id, format!("{} ({}×{})", v.name, v.width, v.height));
                            app.auto.add(&format!("templates.variant.{}", v.id), r.rect, &v.name);
                            if r.clicked() {
                                d.variant = v.id.clone();
                            }
                        }
                    });
                    add(&mut app.auto, ui, "templates.variant", cb.response.rect, "Layout");
                });
                let th = (w * 0.5).clamp(90.0, 220.0);
                let (r, _) = ui.allocate_exact_size(vec2(w, th), Sense::hover());
                draw_image(ui, r, tex.as_ref());
                ui.add_space(6.0);
                ui.horizontal_wrapped(|ui| {
                    let d = &mut app.ui.templates;
                    let c = ui.checkbox(&mut d.google_photos, "Only Google Photos picks");
                    add(&mut app.auto, ui, "templates.googlePhotos", c.rect, "Only Google Photos picks");
                    let b = ui.button("Auto-fill from the bin");
                    add(&mut app.auto, ui, "templates.autofill", b.rect, "Auto-fill from the bin");
                    if b.clicked() {
                        actions.push(Action::AutoFill);
                    }
                    let b = ui.small_button("Clear");
                    add(&mut app.auto, ui, "templates.clear", b.rect, "Clear");
                    if b.clicked() {
                        actions.push(Action::Clear);
                    }
                });
                // the slots
                let mut slots: Vec<_> = m.slots.iter().collect();
                slots.sort_by_key(|s| s.order);
                for s in slots {
                    ui.add_space(6.0);
                    egui::Frame::new().fill(t.field_bg).stroke(egui::Stroke::new(1.0, t.field_border)).corner_radius(4.0).inner_margin(6.0).show(ui, |ui| {
                        ui.set_width(w - 14.0);
                        let d = &mut app.ui.templates;
                        let values = d.fills.entry(s.id.clone()).or_default();
                        let need = if s.min_count > 0 { format!("needs {}", s.min_count) } else { "optional".into() };
                        let head = match s.kind {
                            SlotKind::Text | SlotKind::Captions => s.name.clone(),
                            _ => format!("{} · {} of {} ({need})", s.name, values.len(), s.count),
                        };
                        ui.label(egui::RichText::new(head).color(t.text).size(12.0).strong());
                        if !s.prompt.is_empty() {
                            small(ui, &t, &s.prompt);
                        }
                        match s.kind {
                            SlotKind::Text => {
                                // start from the template's own text
                                let own = placeholders.get(&s.id).cloned().unwrap_or_default();
                                while values.len() < s.count as usize {
                                    values.push(json!(own.get(values.len()).cloned().unwrap_or_default()));
                                }
                                values.truncate(s.count as usize);
                                for (i, v) in values.iter_mut().enumerate() {
                                    let mut text = v.as_str().unwrap_or("").to_string();
                                    let max = s.constraints.max_chars.map_or(MAX_TEXT, |n| n as usize).min(MAX_TEXT);
                                    let r = ui.add(
                                        egui::TextEdit::singleline(&mut text).hint_text("the template's own text").desired_width(w - 30.0).char_limit(max),
                                    );
                                    add(&mut app.auto, ui, &format!("templates.slot.{}.text.{i}", s.id), r.rect, &s.name);
                                    *v = json!(text);
                                }
                            }
                            SlotKind::Captions => {
                                let from: Vec<String> =
                                    s.captions.iter().flat_map(|c| c.source.iter()).filter_map(|x| m.slot(x).map(|sl| sl.name.clone())).collect();
                                small(ui, &t, &format!("Transcribed from {} when you build (needs a speech model: Transcribe…)", from.join(" and ")));
                            }
                            _ => {
                                let mut remove = None;
                                for (i, v) in values.iter().enumerate() {
                                    let name = v.as_u64().and_then(|n| all.iter().find(|c| c.item.0 == n)).map_or_else(|| v.to_string(), |c| c.name.clone());
                                    ui.horizontal(|ui| {
                                        ui.label(egui::RichText::new(format!("{}. {name}", i + 1)).color(t.text).size(11.5));
                                        let b = ui.small_button("✕");
                                        add(&mut app.auto, ui, &format!("templates.slot.{}.{i}.remove", s.id), b.rect, "Remove");
                                        if b.clicked() {
                                            remove = Some(i);
                                        }
                                    });
                                }
                                if let Some(i) = remove {
                                    values.remove(i);
                                }
                                if values.len() < s.count as usize {
                                    let choices = offer(&all, s.kind, d.google_photos);
                                    let label = if choices.is_empty() { "Nothing in the project fits" } else { "Add from the project…" };
                                    let cb = egui::ComboBox::from_id_salt(("templates-add", &s.id)).selected_text(label).width((w - 30.0).min(320.0)).show_ui(
                                        ui,
                                        |ui| {
                                            for c in &choices {
                                                let text = format!("{}{} ({})", if c.google_photos { "Google Photos · " } else { "" }, c.name, c.kind);
                                                let r = ui.selectable_label(false, text);
                                                app.auto.add(&format!("templates.slot.{}.option.{}", s.id, c.item.0), r.rect, &c.name);
                                                if r.clicked() {
                                                    values.push(json!(c.item.0));
                                                }
                                            }
                                        },
                                    );
                                    add(&mut app.auto, ui, &format!("templates.slot.{}.add", s.id), cb.response.rect, "Add from the project");
                                }
                                if s.kind == SlotKind::Music && values.is_empty() {
                                    let what = match &s.default {
                                        Some(filmcraft_templates::SlotDefault::Generate { mood, bpm, .. }) => {
                                            format!("Empty: a {mood} bed at {bpm:.0} bpm is generated")
                                        }
                                        _ => "Empty: the template's placeholder stays".into(),
                                    };
                                    small(ui, &t, &what);
                                }
                            }
                        }
                    });
                }
                ui.add_space(8.0);
                let d = &mut app.ui.templates;
                let r = ui.add(egui::TextEdit::singleline(&mut d.name).hint_text(format!("{} ({})", m.name, d.variant)).desired_width(w).char_limit(200));
                add(&mut app.auto, ui, "templates.name", r.rect, "Sequence name");
                // the preview filmstrip
                if !preview.is_empty() {
                    ui.add_space(6.0);
                    ui.label(egui::RichText::new("Preview").color(t.text).size(12.0).strong());
                    let n = preview.len().max(1) as f32;
                    let fw = ((w - 4.0 * (n - 1.0)) / n).max(20.0);
                    let aspect = m.variant(Some(&d.variant)).map_or(16.0 / 9.0, |v| f64::from(v.width) / f64::from(v.height.max(1))) as f32;
                    let fh = (fw / aspect).clamp(20.0, 220.0);
                    let (strip, _) = ui.allocate_exact_size(vec2(w, fh), Sense::hover());
                    for (k, (tx, secs)) in preview.iter().enumerate() {
                        let r = Rect::from_min_size(pos2(strip.min.x + k as f32 * (fw + 4.0), strip.min.y), vec2(fw, fh));
                        draw_image(ui, r, Some(tx));
                        add(&mut app.auto, ui, &format!("templates.previewFrame.{k}"), r, &format!("{secs:.1} s"));
                    }
                }
                if d.built.is_some() {
                    ui.add_space(6.0);
                    ui.horizontal(|ui| {
                        small(ui, &t, "Built: it is open in the Timeline to tweak.");
                        let b = ui.small_button("Show the Timeline");
                        add(&mut app.auto, ui, "templates.openTimeline", b.rect, "Show the Timeline");
                        if b.clicked() {
                            actions.push(Action::OpenTimeline);
                        }
                    });
                }
            });
            // footer: Preview | Build
            let fy = rect.max.y - FOOTER_H + 8.0;
            let half = (rect.width() - PAD * 2.0 - 8.0) / 2.0;
            for (k, (bid, text)) in [("templates.preview", "Preview"), ("templates.build", "Build")].into_iter().enumerate() {
                let r = Rect::from_min_size(pos2(rect.min.x + PAD + k as f32 * (half + 8.0), fy), vec2(half, 28.0));
                let primary = k == 1;
                let b = ui.put(
                    r,
                    egui::Button::new(egui::RichText::new(text).strong().color(if primary { egui::Color32::WHITE } else { t.text })).fill(if primary {
                        t.accent
                    } else {
                        t.field_bg
                    }),
                );
                app.auto.add(bid, r, text);
                if b.clicked() {
                    actions.push(if primary { Action::Build } else { Action::Preview });
                }
            }
        }
    }
    if missing {
        ctx.request_repaint();
    }
    for a in actions {
        run(app, &ctx, a);
    }
}

fn run(app: &mut FilmcraftApp, ctx: &egui::Context, a: Action) {
    match a {
        Action::Refresh => {
            ctx.data_mut(|d| {
                d.remove::<Library>(lib_id());
                d.remove::<Textures>(tex_id());
                d.remove::<Selected>(sel_id());
            });
        }
        Action::Select(id) => {
            let d = &mut app.ui.templates;
            let variant = vt::find_template(&app.session, &id, "templates").map(|e| {
                let m = e.package.manifest;
                if !d.aspect.is_empty() && m.variant(Some(&d.aspect)).is_some() { d.aspect.clone() } else { m.default_variant }
            });
            d.variant = variant.unwrap_or_default();
            d.selected = Some(id);
            d.fills.clear();
            d.built = None;
            d.name.clear();
        }
        Action::Back => app.ui.templates.selected = None,
        Action::Clear => app.ui.templates.fills.clear(),
        Action::AutoFill => {
            let Some(id) = app.ui.templates.selected.clone() else { return };
            let Ok(e) = vt::find_template(&app.session, &id, "templates") else { return };
            let all = frames::candidates(&app.session, None);
            let n = autofill(&e.package.manifest, &mut app.ui.templates, &all);
            app.ui.status = if n == 0 { "Nothing in the project fits the empty slots".into() } else { format!("Filled {n} slots from the project") };
        }
        Action::Preview => {
            let Some(p) = app.ui.templates.apply_params() else { return };
            let id = app.ui.templates.selected.clone().unwrap_or_default();
            let r = vt::find_template(&app.session, &id, "template.preview").and_then(|e| {
                let wanted = vt::read_slots(&e.package.manifest, &p, "template.preview")?;
                let fills = vt::fills_without_import(&e.package.manifest, wanted, "template.preview")?;
                frames::preview(&app.session, &id, p["variant"].as_str(), fills, PREVIEW_FRAMES, 240)
            });
            match r {
                Ok((fs, warnings)) => {
                    let key = format!("{}|{}|{}", id, app.ui.templates.variant, serde_json::to_string(&app.ui.templates.fills).unwrap_or_default());
                    let list: Vec<(egui::TextureHandle, f64)> = fs
                        .iter()
                        .enumerate()
                        .map(|(k, f)| {
                            let img = egui::ColorImage::from_rgba_unmultiplied([f.width as usize, f.height as usize], &f.rgba);
                            (ctx.load_texture(format!("template-preview-{k}"), img, egui::TextureOptions::LINEAR), f.at.seconds())
                        })
                        .collect();
                    ctx.data_mut(|d| {
                        let tx = d.get_temp_mut_or_default::<Textures>(tex_id());
                        tx.preview_key = key;
                        tx.preview = list;
                    });
                    app.ui.status = match warnings.first() {
                        Some(w) => format!("Preview: {w}"),
                        None => format!("Preview: {} frames", fs.len()),
                    };
                }
                Err(e) => app.ui.status = format!("Preview: {e}"),
            }
        }
        Action::Build => {
            let Some(p) = app.ui.templates.apply_params() else { return };
            match app.session.execute("template.apply", p) {
                Ok(r) => {
                    app.ui.templates.built = r["sequence"].as_u64();
                    let warn = r["warnings"].as_array().and_then(|w| w.first()).and_then(Value::as_str).map(|w| format!(" ({w})")).unwrap_or_default();
                    app.ui.status = format!("Built “{}”, {:.1} s{warn}", r["name"].as_str().unwrap_or("the sequence"), r["seconds"].as_f64().unwrap_or(0.0));
                    app.show_panel(PanelKind::Timeline);
                }
                Err(e) => app.ui.status = format!("Build: {e}"),
            }
        }
        Action::OpenTimeline => app.show_panel(PanelKind::Timeline),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn card(id: &str, cat: &str, media: usize, aspects: &[&str]) -> Card {
        Card {
            id: id.into(),
            name: id.into(),
            category: cat.into(),
            description: String::new(),
            tags: vec!["beat".into()],
            seconds: 10.0,
            media_count: media,
            aspects: aspects.iter().map(|a| a.to_string()).collect(),
            default_variant: aspects.first().copied().unwrap_or("16:9").into(),
            captions: false,
            user: false,
        }
    }

    #[test]
    fn filters_by_layout_category_count_and_words() {
        let cards = vec![card("a", "Vlog", 1, &["16:9", "9:16"]), card("b", "Social", 4, &["1:1", "16:9", "9:16"]), card("c", "Vlog", 2, &["16:9", "9:16"])];
        let ids = |d: &TemplatesDraft| filtered(&cards, d).iter().map(|c| c.id.clone()).collect::<Vec<_>>();
        assert_eq!(ids(&TemplatesDraft::default()), ["a", "b", "c"]);
        assert_eq!(ids(&TemplatesDraft { aspect: "1:1".into(), ..Default::default() }), ["b"]);
        assert_eq!(ids(&TemplatesDraft { category: "vlog".into(), ..Default::default() }), ["a", "c"]);
        assert_eq!(ids(&TemplatesDraft { slots: "2-3".into(), ..Default::default() }), ["c"]);
        assert_eq!(ids(&TemplatesDraft { slots: "4-6".into(), ..Default::default() }), ["b"]);
        assert_eq!(ids(&TemplatesDraft { query: "BEAT".into(), ..Default::default() }), ["a", "b", "c"]);
        assert!(ids(&TemplatesDraft { query: "nothing".into(), ..Default::default() }).is_empty());
        // a card shows the filtered layout when it has it
        let d = TemplatesDraft { aspect: "1:1".into(), ..Default::default() };
        assert_eq!(card_variant(&cards[1], &d), "1:1");
        assert_eq!(card_variant(&cards[0], &d), "16:9");
    }

    #[test]
    fn apply_params_carry_the_fills() {
        let mut d = TemplatesDraft { selected: Some("builtin:quote-card".into()), variant: "1:1".into(), ..Default::default() };
        d.fills.insert("quote".into(), vec![json!("Be kind")]);
        d.fills.insert("background".into(), vec![json!(12)]);
        let p = d.apply_params().unwrap();
        assert_eq!(p, json!({"template": "builtin:quote-card", "variant": "1:1", "slots": {"background": [12], "quote": ["Be kind"]}}));
        assert!(TemplatesDraft::default().apply_params().is_none());
    }
}
