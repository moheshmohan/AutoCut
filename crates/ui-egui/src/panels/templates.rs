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
//!
//! Authoring and sharing (A3): Save as Template… (`templates.saveAs`, also Sequence ▸ Save as Video
//! Template…) opens a form — `templates.save.name` / `.description` / `.author` / `.license` /
//! `.category`, `templates.save.slotId` and `templates.save.mark.<kind>` / `.mark.clear` (flag the
//! selected clips), `templates.save.slot.<id>.name`, `templates.save.captions`,
//! `templates.save.variant.<layout>`, `templates.save.posterOn` / `.poster`,
//! `templates.save.bundle.<item>`, `templates.save.zip`, `templates.save.cancel` / `.close` /
//! `.go` — that writes into the user templates folder. `templates.install.path` and
//! `templates.install.go` install a `.zip` or an https link. An open template's Share section:
//! `templates.share.exportPath` / `.export`, `templates.share.otioPath` / `.otio`,
//! `templates.share.link` and its QR code `templates.share.qr`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use egui::{Rect, Sense, pos2, vec2};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use filmcraft_engine::video_template_frames::{self as frames, Candidate, THUMB_SIDE};
use filmcraft_engine::video_template_share as vts;
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
    /// The Save as Template form, open while set.
    pub save: Option<SaveDraft>,
    /// Install: a `.zip` package's path or an https link to one.
    pub install: String,
    /// The install-from-a-link job running.
    pub install_job: Option<u64>,
    /// Share (the template open): where Export .zip writes (a folder or a `.zip`).
    pub export_path: String,
    /// Share: where Export OpenTimelineIO writes (`.otio`).
    pub otio_path: String,
    /// Share: the link the package will be downloaded from, shown as a QR code.
    pub link: String,
}

/// The Save as Template form (UI state): `template.save`'s parameters for the active sequence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct SaveDraft {
    pub name: String,
    pub description: String,
    pub author: String,
    pub license: String,
    pub category: String,
    /// The slot id the Mark buttons flag the selected clips with (empty: the kind's own).
    pub slot: String,
    /// Display names of the slots, by slot id.
    pub slot_names: BTreeMap<String, String>,
    /// A captions slot (of the video slots with sound).
    pub captions: bool,
    /// The layouts besides the sequence's own.
    pub variants: Vec<String>,
    /// Where the thumbnail is taken, seconds (none: the template's own choice).
    pub poster_seconds: Option<f64>,
    /// Media items of the user's (no slot) to bundle with their attribution.
    pub bundle: Vec<u64>,
    /// Write one `.zip` instead of a folder.
    pub zip: bool,
}

impl Default for SaveDraft {
    fn default() -> Self {
        SaveDraft {
            name: String::new(),
            description: String::new(),
            author: String::new(),
            license: "MIT OR Apache-2.0".into(),
            category: "My Templates".into(),
            slot: String::new(),
            slot_names: BTreeMap::new(),
            captions: false,
            variants: vec!["9:16".into(), "1:1".into(), "4:5".into()],
            poster_seconds: None,
            bundle: Vec::new(),
            zip: false,
        }
    }
}

impl SaveDraft {
    /// `template.save`'s parameters (into the user templates folder).
    pub fn params(&self) -> Value {
        let mut p = json!({
            "name": self.name.trim(), "description": self.description, "author": self.author.trim(),
            "license": self.license.trim(), "category": self.category.trim(), "variants": self.variants,
            "captions": self.captions, "zip": self.zip,
        });
        if !self.slot_names.is_empty() {
            p["slots"] =
                Value::Object(self.slot_names.iter().filter(|(_, n)| !n.trim().is_empty()).map(|(k, n)| (k.clone(), json!({"name": n.trim()}))).collect());
        }
        if let Some(x) = self.poster_seconds.filter(|x| x.is_finite() && *x >= 0.0) {
            p["posterSeconds"] = json!(x);
        }
        if !self.bundle.is_empty() {
            p["bundle"] = json!(self.bundle);
        }
        p
    }
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
    let cap = |s: &mut String, n: usize| {
        if s.chars().count() > n {
            *s = s.chars().take(n).collect();
        }
    };
    for s in [&mut d.install, &mut d.export_path, &mut d.otio_path, &mut d.link] {
        cap(s, 4096);
    }
    if let Some(sd) = d.save.as_mut() {
        for (s, n) in
            [(&mut sd.name, 200), (&mut sd.description, 4000), (&mut sd.author, 200), (&mut sd.license, 200), (&mut sd.category, 200), (&mut sd.slot, 64)]
        {
            cap(s, n);
        }
        sd.variants.truncate(8);
        sd.bundle.truncate(256);
        if sd.slot_names.len() > 64 {
            sd.slot_names = sd.slot_names.iter().take(64).map(|(k, v)| (k.clone(), v.clone())).collect();
        }
        for v in sd.slot_names.values_mut() {
            cap(v, 200);
        }
        sd.poster_seconds = sd.poster_seconds.filter(|x| x.is_finite() && *x >= 0.0 && *x <= 3600.0);
    }
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
    /// The user folder's revision when it was read ([`vt::library_revision`]).
    revision: u64,
}

/// Textures of thumbnails (`id|variant`) and of the current preview.
#[derive(Clone, Default)]
struct Textures {
    thumbs: HashMap<String, egui::TextureHandle>,
    failed: HashMap<String, String>,
    preview_key: String,
    preview: Vec<(egui::TextureHandle, f64)>,
    /// The template the preview was drawn from (the key it was found by, and the package as read).
    preview_of: Option<(String, Arc<filmcraft_templates::Package>)>,
}

impl Textures {
    /// The user templates folder changed (a template saved or installed, by this window, another
    /// one or an agent): what did not change stays. The built-in thumbnails are compiled in; the
    /// preview stays while its template reads the same. The rest is drawn anew. (Dropping it all
    /// made the preview vanish whenever any template was installed, and with it a UI test that
    /// ran beside one that installs.)
    fn library_changed(&mut self, session: &filmcraft_engine::Session) {
        let builtin = |key: &String| key.rsplit_once('|').is_some_and(|(id, _)| filmcraft_templates::builtin::find(id).is_some_and(|p| p.manifest.id == id));
        self.thumbs.retain(|k, _| builtin(k));
        self.failed.retain(|k, _| builtin(k));
        // (by their files: `Package`'s `==` is false between two copies of the same template, as
        // a fragment's motion effects hold NaN for "the frame's centre")
        let same =
            self.preview_of.as_ref().is_some_and(|(key, was)| vt::find_template(session, key, "templates").is_ok_and(|e| e.package.files() == was.files()));
        if !same {
            self.preview_key.clear();
            self.preview.clear();
            self.preview_of = None;
        }
    }
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
    Library { cards: Arc::new(cards), invalid: invalid.len(), read: true, revision: vt::library_revision() }
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
    OpenSave,
    CancelSave,
    Save,
    /// Flag the selected clips as a slot of this kind (None: clear the flag).
    Mark(Option<SlotKind>),
    Install,
    Export,
    ExportOtio,
}

/// The kinds the Mark buttons flag clips with: (kind, label).
const MARK_KINDS: [(SlotKind, &str); 5] =
    [(SlotKind::Video, "Video"), (SlotKind::Photo, "Photo"), (SlotKind::Logo, "Logo"), (SlotKind::Music, "Music"), (SlotKind::Text, "Text")];

/// The layouts a saved template can have besides the sequence's own.
const LAYOUTS: [&str; 4] = ["16:9", "9:16", "1:1", "4:5"];

fn kind_label(k: SlotKind) -> &'static str {
    match k {
        SlotKind::Video => tl!("Video"),
        SlotKind::Photo => tl!("Photo"),
        SlotKind::Logo => tl!("Logo"),
        SlotKind::Music => tl!("Music"),
        SlotKind::Text => tl!("Text"),
        SlotKind::Captions => tl!("Captions"),
    }
}

/// The slots flagged on the active sequence's clips: id → (kind, clips).
fn flagged(app: &FilmcraftApp) -> BTreeMap<String, (SlotKind, usize)> {
    let mut out: BTreeMap<String, (SlotKind, usize)> = BTreeMap::new();
    let Some(q) = app.session.active_sequence() else { return out };
    for t in q.video_tracks.iter().chain(q.audio_tracks.iter()) {
        for c in &t.items {
            for f in c.markers.iter().filter_map(|m| filmcraft_templates::apply::parse_slot_marker(&m.name)) {
                out.entry(f.slot).or_insert((f.kind, 0)).1 += 1;
            }
        }
    }
    out
}

/// The Template Slot submenu of the Timeline's clip menu: flag the selected clips as a slot of a
/// kind (`template.markSlot`), or clear the flag. Automation ids
/// `timeline.clipMenu.templateSlot.<kind>` and `timeline.clipMenu.templateSlot.clear`.
pub fn slot_menu(app: &mut FilmcraftApp, ui: &mut egui::Ui, any: bool) {
    let mut run: Option<Value> = None;
    ui.add_enabled_ui(any, |ui| {
        let r = ui.menu_button(crate::i18n::t("Template Slot"), |ui| {
            for (k, _) in MARK_KINDS {
                let label = kind_label(k);
                let r = ui.button(label);
                app.auto.add(&format!("timeline.clipMenu.templateSlot.{}", k.id()), r.rect, label);
                if r.clicked() {
                    run = Some(json!({"kind": k.id()}));
                    ui.close();
                }
            }
            ui.separator();
            let r = ui.button(tl!("Not a Slot"));
            app.auto.add("timeline.clipMenu.templateSlot.clear", r.rect, tl!("Not a Slot"));
            if r.clicked() {
                run = Some(json!({"clear": true}));
                ui.close();
            }
        });
        app.auto.add_response("timeline.clipMenu.template.markSlot", &r.response, crate::i18n::t("Template Slot"));
    });
    if let Some(p) = run {
        app.ui.status = match app.session.execute("template.markSlot", p) {
            Ok(v) if v["slot"].is_null() => tlf!("{n} clips are no slot now", n = v["clips"]),
            Ok(v) => tlf!("{n} clips are now slot “{slot}”", n = v["clips"], slot = v["slot"].as_str().unwrap_or_default()),
            Err(e) => e.to_string(),
        };
    }
}

/// Sequence ▸ Save as Video Template…: the panel with the save form open.
pub fn open_save(app: &mut FilmcraftApp) -> Result<Value, String> {
    filmcraft_engine::find_command("template.save").map_or(Ok(()), |c| (c.enabled)(&app.session))?;
    start_save(app);
    app.show_panel(PanelKind::Templates);
    Ok(json!({"panel": "Templates"}))
}

fn start_save(app: &mut FilmcraftApp) {
    let name = app.session.state.active_sequence.and_then(|s| app.session.project.item(s)).map(|i| i.name.clone()).unwrap_or_default();
    let mut sd = app.ui.templates.save.take().unwrap_or_default();
    if sd.name.trim().is_empty() {
        sd.name = name;
    }
    app.ui.templates.save = Some(sd);
}

/// A finished install-from-a-link job: say so and read the folder again.
fn poll_install(app: &mut FilmcraftApp, ctx: &egui::Context) {
    let Some(id) = app.ui.templates.install_job else { return };
    let Some(j) = app.session.jobs.iter().find(|j| j.id == id).map(|j| j.to_json()) else {
        app.ui.templates.install_job = None;
        return;
    };
    if j["finished"] != json!(true) {
        ctx.request_repaint_after(std::time::Duration::from_millis(100));
        return;
    }
    app.ui.templates.install_job = None;
    app.ui.status = match j["result"]["error"].as_str() {
        Some(e) => tlf!("Install: {e}", e),
        None => {
            app.ui.templates.install.clear();
            tlf!("Installed {path}", path = j["result"]["path"].as_str().unwrap_or_default())
        }
    };
}

/// The QR code of `link` as a texture (cached by the link).
fn qr_texture(ctx: &egui::Context, link: &str) -> Option<egui::TextureHandle> {
    #[derive(Clone, Default)]
    struct Qr(String, Option<egui::TextureHandle>);
    let id = egui::Id::new("templates-panel-qr");
    if let Some(Qr(k, t)) = ctx.data(|d| d.get_temp::<Qr>(id))
        && k == link
    {
        return t;
    }
    let tex = vts::qr_code(link, filmcraft_templates::qr::EcLevel::M).ok().map(|q| {
        let (w, h, px) = q.to_rgba(4, 4);
        ctx.load_texture("template-share-qr", egui::ColorImage::from_rgba_unmultiplied([w as usize, h as usize], &px), egui::TextureOptions::NEAREST)
    });
    ctx.data_mut(|d| d.insert_temp(id, Qr(link.to_string(), tex.clone())));
    tex
}

/// The Save as Template form: what the template is called and who made it, its placeholders
/// (mark the selected clips), a captions slot, its layouts, the poster frame, the user's own
/// files to bundle; Save writes it into the user templates folder.
fn save_form(app: &mut FilmcraftApp, ui: &mut egui::Ui, rect: Rect, body: Rect, actions: &mut Vec<Action>) {
    let t = app.tokens;
    let slots = flagged(app);
    let own: Vec<(u64, String)> = app
        .session
        .active_sequence()
        .map(|q| {
            vt::own_media(&app.session, q).into_iter().map(|(i, _)| (i.0, app.session.project.item(i).map(|x| x.name.clone()).unwrap_or_default())).collect()
        })
        .unwrap_or_default();
    let selected = app.session.state.selection.len();
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("templates-save"));
    let ui = &mut child;
    egui::ScrollArea::vertical().id_salt("templates-save-scroll").auto_shrink([false, false]).show(ui, |ui| {
        ui.set_max_width(ui.available_width() - 6.0);
        let w = ui.available_width();
        ui.horizontal(|ui| {
            let b = ui.small_button(tl!("‹ All templates"));
            add(&mut app.auto, ui, "templates.save.cancel", b.rect, tl!("All templates"));
            if b.clicked() {
                actions.push(Action::CancelSave);
            }
            ui.label(egui::RichText::new(tl!("Save as Template")).color(t.text).size(14.0).strong());
        });
        small(ui, &t, tl!("Your clips become placeholders: the template never carries your footage, photos or words."));
        ui.add_space(6.0);
        let Some(sd) = app.ui.templates.save.as_mut() else { return };
        for (id, label, field, max) in [
            ("templates.save.name", tl!("Name"), &mut sd.name, 200),
            ("templates.save.description", tl!("Description"), &mut sd.description, 4000),
            ("templates.save.author", tl!("Author"), &mut sd.author, 200),
            ("templates.save.license", tl!("Licence (SPDX)"), &mut sd.license, 200),
            ("templates.save.category", tl!("Category"), &mut sd.category, 200),
        ] {
            ui.horizontal(|ui| {
                ui.add_sized(vec2(90.0, 18.0), egui::Label::new(egui::RichText::new(label).color(t.text_dim).size(11.5)));
                let r = ui.add(egui::TextEdit::singleline(field).desired_width((w - 100.0).max(60.0)).char_limit(max));
                add(&mut app.auto, ui, id, r.rect, label);
            });
        }
        ui.add_space(8.0);
        ui.label(egui::RichText::new(tl!("Placeholders")).color(t.text).size(12.0).strong());
        small(ui, &t, &tlf!("Mark the selected clips ({n}) as a slot:", n = selected));
        ui.horizontal_wrapped(|ui| {
            let r = ui.add(egui::TextEdit::singleline(&mut sd.slot).hint_text(tl!("slot id (optional)")).desired_width(120.0).char_limit(64));
            add(&mut app.auto, ui, "templates.save.slotId", r.rect, tl!("slot id (optional)"));
            for (k, _) in MARK_KINDS {
                let b = ui.add_enabled(selected > 0, egui::Button::new(kind_label(k)).small());
                add(&mut app.auto, ui, &format!("templates.save.mark.{}", k.id()), b.rect, kind_label(k));
                if b.clicked() {
                    actions.push(Action::Mark(Some(k)));
                }
            }
            let b = ui.add_enabled(selected > 0, egui::Button::new(tl!("Not a Slot")).small());
            add(&mut app.auto, ui, "templates.save.mark.clear", b.rect, tl!("Not a Slot"));
            if b.clicked() {
                actions.push(Action::Mark(None));
            }
        });
        if slots.is_empty() {
            small(ui, &t, tl!("No clip is a slot yet: select clips in the Timeline and mark them."));
        }
        for (id, (kind, n)) in &slots {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(format!("{} · {id} · {}", kind_label(*kind), tlf!("{n} clips", n))).color(t.text).size(11.5));
                let name = sd.slot_names.entry(id.clone()).or_default();
                let r = ui.add(egui::TextEdit::singleline(name).hint_text(tl!("name shown when filling")).desired_width((w - 220.0).max(60.0)).char_limit(200));
                add(&mut app.auto, ui, &format!("templates.save.slot.{id}.name"), r.rect, id);
            });
        }
        ui.add_space(8.0);
        let c = ui.checkbox(&mut sd.captions, tl!("Captions slot (transcribed from the video slots with sound)"));
        add(&mut app.auto, ui, "templates.save.captions", c.rect, tl!("Captions slot"));
        ui.horizontal_wrapped(|ui| {
            small(ui, &t, tl!("Layouts:"));
            for l in LAYOUTS {
                let mut on = sd.variants.iter().any(|v| v == l);
                let c = ui.checkbox(&mut on, l);
                add(&mut app.auto, ui, &format!("templates.save.variant.{l}"), c.rect, l);
                if c.changed() {
                    sd.variants.retain(|v| v != l);
                    if on {
                        sd.variants.push(l.to_string());
                    }
                }
            }
        });
        ui.horizontal(|ui| {
            let mut has = sd.poster_seconds.is_some();
            let c = ui.checkbox(&mut has, tl!("Poster frame at"));
            add(&mut app.auto, ui, "templates.save.posterOn", c.rect, tl!("Poster frame at"));
            if c.changed() {
                sd.poster_seconds = has.then_some(0.0);
            }
            if let Some(x) = sd.poster_seconds.as_mut() {
                let r = ui.add(egui::DragValue::new(x).range(0.0..=3600.0).speed(0.1).suffix(" s"));
                add(&mut app.auto, ui, "templates.save.poster", r.rect, tl!("Poster frame at"));
            }
        });
        if !own.is_empty() {
            ui.add_space(8.0);
            ui.label(egui::RichText::new(tl!("Your own files outside the slots")).color(t.text).size(12.0).strong());
            small(ui, &t, tl!("Left out unless you bundle them; a bundled file is marked as yours, under the template's licence."));
            for (item, name) in &own {
                let mut on = sd.bundle.contains(item);
                let c = ui.checkbox(&mut on, tlf!("Bundle “{name}”", name));
                add(&mut app.auto, ui, &format!("templates.save.bundle.{item}"), c.rect, name);
                if c.changed() {
                    sd.bundle.retain(|x| x != item);
                    if on {
                        sd.bundle.push(*item);
                    }
                }
            }
        }
        ui.add_space(6.0);
        let c = ui.checkbox(&mut sd.zip, tl!("Save as one .zip file"));
        add(&mut app.auto, ui, "templates.save.zip", c.rect, tl!("Save as one .zip file"));
    });
    // footer: Cancel | Save
    let fy = rect.max.y - FOOTER_H + 8.0;
    let half = (rect.width() - PAD * 2.0 - 8.0) / 2.0;
    let ready = app.ui.templates.save.as_ref().is_some_and(|s| !s.name.trim().is_empty()) && app.session.state.active_sequence.is_some();
    for (k, (bid, text)) in [("templates.save.close", tl!("Cancel")), ("templates.save.go", tl!("Save Template"))].into_iter().enumerate() {
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
        if b.clicked() && (!primary || ready) {
            actions.push(if primary { Action::Save } else { Action::CancelSave });
        }
    }
}

/// The open template's Share section: one `.zip` to pass on, its timeline as OpenTimelineIO, and
/// the QR code of the link it will be downloaded from.
fn share_section(app: &mut FilmcraftApp, ui: &mut egui::Ui, ctx: &egui::Context, w: f32, actions: &mut Vec<Action>) {
    let t = app.tokens;
    ui.add_space(8.0);
    ui.label(egui::RichText::new(tl!("Share")).color(t.text).size(12.0).strong());
    small(ui, &t, tl!("Templates are plain files: pass on the .zip, or put it online and share its link."));
    let d = &mut app.ui.templates;
    for (field, hint, bid, fid, label, action) in [
        (&mut d.export_path, tl!("Folder or .zip path"), "templates.share.export", "templates.share.exportPath", tl!("Export .zip"), Action::Export),
        (&mut d.otio_path, tl!(".otio path"), "templates.share.otio", "templates.share.otioPath", tl!("Export OpenTimelineIO"), Action::ExportOtio),
    ] {
        ui.horizontal(|ui| {
            let r = ui.add(egui::TextEdit::singleline(field).hint_text(hint).desired_width((w - 170.0).max(60.0)).char_limit(4096));
            add(&mut app.auto, ui, fid, r.rect, hint);
            let b = ui.add_enabled(!field.trim().is_empty(), egui::Button::new(label).small());
            add(&mut app.auto, ui, bid, b.rect, label);
            if b.clicked() {
                actions.push(action);
            }
        });
    }
    let r = ui.add(egui::TextEdit::singleline(&mut d.link).hint_text(tl!("https:// link to the .zip, for a QR code")).desired_width(w - 10.0).char_limit(2048));
    add(&mut app.auto, ui, "templates.share.link", r.rect, tl!("https:// link to the .zip, for a QR code"));
    let link = d.link.trim().to_string();
    if !link.is_empty() {
        match vts::check_url(&link) {
            Ok(()) => {
                if let Some(tex) = qr_texture(ctx, &link) {
                    let side = (w * 0.5).clamp(96.0, 220.0);
                    let (r, _) = ui.allocate_exact_size(vec2(side, side), Sense::hover());
                    ui.painter().image(tex.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), egui::Color32::WHITE);
                    add(&mut app.auto, ui, "templates.share.qr", r, tl!("QR code of the install link"));
                    small(ui, &t, tl!("Scan to download; open the file with Templates ▸ Install."));
                }
            }
            Err(e) => small(ui, &t, &e),
        }
    }
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
    if !lib.read || lib.revision != vt::library_revision() {
        if lib.read {
            // a template was saved or installed: the open one is read anew, and the textures
            // of what changed are drawn anew
            let mut tx = ctx.data(|d| d.get_temp::<Textures>(tex_id())).unwrap_or_default();
            tx.library_changed(&app.session);
            ctx.data_mut(|d| {
                d.insert_temp(tex_id(), tx);
                d.remove::<Selected>(sel_id());
            });
        }
        lib = read_library(app);
        ctx.data_mut(|d| d.insert_temp(lib_id(), lib.clone()));
    }
    poll_install(app, &ctx);
    let mut actions: Vec<Action> = Vec::new();
    let mut budget = THUMBS_PER_FRAME;
    let mut missing = false;
    let body = Rect::from_min_max(pos2(rect.min.x + PAD, rect.min.y + 8.0), pos2(rect.max.x - PAD, rect.max.y - FOOTER_H));
    if app.ui.templates.save.is_some() {
        save_form(app, ui, rect, body, &mut actions);
        for a in actions {
            run(app, &ctx, a);
        }
        return;
    }
    match app.ui.templates.selected.clone() {
        None => {
            // the filters
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(body).id_salt("templates-grid"));
            let ui = &mut child;
            let w = ui.available_width();
            // authoring and sharing: save the open sequence, install a package
            ui.horizontal(|ui| {
                let has_seq = app.session.state.active_sequence.is_some();
                let b = ui.add_enabled(has_seq, egui::Button::new(tl!("Save as Template…")));
                add(&mut app.auto, ui, "templates.saveAs", b.rect, tl!("Save as Template…"));
                if b.on_hover_text(tl!("Make a template of the open sequence: mark its placeholders, choose what travels with it")).clicked() {
                    actions.push(Action::OpenSave);
                }
                let d = &mut app.ui.templates;
                let r = ui.add(
                    egui::TextEdit::singleline(&mut d.install)
                        .hint_text(tl!("A .zip package or an https:// link"))
                        .desired_width((w - 220.0).max(60.0))
                        .char_limit(2048),
                );
                add(&mut app.auto, ui, "templates.install.path", r.rect, tl!("A .zip package or an https:// link"));
                let ready = !d.install.trim().is_empty() && d.install_job.is_none();
                let b = ui.add_enabled(ready, egui::Button::new(tl!("Install")));
                add(&mut app.auto, ui, "templates.install.go", b.rect, tl!("Install"));
                if b.on_hover_text(tl!("Check the package and add it to your templates; nothing in it is run")).clicked() {
                    actions.push(Action::Install);
                }
            });
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
                share_section(app, ui, &ctx, w, &mut actions);
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
                let (fs, warnings) = frames::preview(&app.session, &id, p["variant"].as_str(), fills, PREVIEW_FRAMES, 240)?;
                Ok((fs, warnings, Arc::new(e.package)))
            });
            match r {
                Ok((fs, warnings, package)) => {
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
                        tx.preview_of = Some((id, package));
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
        Action::OpenSave => start_save(app),
        Action::CancelSave => app.ui.templates.save = None,
        Action::Mark(kind) => {
            let slot = app.ui.templates.save.as_ref().map(|s| s.slot.trim().to_string()).unwrap_or_default();
            let mut p = match kind {
                Some(k) => json!({"kind": k.id()}),
                None => json!({"clear": true}),
            };
            if kind.is_some() && !slot.is_empty() {
                p["slot"] = json!(slot);
            }
            app.ui.status = match app.session.execute("template.markSlot", p) {
                Ok(v) if v["slot"].is_null() => tlf!("{n} clips are no slot now", n = v["clips"]),
                Ok(v) => tlf!("{n} clips are now slot “{slot}”", n = v["clips"], slot = v["slot"].as_str().unwrap_or_default()),
                Err(e) => e.to_string(),
            };
        }
        Action::Save => {
            let Some(sd) = app.ui.templates.save.clone() else { return };
            match app.session.execute("template.save", sd.params()) {
                Ok(r) => {
                    let name = r["template"]["name"].as_str().unwrap_or_default().to_string();
                    let warn = r["warnings"].as_array().map_or(0, Vec::len);
                    app.ui.status = if warn == 0 {
                        tlf!("Saved “{name}” to your templates", name)
                    } else {
                        tlf!("Saved “{name}” to your templates ({n} notes: template.save lists them)", name, n = warn)
                    };
                    app.ui.templates.save = None;
                    // open it: the user sees what was saved
                    if let Some(id) = r["template"]["id"].as_str() {
                        app.ui.templates.selected = Some(id.to_string());
                        app.ui.templates.variant = r["template"]["defaultVariant"].as_str().unwrap_or_default().to_string();
                        app.ui.templates.fills.clear();
                        app.ui.templates.built = None;
                    }
                }
                Err(e) => app.ui.status = tlf!("Save as Template: {e}", e),
            }
        }
        Action::Install => {
            let what = app.ui.templates.install.trim().to_string();
            let p = if what.starts_with("https://") || what.starts_with("http://") { json!({"url": what}) } else { json!({"path": what}) };
            match app.session.execute("template.install", p) {
                Ok(r) if r["job"].is_u64() => {
                    app.ui.templates.install_job = r["job"].as_u64();
                    app.ui.status = tl!("Downloading the template…").into();
                }
                Ok(r) => {
                    app.ui.templates.install.clear();
                    app.ui.status = tlf!("Installed “{name}”", name = r["name"].as_str().unwrap_or_default());
                }
                Err(e) => app.ui.status = tlf!("Install: {e}", e),
            }
        }
        Action::Export | Action::ExportOtio => {
            let Some(id) = app.ui.templates.selected.clone() else { return };
            let otio = matches!(a, Action::ExportOtio);
            let d = &app.ui.templates;
            let p = if otio {
                json!({"template": id, "path": d.otio_path.trim(), "variant": d.variant})
            } else {
                json!({"template": id, "path": d.export_path.trim()})
            };
            match app.session.execute(if otio { "template.exportOtio" } else { "template.export" }, p) {
                Ok(r) => app.ui.status = tlf!("Wrote {path}", path = r["path"].as_str().unwrap_or_default()),
                Err(e) => app.ui.status = e.to_string(),
            }
        }
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
