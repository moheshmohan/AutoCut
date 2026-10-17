//! Effects panel ▸ Elements: the bundled Elements library (`filmcraft_engine::elements`) in three
//! folders — Sounds, Stickers, Animated Text. Double-click a row (or right-click ▸ Insert at
//! Playhead) to put the element at the playhead on a free track (`elements.insert`, the command
//! agents and the CLI use). Sticker and text rows show a thumbnail of the design.
//!
//! Automation ids: `effects.folder.Elements`, `effects.folder.Elements/<Sounds|Stickers|Animated
//! Text>`, `effects.element.<element id>` (e.g. `effects.element.sound.pop`),
//! `effects.elementMenu.insert`.

use std::sync::OnceLock;

use egui::{Align2, Color32, Rect, Sense, pos2, vec2};
use filmcraft_engine::elements::{self, Kind};
use filmcraft_project::elements as gfx;
use filmcraft_project::gtemplate::GraphicsTemplate;
use serde_json::json;

use crate::FilmcraftApp;
use crate::theme::Tokens;

/// The top folder's name.
pub const FOLDER: &str = "Elements";

/// One row: element id, name, description, kind, and the design of a sticker or text preset.
struct Row {
    id: String,
    name: String,
    description: String,
    kind: Kind,
    template: Option<GraphicsTemplate>,
}

/// The library's rows matching `filter` (lower case). The library is built once.
fn rows(filter: &str) -> Vec<&'static Row> {
    static ALL: OnceLock<Vec<Row>> = OnceLock::new();
    let all = ALL.get_or_init(|| {
        let mut v: Vec<Row> = elements::SOUNDS
            .iter()
            .map(|s| Row { id: s.id.into(), name: s.name.into(), description: s.description.into(), kind: Kind::Sound, template: None })
            .collect();
        v.extend(gfx::all().into_iter().map(|t| Row {
            id: gfx::element_id(&t).to_string(),
            name: t.name.clone(),
            description: t.description.clone(),
            kind: if t.category == gfx::STICKERS { Kind::Sticker } else { Kind::Text },
            template: Some(t),
        }));
        v
    });
    all.iter().filter(|r| crate::i18n::matches_query(&r.name, filter)).collect()
}

/// Thumbnail size (twice the row's picture, for sharp edges on high-DPI screens).
const THUMB: (usize, usize) = (64, 36);

/// A graphic element cropped to what it shows a moment in (stickers are small on their canvas):
/// straight sRGB RGBA8, [`THUMB`] in size.
fn render_thumb(t: &GraphicsTemplate) -> Vec<u8> {
    use filmcraft_render::graphic_clip::{graphic_specs, layer_bounds, render_graphic_layers};
    let canvas = (t.canvas[0].max(1), t.canvas[1].max(1));
    let at = filmcraft_time::Tick((t.duration.0 / 2).clamp(0, filmcraft_time::TICKS_PER_SECOND * 3 / 2));
    let mut b = [f64::MAX, f64::MAX, f64::MIN, f64::MIN];
    for s in graphic_specs(&t.layers, at, canvas).iter().filter(|s| s.enabled && s.transform.opacity > 0.0) {
        let l = layer_bounds(s);
        b = [b[0].min(l[0]), b[1].min(l[1]), b[2].max(l[2]), b[3].max(l[3])];
    }
    if !(b[0] < b[2] && b[1] < b[3]) || b.iter().any(|v| !v.is_finite()) {
        b = [0.0, 0.0, f64::from(canvas.0), f64::from(canvas.1)];
    }
    let (w, h) = (THUMB.0 as f64, THUMB.1 as f64);
    let k = ((w - 4.0) / (b[2] - b[0]) * 0.9).min((h - 4.0) / (b[3] - b[1]) * 0.9).min(1.0);
    let (cx, cy) = ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0);
    let m = filmcraft_geom::Affine::translate(w / 2.0, h / 2.0)
        .then_apply(&filmcraft_geom::Affine::scale(k, k))
        .then_apply(&filmcraft_geom::Affine::translate(-cx, -cy));
    let mut img = filmcraft_render::image::Image::new(THUMB.0, THUMB.1);
    render_graphic_layers(&t.layers, at, canvas, &m, &mut img);
    img.to_rgba8()
}

/// A graphic element's thumbnail texture (rendered once per element, kept by name).
fn thumb(app: &mut FilmcraftApp, ctx: &egui::Context, t: &GraphicsTemplate) -> egui::TextureId {
    let name = format!("element-thumb-{}", t.id);
    if let Some((id, _)) = app.texture_existing(&name) {
        return id;
    }
    let rgba = crate::frames::Rgba { w: THUMB.0, h: THUMB.1, px: render_thumb(t) };
    let key =
        crate::frames::FrameKey { target: crate::frames::Target::Item(filmcraft_project::ItemId(u64::MAX - 1)), frame: 0, size: 64, revision: 0, draft: false };
    app.texture_for(ctx, &name, key, &rgba)
}

/// Right-click menu of a row: Insert at Playhead. True when chosen.
fn insert_menu(app: &mut FilmcraftApp, resp: &egui::Response) -> bool {
    let mut chosen = false;
    resp.context_menu(|ui| {
        let b = ui.button(tl!("Insert at Playhead"));
        app.auto.add("effects.elementMenu.insert", b.rect, "Insert at Playhead");
        if b.clicked() {
            chosen = true;
            ui.close();
        }
    });
    chosen
}

/// The Elements folder of the Effects tree. Inserts the element a row asks for and reports how
/// it went in the status bar.
pub fn folder(app: &mut FilmcraftApp, ui: &mut egui::Ui, filter: &str) {
    let all = rows(filter);
    if !filter.is_empty() && all.is_empty() {
        return;
    }
    if !super::effects::folder_row(app, ui, FOLDER, 0, filter) {
        return;
    }
    let t = app.tokens;
    let mut insert: Option<String> = None;
    for kind in Kind::ALL {
        let items: Vec<&Row> = all.iter().copied().filter(|r| r.kind == kind).collect();
        if items.is_empty() || !super::effects::folder_row(app, ui, &format!("{FOLDER}/{}", kind.category()), 1, filter) {
            continue;
        }
        for r in items {
            let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 24.0), Sense::click());
            if resp.hovered() {
                ui.painter().rect_filled(rect, 0.0, t.hover);
            }
            let tr = Rect::from_min_size(pos2(rect.min.x + 40.0, rect.min.y + 3.0), vec2(32.0, 18.0));
            ui.painter().rect_filled(tr, 2.0, t.field_bg);
            match &r.template {
                Some(tpl) => {
                    let tex = thumb(app, ui.ctx(), tpl);
                    ui.painter().image(tex, tr, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
                None => {
                    // a sound: a few bars of a level meter
                    for (k, hgt) in [0.35f32, 0.8, 0.55, 1.0, 0.45].iter().enumerate() {
                        let x = tr.min.x + 7.0 + k as f32 * 4.5;
                        let hh = (tr.height() - 6.0) * hgt;
                        ui.painter().line_segment([pos2(x, tr.center().y - hh / 2.0), pos2(x, tr.center().y + hh / 2.0)], egui::Stroke::new(2.0, t.text_dim));
                    }
                }
            }
            ui.painter().text(pos2(tr.max.x + 8.0, rect.center().y), Align2::LEFT_CENTER, crate::i18n::t(&r.name), Tokens::ui(12.0), t.text);
            app.auto.add(&format!("effects.element.{}", r.id), rect, &r.name);
            let resp = resp.on_hover_text(crate::i18n::t(&r.description));
            if insert_menu(app, &resp) || resp.double_clicked() {
                insert = Some(r.id.clone());
            }
        }
    }
    if let Some(id) = insert {
        app.ui.status = match app.session.execute("elements.insert", json!({"id": id})) {
            Ok(r) => tlf!("Added {name} on {track}", name = r["name"].as_str().unwrap_or_default(), track = r["track"].as_str().unwrap_or_default()),
            Err(e) => e.to_string(),
        };
    }
}
