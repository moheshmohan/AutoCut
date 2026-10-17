//! Video templates (`.fcvideo-template`): whole-sequence starters with media slots — the format,
//! validation, slot fit, beat grids and built-in starters live in `filmcraft-templates`; this
//! module is their commands.
//!
//! | command | does |
//! |---|---|
//! | `template.list` | the built-in starters and the user's templates (`<data dir>/Video Templates/`: `*.fcvideo-template` folders and `.zip` packages); `query`, `category`, `source` filter; templates that fail to read are listed under `invalid` with why |
//! | `template.inspect` | one template (`template`: id, name or path): slots with their placements, variants, beat grid, look, caption style, export preset, the manifest |
//! | `template.validate` | check a package (`path`: folder, `.zip` or `template.json`): `valid`, and every problem with the field it is about |
//! | `template.apply` | a new sequence from a template: `variant` (`16:9`, `9:16`…), `slots` (`{"photos": ["/a.jpg", 12, {"path": "/b.mov", "inSeconds": 3, "focus": [0.4, 0.5]}], "title": "Lisbon"}`), `name`; one undo step |
//! | `template.save` | the minimal "save as template": clips flagged as slots (`template.markSlot`, or left by an earlier apply) become slots with generated placeholders; other personal media is left out; writes a folder or a `.zip` |
//! | `template.markSlot` | flag clips as placeholders of a slot (`kind`, `slot`) for `template.save`, or `clear` the flag |
//!
//! **Applying.** Files named in `slots` are imported (a file the project already holds is reused,
//! and one file named for several placements is one item). A filled music slot is measured
//! (`media.beats`' analysis) and the cuts re-quantized to its beats; an unfilled one with a
//! `generate` default gets a bed made with `audio.generate` at the template's tempo and length.
//! Music shorter than the show is Remixed to length (looped when Remix cannot). The template's
//! look goes on the picture slots (`lumetri.applyPreset` / `presets.apply`), its caption style on
//! the caption tracks, and its export preset (the variant's) becomes Quick Export's preset.

use std::collections::BTreeMap;

use filmcraft_project::{ClipId, ItemId, Label, Marker, MarkerId, MarkerKind, TrackKind};
use filmcraft_templates::apply::{self as tapply, ApplyOptions, MediaFill, MusicBeats, SlotFill, parse_slot_marker, slot_marker_name};
use filmcraft_templates::manifest::{Look, MAX_TEMPLATE_TICKS, Manifest, Slot, SlotDefault, SlotKind};
use filmcraft_templates::{EXTENSION, MANIFEST_FILE, Package, TemplateError};
use filmcraft_time::Tick;
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, f64_p, has_seq, str_p};
use crate::{EngineError, Result, Session};

/// The folder of user templates under the data directory.
pub const USER_FOLDER: &str = "Video Templates";
/// Most files read from a template folder.
const MAX_PACKAGE_FILES: usize = 1024;
/// Most bytes read from a template folder or archive.
const MAX_PACKAGE_BYTES: u64 = 256 * 1024 * 1024;
/// Deepest folder walked inside a template folder.
const MAX_DEPTH: usize = 16;
/// Most media named in one `template.apply`.
const MAX_FILL_MEDIA: usize = 1024;

type Run = fn(&mut Session, &Value) -> Result<Value>;

fn spec(
    id: &'static str,
    label: &'static str,
    params: &'static str,
    enabled: fn(&Session) -> std::result::Result<(), String>,
    run: Run,
    journal: bool,
) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec("template.list", "List Video Templates", r#"{"query":str?,"category":str?,"source":"builtin"|"user"?}"#, always, list, false),
        spec("template.inspect", "Inspect Video Template", r#"{"template":id|name|path}"#, always, inspect, false),
        spec(
            "template.validate",
            "Validate Video Template",
            r#"{"path":str (a .fcvideo-template folder, a .zip or a template.json)}"#,
            always,
            validate,
            false,
        ),
        spec(
            "template.apply",
            "Apply Video Template",
            r#"{"template":id|name|path,"variant":"16:9"|"9:16"|…?,"slots":{"<slot>": path|itemId|text|{"path"|"item"|"text","in":ticks?,"inSeconds":f64?,"focus":[x,y]?,"focusByVariant":{"9:16":[x,y]}?}|[…]}?,"name":str?,"sequence":str (name)?,"requireAll":bool?,"generateMusic":bool=true,"look":bool=true}"#,
            always,
            apply,
            true,
        ),
        spec(
            "template.save",
            "Save Sequence as Video Template",
            r#"{"path":str (.fcvideo-template folder or .zip),"sequence":id?,"name":str?,"id":str?,"description":str?,"author":str?,"license":str="MIT OR Apache-2.0","category":str?,"overwrite":bool?}"#,
            always,
            save,
            false,
        ),
        spec(
            "template.markSlot",
            "Mark Clips as Template Slot",
            r#"{"clips":[id]?,"kind":"video"|"photo"|"text"|"logo"|"music","slot":str?,"clear":bool?}"#,
            has_seq,
            mark_slot,
            true,
        ),
    ]
}

// ------------------------------------------------------------------------------------------------
// The library
// ------------------------------------------------------------------------------------------------

/// Where user templates live: `<data dir>/Video Templates` (None without a data directory).
pub fn user_dir(s: &Session) -> Option<std::path::PathBuf> {
    s.prefs_path.as_ref().and_then(|p| p.parent()).map(|d| d.join(USER_FOLDER))
}

/// A template from the library or a file.
#[derive(Clone, Debug)]
pub struct Entry {
    pub package: Package,
    /// `builtin`, `user` or `file`.
    pub source: &'static str,
    pub path: Option<String>,
    /// The folder the package's `assets/` are in, on disk.
    pub asset_dir: Option<String>,
}

fn join(dir: &str, name: &str) -> String {
    if dir.is_empty() {
        return name.to_string();
    }
    let sep = if dir.ends_with('/') || dir.ends_with('\\') { "" } else { "/" };
    format!("{dir}{sep}{name}")
}

/// Every file of a package folder (relative path → bytes), bounded.
fn read_folder(s: &Session, dir: &str) -> std::result::Result<BTreeMap<String, Vec<u8>>, String> {
    let mut out = BTreeMap::new();
    let mut total: u64 = 0;
    let mut stack: Vec<(String, String, usize)> = vec![(dir.to_string(), String::new(), 0)];
    while let Some((abs, rel, depth)) = stack.pop() {
        let entries = match s.services.list_entries(&abs) {
            Some(Ok(e)) => e,
            Some(Err(e)) => return Err(format!("{abs}: {e}")),
            None => return Err("this host cannot read folders; use a .zip package".into()),
        };
        for e in entries {
            if e.name.starts_with('.') {
                continue;
            }
            let a = join(&abs, &e.name);
            let r = if rel.is_empty() { e.name.clone() } else { format!("{rel}/{}", e.name) };
            if e.is_dir {
                if depth + 1 > MAX_DEPTH {
                    return Err(format!("{a}: folders nest deeper than {MAX_DEPTH}"));
                }
                stack.push((a, r, depth + 1));
                continue;
            }
            if out.len() >= MAX_PACKAGE_FILES {
                return Err(format!("{dir}: more than {MAX_PACKAGE_FILES} files"));
            }
            let size = s.services.file_size(&a).map_err(|e| format!("{a}: {e}"))?;
            total = total.saturating_add(size);
            if total > MAX_PACKAGE_BYTES {
                return Err(format!("{dir}: more than {MAX_PACKAGE_BYTES} bytes"));
            }
            let b = s.services.read_file(&a).map_err(|e| format!("{a}: {e}"))?;
            out.insert(r, b);
        }
    }
    Ok(out)
}

fn is_dir(s: &Session, path: &str) -> bool {
    s.services.file_size(path).is_err() && s.services.list_entries(path).is_some_and(|r| r.is_ok())
}

/// Read a package from `path`: a folder, a `.zip`, or the `template.json` inside a folder.
pub fn read_package(s: &Session, path: &str) -> std::result::Result<Entry, TemplateError> {
    let path = crate::export_tools::expand_home(path);
    let p = path.trim_end_matches(['/', '\\']).to_string();
    let folder = if p.ends_with(MANIFEST_FILE) {
        Some(p.trim_end_matches(MANIFEST_FILE).trim_end_matches(['/', '\\']).to_string())
    } else if is_dir(s, &p) {
        Some(p.clone())
    } else {
        None
    };
    match folder {
        Some(dir) => {
            let files = read_folder(s, &dir).map_err(TemplateError::Package)?;
            let package = Package::from_files(files)?;
            Ok(Entry { package, source: "file", path: Some(dir.clone()), asset_dir: Some(dir) })
        }
        None => {
            let size = s.services.file_size(&p).map_err(|e| TemplateError::Package(format!("{p}: {e}")))?;
            if size > MAX_PACKAGE_BYTES {
                return Err(TemplateError::Package(format!("{p}: {size} bytes (at most {MAX_PACKAGE_BYTES})")));
            }
            let bytes = s.services.read_file(&p).map_err(|e| TemplateError::Package(format!("{p}: {e}")))?;
            if !filmcraft_templates::zip::is_zip(&bytes) {
                return Err(TemplateError::NotTemplate(format!("{p} is neither a .{EXTENSION} folder nor a ZIP package")));
            }
            let package = Package::from_zip(&bytes)?;
            Ok(Entry { package, source: "file", path: Some(p), asset_dir: None })
        }
    }
}

/// Built-in and user templates, and the user files that could not be read (path, why).
pub fn library(s: &Session) -> (Vec<Entry>, Vec<(String, String)>) {
    let mut v: Vec<Entry> = filmcraft_templates::builtin::builtin_templates()
        .into_iter()
        .map(|package| Entry { package, source: "builtin", path: None, asset_dir: None })
        .collect();
    let mut invalid = Vec::new();
    if let Some(dir) = user_dir(s) {
        let dir = dir.to_string_lossy().to_string();
        if let Some(Ok(mut entries)) = s.services.list_entries(&dir) {
            entries.sort_by(|a, b| a.name.cmp(&b.name));
            for e in entries.into_iter().take(MAX_PACKAGE_FILES) {
                let lower = e.name.to_ascii_lowercase();
                if e.name.starts_with('.') || !(lower.ends_with(&format!(".{EXTENSION}")) || lower.ends_with(".zip")) {
                    continue;
                }
                let path = join(&dir, &e.name);
                match read_package(s, &path) {
                    Ok(mut x) => {
                        x.source = "user";
                        v.push(x);
                    }
                    Err(err) => invalid.push((path, err.to_string())),
                }
            }
        }
    }
    (v, invalid)
}

/// Whether a template key is a path (else an id or a name).
fn looks_like_path(key: &str) -> bool {
    let lower = key.to_ascii_lowercase();
    key.contains('/')
        || key.contains('\\')
        || key.starts_with('~')
        || lower.ends_with(".zip")
        || lower.ends_with(&format!(".{EXTENSION}"))
        || lower.ends_with(".json")
}

/// A template by id, name (case-insensitive) or path.
pub fn find_template(s: &Session, key: &str, cmd: &str) -> Result<Entry> {
    let key = key.trim();
    if key.is_empty() {
        return Err(bad(cmd, "need `template` (an id, a name or a path; see template.list)"));
    }
    if looks_like_path(key) {
        return read_package(s, key).map_err(|e| EngineError::Other(format!("{key}: {e}")));
    }
    let builtin = filmcraft_templates::builtin::builtin_templates();
    if let Some(p) = builtin.iter().find(|p| p.manifest.id == key || p.manifest.name.eq_ignore_ascii_case(key)) {
        return Ok(Entry { package: p.clone(), source: "builtin", path: None, asset_dir: None });
    }
    let (lib, _) = library(s);
    lib.into_iter()
        .find(|e| e.package.manifest.id == key)
        .or_else(|| library(s).0.into_iter().find(|e| e.package.manifest.name.eq_ignore_ascii_case(key)))
        .ok_or_else(|| bad(cmd, format!("no video template `{key}` (see template.list)")))
}

// ------------------------------------------------------------------------------------------------
// Queries
// ------------------------------------------------------------------------------------------------

fn slot_json(s: &Slot) -> Value {
    json!({
        "id": s.id, "kind": s.kind.id(), "name": s.name, "prompt": s.prompt, "order": s.order, "count": s.count, "minCount": s.min_count,
        "required": s.min_count > 0, "beatSync": s.beat_sync, "placements": s.placements.len(),
        "fit": {"short": s.fit.short.id(), "portion": format!("{:?}", s.fit.portion).to_ascii_lowercase(), "frame": s.fit.frame.id(), "minSpeed": s.fit.min_speed},
        "constraints": serde_json::to_value(&s.constraints).unwrap_or_default(),
        "default": s.default.as_ref().map(|d| serde_json::to_value(d).unwrap_or_default()),
    })
}

fn look_json(m: &Manifest) -> Value {
    match &m.look {
        Some(Look::Lumetri(n)) => json!({"lumetri": n}),
        Some(Look::Preset(n)) => json!({"preset": n}),
        None => Value::Null,
    }
}

fn summary(e: &Entry) -> Value {
    let m = &e.package.manifest;
    json!({
        "id": m.id, "name": m.name, "description": m.description, "version": m.version, "author": m.author, "license": m.license,
        "category": m.category, "tags": m.tags, "source": e.source, "path": e.path,
        "duration": m.duration.0, "seconds": m.duration.seconds(),
        "variants": m.variants.iter().map(|v| json!({"id": v.id, "name": v.name, "width": v.width, "height": v.height, "exportPreset": v.export_preset.as_ref().or(m.export_preset.as_ref())})).collect::<Vec<_>>(),
        "defaultVariant": m.default_variant,
        "slots": m.slots.iter().map(slot_json).collect::<Vec<_>>(),
        "look": look_json(m),
        "exportPreset": m.export_preset,
        "beatGrid": m.beat_grid.as_ref().map(|g| json!({"bpm": g.bpm, "beats": g.beats.len(), "mode": format!("{:?}", g.mode).to_ascii_lowercase()})),
        "assets": m.assets.len(),
    })
}

fn list(s: &mut Session, p: &Value) -> Result<Value> {
    let q = str_p(p, "query").unwrap_or("").to_ascii_lowercase();
    let cat = str_p(p, "category").map(str::to_ascii_lowercase);
    let src = str_p(p, "source");
    let (lib, invalid) = library(s);
    let v: Vec<Value> = lib
        .iter()
        .filter(|e| src.is_none_or(|x| x.eq_ignore_ascii_case(e.source)))
        .filter(|e| cat.as_deref().is_none_or(|c| e.package.manifest.category.to_ascii_lowercase() == c))
        .filter(|e| {
            let m = &e.package.manifest;
            q.is_empty()
                || m.name.to_ascii_lowercase().contains(&q)
                || m.description.to_ascii_lowercase().contains(&q)
                || m.id.contains(&q)
                || m.tags.iter().any(|t| t.to_ascii_lowercase().contains(&q))
        })
        .map(summary)
        .collect();
    Ok(json!({
        "templates": v,
        "invalid": invalid.iter().map(|(p, e)| json!({"path": p, "error": e})).collect::<Vec<_>>(),
        "userFolder": user_dir(s).map(|d| d.to_string_lossy().to_string()),
    }))
}

fn inspect(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.inspect";
    let key = str_p(p, "template").or_else(|| str_p(p, "id")).or_else(|| str_p(p, "path")).unwrap_or("");
    let e = find_template(s, key, CMD)?;
    let mut v = summary(&e);
    let seq = e.package.sequence();
    let clip_of = |id: u64| seq.and_then(|q| q.all_tracks().find_map(|t| t.items.iter().find(|c| c.id.0 == id).map(|c| (t.kind, c))));
    if let Some(slots) = v.get_mut("slots").and_then(Value::as_array_mut) {
        for (sv, s) in slots.iter_mut().zip(&e.package.manifest.slots) {
            sv["placements"] = Value::Array(
                s.placements
                    .iter()
                    .map(|pl| {
                        let c = clip_of(pl.clip);
                        json!({
                            "clip": pl.clip, "media": pl.media, "layer": pl.layer,
                            "track": c.map(|(k, _)| if k == TrackKind::Video { "video" } else { "audio" }),
                            "start": c.map(|(_, c)| c.start.0), "duration": c.map(|(_, c)| c.duration.0),
                            "seconds": c.map(|(_, c)| c.duration.seconds()), "placeholder": c.map(|(_, c)| c.name.clone()),
                        })
                    })
                    .collect(),
            );
        }
    }
    v["captionStyle"] = e.package.manifest.caption_style.as_ref().map(|c| serde_json::to_value(c).unwrap_or_default()).unwrap_or(Value::Null);
    v["manifest"] = serde_json::to_value(&e.package.manifest).unwrap_or_default();
    v["checks"] = Value::Array(engine_checks(s, &e.package.manifest).iter().map(|(f, m)| json!({"field": f, "message": m})).collect());
    Ok(v)
}

/// What the format cannot check alone: the look, the export presets.
fn engine_checks(s: &Session, m: &Manifest) -> Vec<(String, String)> {
    let mut out = Vec::new();
    match &m.look {
        Some(Look::Lumetri(n)) if filmcraft_render::lumetri_presets::find(n).is_none() => {
            out.push(("look.lumetri".into(), format!("no Lumetri preset `{n}` (see lumetri.presets)")));
        }
        Some(Look::Preset(n)) if s.presets.find(n).is_none() => out.push(("look.preset".into(), format!("no effect preset `{n}` (see presets.list)"))),
        _ => {}
    }
    if let Some(e) = &m.export_preset
        && s.export_presets.find(e).is_none()
    {
        out.push(("exportPreset".into(), format!("no export preset `{e}` (see export.presets.list)")));
    }
    for (i, v) in m.variants.iter().enumerate() {
        if let Some(e) = &v.export_preset
            && s.export_presets.find(e).is_none()
        {
            out.push((format!("variants[{i}].exportPreset"), format!("no export preset `{e}` (see export.presets.list)")));
        }
    }
    out
}

fn validate(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.validate";
    let path = str_p(p, "path").or_else(|| str_p(p, "template")).ok_or_else(|| bad(CMD, "need `path`"))?;
    let e = if looks_like_path(path) { read_package(s, path) } else { find_template(s, path, CMD).map_err(|e| TemplateError::Package(e.to_string())) };
    Ok(match e {
        Ok(e) => {
            let checks = engine_checks(s, &e.package.manifest);
            json!({
                "valid": checks.is_empty(),
                "errors": checks.iter().map(|(f, m)| json!({"field": f, "message": m})).collect::<Vec<_>>(),
                "template": summary(&e),
            })
        }
        Err(err) => json!({
            "valid": false,
            "errors": err.issues().iter().map(|i| json!({"field": i.field, "message": i.message})).collect::<Vec<_>>(),
            "template": Value::Null,
        }),
    })
}

// ------------------------------------------------------------------------------------------------
// Apply
// ------------------------------------------------------------------------------------------------

/// One entry of a slot fill, before importing.
enum Given {
    Path { path: String, source_in: Option<Tick>, focus: Option<[f64; 2]>, by_variant: BTreeMap<String, [f64; 2]> },
    Item { item: ItemId, source_in: Option<Tick>, focus: Option<[f64; 2]>, by_variant: BTreeMap<String, [f64; 2]> },
    Text(String),
}

fn pair(v: &Value) -> Option<[f64; 2]> {
    let a = v.as_array().filter(|a| a.len() == 2)?;
    Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?])
}

fn given(v: &Value, slot: &Slot, cmd: &str) -> Result<Given> {
    const MAX_STR: usize = 4096;
    let text_slot = slot.kind == SlotKind::Text;
    match v {
        Value::String(t) if t.len() > MAX_STR => Err(bad(cmd, format!("slot `{}`: a value is longer than {MAX_STR} bytes", slot.id))),
        Value::String(t) if text_slot => Ok(Given::Text(t.clone())),
        Value::String(path) => Ok(Given::Path { path: path.clone(), source_in: None, focus: None, by_variant: BTreeMap::new() }),
        Value::Number(n) if !text_slot => {
            let id = n.as_u64().ok_or_else(|| bad(cmd, format!("slot `{}`: item ids are whole numbers", slot.id)))?;
            Ok(Given::Item { item: ItemId(id), source_in: None, focus: None, by_variant: BTreeMap::new() })
        }
        Value::Object(o) => {
            if text_slot {
                return match o.get("text").and_then(Value::as_str) {
                    Some(t) if t.len() <= MAX_STR => Ok(Given::Text(t.to_string())),
                    _ => Err(bad(cmd, format!("slot `{}` takes text: a string or {{\"text\": …}}", slot.id))),
                };
            }
            let source_in = match (o.get("in").and_then(Value::as_i64), f64_p(v, "inSeconds")) {
                (Some(t), _) => Some(Tick(t)),
                (None, Some(sec)) if sec.is_finite() && (0.0..=86_400.0).contains(&sec) => Some(Tick::from_seconds_f64(sec)),
                (None, Some(_)) => return Err(bad(cmd, format!("slot `{}`: `inSeconds` must be from 0 to 86400", slot.id))),
                (None, None) => None,
            };
            if source_in.is_some_and(|t| t < Tick::ZERO || t.0 > MAX_TEMPLATE_TICKS * 24) {
                return Err(bad(cmd, format!("slot `{}`: the in point is out of range", slot.id)));
            }
            let focus = match o.get("focus") {
                Some(f) => Some(pair(f).ok_or_else(|| bad(cmd, format!("slot `{}`: `focus` is [x, y] (fractions of the picture)", slot.id)))?),
                None => None,
            };
            let mut by_variant = BTreeMap::new();
            if let Some(fv) = o.get("focusByVariant") {
                let m = fv.as_object().ok_or_else(|| bad(cmd, format!("slot `{}`: `focusByVariant` is {{\"9:16\": [x, y]}}", slot.id)))?;
                for (k, f) in m.iter().take(16) {
                    by_variant.insert(k.clone(), pair(f).ok_or_else(|| bad(cmd, format!("slot `{}`: focusByVariant `{k}` is [x, y]", slot.id)))?);
                }
            }
            match (o.get("path").and_then(Value::as_str), o.get("item").and_then(Value::as_u64)) {
                (Some(path), None) if path.len() <= MAX_STR => Ok(Given::Path { path: path.to_string(), source_in, focus, by_variant }),
                (None, Some(item)) => Ok(Given::Item { item: ItemId(item), source_in, focus, by_variant }),
                _ => Err(bad(cmd, format!("slot `{}`: give `path` (a file) or `item` (a project item id)", slot.id))),
            }
        }
        _ if text_slot => Err(bad(cmd, format!("slot `{}` takes text", slot.id))),
        _ => Err(bad(cmd, format!("slot `{}` takes a file path, an item id or {{\"path\": …}}", slot.id))),
    }
}

/// Import a file once per call (a file the project already holds is reused).
fn import_once(s: &mut Session, path: &str, cache: &mut BTreeMap<String, ItemId>, imported: &mut Vec<Value>, cmd: &str) -> Result<ItemId> {
    if let Some(id) = cache.get(path) {
        return Ok(*id);
    }
    let r = s.execute("file.import", json!({"paths": [path], "reuse": true})).map_err(|e| bad(cmd, format!("importing {path}: {e}")))?;
    let id = r["items"].as_array().and_then(|a| a.first()).and_then(Value::as_u64).map(ItemId).ok_or_else(|| {
        let why = r["errors"].as_array().and_then(|a| a.first()).and_then(Value::as_str).unwrap_or("it is not media FilmCraft can open").to_string();
        bad(cmd, format!("importing {path}: {why}"))
    })?;
    cache.insert(path.to_string(), id);
    imported.push(json!({"path": path, "item": id.0}));
    Ok(id)
}

/// Extract a ZIP package's bundled files next to the user templates (or in the temporary
/// folder), so its media can be opened.
fn extract_assets(s: &Session, e: &Entry) -> Result<Option<String>> {
    if e.package.assets.is_empty() || e.asset_dir.is_some() {
        return Ok(e.asset_dir.clone());
    }
    let key = format!(
        "{}-{}",
        filmcraft_templates::slug(&e.package.manifest.id),
        filmcraft_templates::sha256_hex(e.package.manifest.to_json().as_bytes()).get(..12).unwrap_or("")
    );
    let base = user_dir(s).map(|d| d.join(".extracted")).unwrap_or_else(|| crate::temp_dir().join("filmcraft-video-templates"));
    let dir = base.join(key);
    for (rel, bytes) in &e.package.assets {
        let path = dir.join(rel);
        if !cfg!(target_arch = "wasm32")
            && let Some(parent) = path.parent()
        {
            std::fs::create_dir_all(parent).map_err(|err| EngineError::Other(format!("{}: {err}", parent.display())))?;
        }
        s.services.write_file(&path.to_string_lossy(), bytes).map_err(|err| EngineError::Other(format!("{}: {err}", path.display())))?;
    }
    Ok(Some(dir.to_string_lossy().to_string()))
}

fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.apply";
    let key = str_p(p, "template").or_else(|| str_p(p, "id")).unwrap_or("");
    let entry = find_template(s, key, CMD)?;
    let m = entry.package.manifest.clone();
    // read every slot value before anything changes
    let mut wanted: BTreeMap<String, Vec<Given>> = BTreeMap::new();
    let mut count = 0usize;
    if let Some(slots) = p.get("slots") {
        let o = slots.as_object().ok_or_else(|| bad(CMD, "`slots` is an object: {\"<slot id>\": value or [values]}"))?;
        for (id, v) in o {
            let slot = m.slot(id).ok_or_else(|| {
                let ids: Vec<&str> = m.slots.iter().map(|s| s.id.as_str()).collect();
                bad(CMD, format!("the template has no slot `{id}` (its slots: {})", ids.join(", ")))
            })?;
            let list: Vec<&Value> = match v {
                Value::Array(a) => a.iter().collect(),
                Value::Null => Vec::new(),
                one => vec![one],
            };
            count = count.saturating_add(list.len());
            if count > MAX_FILL_MEDIA {
                return Err(bad(CMD, format!("at most {MAX_FILL_MEDIA} values in `slots`")));
            }
            let mut gs = Vec::new();
            for x in list {
                gs.push(given(x, slot, CMD)?);
            }
            wanted.insert(id.clone(), gs);
        }
    }
    let variant = str_p(p, "variant").map(str::to_string);
    if let Some(v) = &variant
        && m.variant(Some(v)).is_none()
    {
        let ids: Vec<&str> = m.variants.iter().map(|v| v.id.as_str()).collect();
        return Err(bad(CMD, format!("the template has no variant `{v}` (its variants: {})", ids.join(", "))));
    }
    let name = str_p(p, "name").or_else(|| str_p(p, "sequence")).map(str::to_string);
    let generate_music = bool_p(p, "generateMusic").unwrap_or(true);
    let with_look = bool_p(p, "look").unwrap_or(true);
    let require_all = bool_p(p, "requireAll").unwrap_or(false);
    let label = format!("Apply Template {}", m.name);
    let step = label.clone();
    s.edit_group(&step, move |s| {
        let mut cache = BTreeMap::new();
        let mut imported = Vec::new();
        let mut fills = BTreeMap::new();
        for (id, gs) in wanted {
            let Some(slot) = m.slot(&id) else { continue };
            if slot.kind == SlotKind::Text {
                let texts = gs.into_iter().filter_map(|g| if let Given::Text(t) = g { Some(t) } else { None }).collect();
                fills.insert(id, SlotFill::Text(texts));
                continue;
            }
            let mut list = Vec::new();
            for g in gs {
                let (item, source_in, focus, by_variant) = match g {
                    Given::Path { path, source_in, focus, by_variant } => {
                        (import_once(s, &path, &mut cache, &mut imported, CMD)?, source_in, focus, by_variant)
                    }
                    Given::Item { item, source_in, focus, by_variant } => (item, source_in, focus, by_variant),
                    Given::Text(_) => continue,
                };
                list.push(MediaFill { item, source_in, focus, focus_by_variant: by_variant });
            }
            fills.insert(id, SlotFill::Media(list));
        }
        // the music: its beats, or a generated bed when the slot is empty
        let mut music = json!(null);
        let mut music_beats = None;
        let mut warnings: Vec<String> = Vec::new();
        if let Some(ms) = m.slots.iter().find(|x| x.kind == SlotKind::Music) {
            let filled = matches!(fills.get(&ms.id), Some(SlotFill::Media(l)) if !l.is_empty());
            if filled {
                if let Some(SlotFill::Media(l)) = fills.get(&ms.id)
                    && let Some(f) = l.first()
                {
                    let from = f.source_in.unwrap_or(Tick::ZERO);
                    let len = s.project.item(f.item).map(|i| i.duration()).unwrap_or_default();
                    let dur = (len - from).min(Tick(m.duration.0.saturating_mul(3)));
                    match crate::remix::beat_ticks(s, f.item, from, dur) {
                        Ok((bpm, beats)) => {
                            music = json!({"item": f.item.0, "generated": false, "bpm": bpm, "beats": beats.len()});
                            music_beats = Some(MusicBeats { beats });
                        }
                        Err(e) => {
                            warnings.push(format!("the music's beats could not be measured ({e}); the cuts stay on the template's grid"));
                            music = json!({"item": f.item.0, "generated": false});
                        }
                    }
                }
            } else if let (true, Some(SlotDefault::Generate { mood, bpm, seed })) = (generate_music, &ms.default) {
                let secs = m.duration.seconds() + 0.5;
                match s.execute("audio.generate", json!({"kind": "music", "duration": secs, "mood": mood, "bpm": bpm, "seed": seed, "place": false})) {
                    Ok(r) => {
                        if let Some(id) = r["item"].as_u64() {
                            fills.insert(ms.id.clone(), SlotFill::Media(vec![MediaFill::new(ItemId(id))]));
                            music = json!({"item": id, "generated": true, "bpm": bpm, "mood": mood});
                        }
                    }
                    Err(e) => warnings.push(format!("no music was generated ({e}); the placeholder stays")),
                }
            }
        }
        let asset_dir = extract_assets(s, &entry)?;
        let opts = ApplyOptions { variant: variant.clone(), name: name.clone(), fills, music_beats, look: None, asset_dir, require_all };
        let package = entry.package.clone();
        let applied = s.edit(&label, |pr, st| {
            let a = tapply::apply(pr, &package, &opts).map_err(|e| bad(CMD, e.to_string()))?;
            st.active_sequence = Some(a.sequence);
            if !st.open_sequences.contains(&a.sequence) {
                st.open_sequences.push(a.sequence);
            }
            Ok(a)
        })?;
        warnings.extend(applied.warnings.iter().cloned());
        // music shorter than the show: Remix it to length, or loop it
        let mut remixed = Vec::new();
        for (clip, to) in &applied.remix {
            match s.execute("clip.remix", json!({"clip": clip.0, "duration": to.0})) {
                Ok(_) => remixed.push(json!({"clip": clip.0, "how": "remixed"})),
                Err(e) => {
                    let seq = applied.sequence;
                    let (c, t) = (*clip, *to);
                    match s.edit(&label, |pr, _| tapply::loop_clip(pr, seq, c, t).map_err(|e| EngineError::Other(e.to_string()))) {
                        Ok(_) => remixed.push(json!({"clip": clip.0, "how": "looped"})),
                        Err(e2) => warnings.push(format!("the music could not be fitted to the show ({e}; {e2})")),
                    }
                }
            }
        }
        // the look on the picture slots
        let mut look = Value::Null;
        if with_look && let Some(l) = &m.look {
            let picture: Vec<u64> = applied
                .placed
                .iter()
                .filter(|x| m.slot(&x.slot).is_some_and(|sl| matches!(sl.kind, SlotKind::Video | SlotKind::Photo)))
                .flat_map(|x| x.clips.iter().map(|c| c.0))
                .collect();
            if !picture.is_empty() {
                let r = match l {
                    Look::Lumetri(n) => s.execute("lumetri.applyPreset", json!({"name": n, "clips": picture})).map(|_| json!({"lumetri": n})),
                    Look::Preset(n) => s.execute("presets.apply", json!({"preset": n, "clips": picture})).map(|_| json!({"preset": n})),
                };
                match r {
                    Ok(v) => look = v,
                    Err(e) => warnings.push(format!("the template's look was not applied: {e}")),
                }
            }
        }
        if let Some(e) = &applied.export_preset {
            if s.export_presets.find(e).is_some() {
                s.export_queue.quick_preset = Some(e.clone());
            } else {
                warnings.push(format!("the template's export preset `{e}` is not installed"));
            }
        }
        s.events.push(crate::Event::OpenSequence(applied.sequence));
        let seq = s.project.sequence(applied.sequence);
        let duration = seq.map(|q| q.duration()).unwrap_or(applied.duration);
        Ok(json!({
            "sequence": applied.sequence.0,
            "name": s.project.item(applied.sequence).map(|i| i.name.clone()),
            "template": m.id,
            "variant": applied.variant,
            "duration": duration.0,
            "seconds": duration.seconds(),
            "exportPreset": applied.export_preset,
            "retimed": applied.retimed,
            "cuts": applied.cuts.iter().map(|t| t.0).collect::<Vec<_>>(),
            "slots": applied.placed.iter().map(|x| json!({
                "slot": x.slot, "placement": x.placement, "media": x.media, "item": x.item.0,
                "clips": x.clips.iter().map(|c| c.0).collect::<Vec<_>>(), "how": x.how,
                "start": x.start.0, "duration": x.duration.0, "sourceIn": x.source_in.0, "speed": x.speed,
            })).collect::<Vec<_>>(),
            "imported": imported,
            "music": music,
            "remixed": remixed,
            "look": look,
            "warnings": warnings,
        }))
    })
}

// ------------------------------------------------------------------------------------------------
// Save and slot flags
// ------------------------------------------------------------------------------------------------

fn save(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.save";
    let path = str_p(p, "path").map(crate::export_tools::expand_home).ok_or_else(|| bad(CMD, "need `path` (a .fcvideo-template folder or a .zip)"))?;
    let seq_id = crate::commands::item_p(p, "sequence").or(s.state.active_sequence).ok_or_else(|| bad(CMD, "no sequence: open one or pass `sequence`"))?;
    let q = s.project.sequence(seq_id).ok_or_else(|| bad(CMD, "no such sequence"))?;
    let seq_name = s.project.item(seq_id).map(|i| i.name.clone()).unwrap_or_default();
    let name = str_p(p, "name").map(str::to_string).filter(|n| !n.trim().is_empty()).unwrap_or(seq_name);
    if name.len() > 256 {
        return Err(bad(CMD, "`name` is longer than 256 bytes"));
    }
    // the music's beats on the timeline, when a music slot is flagged
    let mut grid = None;
    let music = q
        .audio_tracks
        .iter()
        .flat_map(|t| t.items.iter())
        .find(|c| c.markers.iter().any(|mk| parse_slot_marker(&mk.name).is_some_and(|x| x.kind == SlotKind::Music)));
    if let Some(c) = music.cloned()
        && let Ok((bpm, beats)) = crate::remix::beat_ticks(s, c.item, c.source_in, c.duration)
    {
        let on_timeline: Vec<Tick> = beats.iter().filter(|b| **b >= c.source_in).map(|b| c.start + (*b - c.source_in)).filter(|t| *t <= c.end()).collect();
        if on_timeline.len() >= 2 {
            let bpm = if bpm.is_finite() { bpm.clamp(20.0, 300.0) } else { 120.0 };
            grid = Some(filmcraft_templates::BeatGrid { bpm, beats: on_timeline, mode: Default::default() });
        }
    }
    let opts = filmcraft_templates::save::SaveOptions {
        id: str_p(p, "id").unwrap_or("").to_string(),
        name: name.clone(),
        description: str_p(p, "description").unwrap_or("").chars().take(4000).collect(),
        author: str_p(p, "author").unwrap_or("").chars().take(200).collect(),
        license: str_p(p, "license").unwrap_or("MIT OR Apache-2.0").chars().take(200).collect(),
        category: str_p(p, "category").unwrap_or("My Templates").chars().take(200).collect(),
    };
    let saved = filmcraft_templates::save::save(&s.project, seq_id, &opts, grid).map_err(|e| bad(CMD, e.to_string()))?;
    let overwrite = bool_p(p, "overwrite").unwrap_or(false);
    let zip = path.to_ascii_lowercase().ends_with(".zip");
    let out = if zip || path.to_ascii_lowercase().ends_with(&format!(".{EXTENSION}")) { path.clone() } else { format!("{path}.{EXTENSION}") };
    if !overwrite && (s.services.file_size(&out).is_ok() || is_dir(s, &out)) {
        return Err(bad(CMD, format!("{out} exists (pass `overwrite: true` to replace it)")));
    }
    if zip {
        let bytes = saved.package.to_zip().map_err(|e| bad(CMD, e.to_string()))?;
        s.services.write_file(&out, &bytes).map_err(|e| EngineError::Other(format!("{out}: {e}")))?;
    } else {
        for (rel, bytes) in saved.package.files() {
            let f = join(&out, &rel);
            if !cfg!(target_arch = "wasm32")
                && let Some(parent) = std::path::Path::new(&f).parent()
            {
                std::fs::create_dir_all(parent).map_err(|e| EngineError::Other(format!("{}: {e}", parent.display())))?;
            }
            s.services.write_file(&f, &bytes).map_err(|e| EngineError::Other(format!("{f}: {e}")))?;
        }
    }
    let e = Entry { package: saved.package, source: "file", path: Some(out.clone()), asset_dir: None };
    Ok(json!({"path": out, "template": summary(&e), "warnings": saved.warnings}))
}

fn mark_slot(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.markSlot";
    let clear = bool_p(p, "clear").unwrap_or(false);
    let kind = match str_p(p, "kind") {
        Some(k) => Some(SlotKind::from_id(k).ok_or_else(|| bad(CMD, format!("unknown kind `{k}` (video, photo, text, logo, music)")))?),
        None if clear => None,
        None => return Err(bad(CMD, "need `kind` (video, photo, text, logo or music)")),
    };
    let slot = str_p(p, "slot").map(str::to_string).or_else(|| kind.map(|k| k.id().to_string())).unwrap_or_default();
    if !clear && !filmcraft_templates::manifest::is_slot_id(&slot) {
        return Err(bad(CMD, format!("`{slot}`: slot ids use lower-case letters, digits, `_` and `-`")));
    }
    let clips: Vec<ClipId> = crate::commands::clips_p(s, p);
    if clips.is_empty() {
        return Err(bad(CMD, "select clips or pass `clips`"));
    }
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    for c in &clips {
        let (tid, it) = seq.find_item(*c).ok_or_else(|| bad(CMD, format!("no clip {} in the active sequence", c.0)))?;
        let track_kind = seq.track(tid).map(|t| t.kind).unwrap_or_default();
        match kind {
            Some(SlotKind::Text) if !matches!(s.project.item(it.item).map(|i| &i.kind), Some(filmcraft_project::ItemKind::Graphic { .. })) => {
                return Err(bad(CMD, format!("“{}” is not a graphic: text slots are graphic clips with a text layer", it.name)));
            }
            Some(SlotKind::Music) if track_kind != TrackKind::Audio => return Err(bad(CMD, format!("“{}” is not on an audio track", it.name))),
            Some(k) if k.is_visual() && track_kind != TrackKind::Video => return Err(bad(CMD, format!("“{}” is not on a video track", it.name))),
            _ => {}
        }
    }
    let n = s.edit_sequence(if clear { "Clear Template Slot" } else { "Mark Template Slot" }, |q, ctx, _| {
        let mut n = 0;
        for c in &clips {
            if let Some((_, it)) = q.find_item_mut(*c) {
                it.markers.retain(|mk| !mk.name.starts_with(tapply::SLOT_MARKER));
                if let Some(k) = kind.filter(|_| !clear) {
                    it.markers.push(Marker {
                        id: MarkerId(ctx.alloc()),
                        start: it.source_in,
                        duration: Tick::ZERO,
                        name: slot_marker_name(k, &slot, None, None),
                        comment: String::new(),
                        kind: MarkerKind::Comment,
                        color: Label::Mango,
                    });
                }
                n += 1;
            }
        }
        Ok(n)
    })?;
    Ok(json!({"clips": n, "kind": kind.map(|k| k.id()), "slot": if clear { Value::Null } else { json!(slot) }}))
}
