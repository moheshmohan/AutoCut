//! Video templates (`.fcvideo-template`): whole-sequence starters with media slots — the format,
//! validation, slot fit, beat grids and built-in starters live in `filmcraft-templates`; this
//! module is their commands.
//!
//! | command | does |
//! |---|---|
//! | `template.list` | the built-in starters and the user's templates (`<data dir>/Video Templates/`: `*.fcvideo-template` folders and `.zip` packages); `query`, `category`, `source`, `aspect` (a layout it has: `9:16`), `kind` (a slot kind it has: `captions`), `minMedia` / `maxMedia` (how many clips or photos it takes) filter; templates that fail to read are listed under `invalid` with why |
//! | `template.inspect` | one template (`template`: id, name or path): slots with their placements and AI hints (`ai`: what each picture slot's shot is for, how lively, words to find it by, speech or not, the least quality, filming order; what each text slot's model text says), variants, beat grid, look, caption style, export preset, the manifest |
//! | `template.validate` | check a package (`path`: folder, `.zip` or `template.json`): `valid`, and every problem with the field it is about |
//! | `template.apply` | a new sequence from a template: `variant` (`16:9`, `9:16`…), `slots` (`{"photos": ["/a.jpg", 12, {"path": "/b.mov", "inSeconds": 3, "focus": [0.4, 0.5]}], "title": "Lisbon", "captions": [{"start": ticks, "end": ticks, "text": "…"}]}`), `name`, `captions: false` (no transcribing); one undo step |
//! | `template.save` | Save as Template: clips flagged as slots (`template.markSlot`, or left by an earlier apply) become slots with neutral generated placeholders; other personal media is left out unless `bundle`d (with its attribution); the base template's (or the author's) slot details, captions slot, layouts and hints, poster, look and export presets carry over; written into the user templates folder (or `path`: a folder or a `.zip`), where the browser shows it |
//! | `template.markSlot` | flag clips as placeholders of a slot (`kind`, `slot`) for `template.save`, or `clear` the flag |
//! | `template.thumbnail` | a frame of the template's own placeholder sequence (`variant`, `seconds`, default its poster frame; `maxSide`) as PNG — what the template browser shows ([`crate::video_template_frames`]) |
//! | `template.preview` | frames of the template filled with project items and text (`slots` as for apply, item ids only), without changing the project |
//! | `template.media` | the project's media that can fill a slot (`kind`), Google Photos picks marked |
//! | `template.export`, `template.install`, `template.qr`, `template.exportOtio`, `sequence.exportOtio` | sharing as plain files and OpenTimelineIO ([`crate::video_template_share`]) |
//!
//! **Applying.** Files named in `slots` are imported (a file the project already holds is reused,
//! and one file named for several placements is one item). A filled music slot is measured
//! (`media.beats`' analysis; or `musicBeats`, its beats in ticks from its start when the caller
//! knows them, a bed it generated) and the cuts re-quantized to its beats; an unfilled one with a
//! `generate` default gets a bed made with `audio.generate` at the template's tempo and length,
//! and the cuts move to that bed's exact beats (its length kept). Music shorter than the show is
//! Remixed to length (looped when Remix cannot). The template's Elements sounds (stand-ins naming
//! a `sound.*` element) are synthesized and swapped in. A captions slot whose sources were filled
//! is transcribed (`transcript.generate`: Whisper, or the host's transcriber) and its captions
//! written on the template's caption track, `maxChars` a line; without speech-to-text the track
//! stays empty with a warning. The template's look goes on the picture slots
//! (`lumetri.applyPreset` / `presets.apply`), its caption style on the caption tracks, and its
//! export preset (the variant's) becomes Quick Export's preset.

use std::collections::BTreeMap;

use filmcraft_project::{ClipId, ItemId, Label, Marker, MarkerId, MarkerKind, TrackKind};
use filmcraft_templates::apply::{self as tapply, ApplyOptions, Cue, MediaFill, MusicBeats, SlotFill, parse_slot_marker, slot_marker_name};
use filmcraft_templates::manifest::{Look, MAX_TEMPLATE_TICKS, Manifest, QuantizeMode, Slot, SlotDefault, SlotKind};
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
    let mut v = vec![
        spec(
            "template.list",
            "List Video Templates",
            r#"{"query":str?,"category":str?,"source":"builtin"|"user"?,"aspect":"16:9"|"9:16"|"1:1"|"4:5"?,"kind":"video"|"photo"|"text"|"logo"|"music"|"captions"?,"minMedia":n?,"maxMedia":n?}"#,
            always,
            list,
            false,
        ),
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
            r#"{"template":id|name|path,"variant":"16:9"|"9:16"|…?,"slots":{"<slot>": path|itemId|text|{"path"|"item"|"text","in":ticks?,"inSeconds":f64?,"focus":[x,y]?,"focusByVariant":{"9:16":[x,y]}?}|[…], "<captions slot>": [{"start":ticks|"startSeconds":f64,"end":ticks|"endSeconds":f64,"text":str,"speaker":str?}]}?,"name":str?,"sequence":str (name)?,"requireAll":bool?,"generateMusic":bool=true,"musicBeats":[ticks]?,"look":bool=true,"sounds":bool=true,"captions":bool=true}"#,
            always,
            apply,
            true,
        ),
        CommandSpec {
            id: "template.save",
            label: "Save as Video Template…",
            menu: &["Sequence"],
            shortcut: None,
            params: SAVE_PARAMS,
            enabled: has_seq,
            run: save,
            journal: false,
        },
        spec(
            "template.markSlot",
            "Mark Clips as Template Slot",
            r#"{"clips":[id]?,"kind":"video"|"photo"|"text"|"logo"|"music","slot":str?,"clear":bool?}"#,
            has_seq,
            mark_slot,
            true,
        ),
    ];
    v.extend(crate::video_template_frames::commands());
    v.extend(crate::video_template_share::commands());
    v
}

/// `template.save`'s parameters.
pub const SAVE_PARAMS: &str = r#"{"path":str (.fcvideo-template folder or .zip; default: the user templates folder)?,"zip":bool=false,"sequence":id?,"name":str?,"id":str?,"description":str?,"author":str?,"license":str="MIT OR Apache-2.0","category":str?,"version":str?,"tags":[str]?,"base":id|"none"?,"slots":{"<slot>":{"name","prompt","order","minCount","beatSync","fit","constraints","default","ai":{"purpose","query","energy","speech","minQuality","chronological","maxHold","prompt","maxChars","about"}}}?,"captions":bool|{"id","name","source":[slot],"maxChars","lines"}?,"variants":["16:9"|"9:16"|"1:1"|"4:5"]?,"poster":ticks?,"posterSeconds":f64?,"look":{"lumetri":str}|{"preset":str}?,"exportPreset":str?,"bundle":[itemId]|true?,"overwrite":bool?}"#;

/// Bumped whenever a template is saved or installed in the user folder (the browser re-reads).
static LIBRARY_REVISION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// The user templates folder's revision ([`bump_library`]).
pub fn library_revision() -> u64 {
    LIBRARY_REVISION.load(std::sync::atomic::Ordering::Relaxed)
}

/// Note that the user templates folder changed.
pub fn bump_library() {
    LIBRARY_REVISION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
    if let Some(p) = filmcraft_templates::builtin::find(key) {
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
        "captions": s.captions.as_ref().map(|c| serde_json::to_value(c).unwrap_or_default()),
        "ai": s.ai.as_ref().map(|a| serde_json::to_value(a).unwrap_or_default()),
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
        "aspects": m.variants.iter().map(|v| v.id.clone()).collect::<Vec<_>>(),
        "mediaSlots": m.media_slots(),
        "mediaCount": m.media_count(),
        "captions": m.captions_slot().is_some(),
        "aiReady": m.ai_ready(),
        "poster": m.poster_time().0,
        "posterSeconds": m.poster_time().seconds(),
    })
}

/// Whether a template matches `template.list`'s (and the browser's) filters: `aspect` a layout
/// it has, `kind` a slot kind it has, the clips or photos it takes ([`Manifest::media_count`])
/// within `min..=max`.
pub fn matches_filters(m: &Manifest, aspect: Option<&str>, kind: Option<SlotKind>, min: Option<usize>, max: Option<usize>) -> bool {
    aspect.is_none_or(|a| m.variants.iter().any(|v| v.id.eq_ignore_ascii_case(a)))
        && kind.is_none_or(|k| m.slots.iter().any(|s| s.kind == k))
        && min.is_none_or(|n| m.media_count() >= n)
        && max.is_none_or(|n| m.media_count() <= n)
}

fn list(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.list";
    let q = str_p(p, "query").unwrap_or("").to_ascii_lowercase();
    let cat = str_p(p, "category").map(str::to_ascii_lowercase);
    let src = str_p(p, "source");
    let aspect = str_p(p, "aspect");
    let kind = match str_p(p, "kind") {
        Some(k) => Some(SlotKind::from_id(k).ok_or_else(|| bad(CMD, format!("unknown slot kind `{k}` (video, photo, text, logo, music, captions)")))?),
        None => None,
    };
    let n = |k: &str| crate::commands::u64_p(p, k).map(|n| usize::try_from(n).unwrap_or(usize::MAX));
    let (min, max) = (n("minMedia"), n("maxMedia"));
    let (lib, invalid) = library(s);
    let v: Vec<Value> = lib
        .iter()
        .filter(|e| src.is_none_or(|x| x.eq_ignore_ascii_case(e.source)))
        .filter(|e| cat.as_deref().is_none_or(|c| e.package.manifest.category.to_ascii_lowercase() == c))
        .filter(|e| matches_filters(&e.package.manifest, aspect, kind, min, max))
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
    v["checks"] = Value::Array(package_checks(s, &e.package).iter().map(|(f, m)| json!({"field": f, "message": m})).collect());
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

/// [`engine_checks`] with the package: the Elements sounds its stand-ins name exist, and its
/// generated beds' moods and genres are the generator's.
pub fn package_checks(s: &Session, p: &Package) -> Vec<(String, String)> {
    let mut out = engine_checks(s, &p.manifest);
    for it in p.fragment.items.values() {
        if let Some(e) = it.metadata.get(filmcraft_templates::ELEMENT_METADATA)
            && crate::elements::sound(e).is_none()
        {
            out.push((format!("sequence item {} (“{}”)", it.id.0, it.name), format!("no Elements sound `{e}` (see elements.list)")));
        }
    }
    for (i, sl) in p.manifest.slots.iter().enumerate() {
        if let Some(SlotDefault::Generate { mood, genre, .. }) = &sl.default {
            if filmcraft_audio_dsp::music::Mood::from_id(mood).is_none() {
                out.push((format!("slots[{i}].default.generate.mood"), format!("`{mood}` is not a mood audio.generate knows")));
            }
            if let Some(g) = genre
                && !filmcraft_audio_dsp::music::Genre::ALL.iter().any(|x| x.id() == g)
            {
                out.push((format!("slots[{i}].default.generate.genre"), format!("`{g}` is not a genre audio.generate knows")));
            }
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
            let checks = package_checks(s, &e.package);
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
pub enum Given {
    Path { path: String, source_in: Option<Tick>, focus: Option<[f64; 2]>, by_variant: BTreeMap<String, [f64; 2]> },
    Item { item: ItemId, source_in: Option<Tick>, focus: Option<[f64; 2]>, by_variant: BTreeMap<String, [f64; 2]> },
    Text(String),
    Cue(Cue),
}

fn pair(v: &Value) -> Option<[f64; 2]> {
    let a = v.as_array().filter(|a| a.len() == 2)?;
    Some([a.first()?.as_f64()?, a.get(1)?.as_f64()?])
}

/// A cue for a captions slot: `{"start": ticks | "startSeconds": s, "end" | "endSeconds", "text", "speaker"?}`.
fn cue(v: &Value, slot: &Slot, cmd: &str) -> Result<Cue> {
    let o = v.as_object().ok_or_else(|| bad(cmd, format!("slot `{}` takes cues: {{\"start\", \"end\", \"text\"}}", slot.id)))?;
    let time = |k: &str| -> Result<Tick> {
        match (o.get(k).and_then(Value::as_i64), f64_p(v, &format!("{k}Seconds"))) {
            (Some(t), _) if (0..=MAX_TEMPLATE_TICKS.saturating_mul(4)).contains(&t) => Ok(Tick(t)),
            (None, Some(sec)) if sec.is_finite() && (0.0..=4.0 * 3600.0).contains(&sec) => Ok(Tick::from_seconds_f64(sec)),
            _ => Err(bad(cmd, format!("slot `{}`: a cue needs `{k}` (ticks) or `{k}Seconds`, from 0 to 4 hours", slot.id))),
        }
    };
    let (start, end) = (time("start")?, time("end")?);
    let text = o
        .get("text")
        .and_then(Value::as_str)
        .filter(|t| t.len() <= 4096)
        .ok_or_else(|| bad(cmd, format!("slot `{}`: a cue needs `text` (at most 4096 bytes)", slot.id)))?;
    let speaker = o.get("speaker").and_then(Value::as_str).map(|s| s.chars().take(200).collect());
    Ok(Cue { start, end, text: text.to_string(), speaker })
}

/// Every slot value of `p["slots"]`, read and checked against the template before anything
/// changes.
pub fn read_slots(m: &Manifest, p: &Value, cmd: &str) -> Result<BTreeMap<String, Vec<Given>>> {
    let mut wanted: BTreeMap<String, Vec<Given>> = BTreeMap::new();
    let mut count = 0usize;
    if let Some(slots) = p.get("slots").filter(|v| !v.is_null()) {
        let o = slots.as_object().ok_or_else(|| bad(cmd, "`slots` is an object: {\"<slot id>\": value or [values]}"))?;
        for (id, v) in o {
            let slot = m.slot(id).ok_or_else(|| {
                let ids: Vec<&str> = m.slots.iter().map(|s| s.id.as_str()).collect();
                bad(cmd, format!("the template has no slot `{id}` (its slots: {})", ids.join(", ")))
            })?;
            let list: Vec<&Value> = match v {
                Value::Array(a) => a.iter().collect(),
                Value::Null => Vec::new(),
                one => vec![one],
            };
            count = count.saturating_add(list.len());
            if count > MAX_FILL_MEDIA.max(tapply::MAX_CUES) || (slot.kind != SlotKind::Captions && count > MAX_FILL_MEDIA) {
                return Err(bad(cmd, format!("at most {MAX_FILL_MEDIA} values in `slots`")));
            }
            let mut gs = Vec::new();
            for x in list {
                gs.push(if slot.kind == SlotKind::Captions { Given::Cue(cue(x, slot, cmd)?) } else { given(x, slot, cmd)? });
            }
            wanted.insert(id.clone(), gs);
        }
    }
    Ok(wanted)
}

/// Fills of the values that need no importing (items, text, cues); `Err` names a file path.
pub fn fills_without_import(m: &Manifest, wanted: BTreeMap<String, Vec<Given>>, cmd: &str) -> Result<BTreeMap<String, SlotFill>> {
    let mut fills = BTreeMap::new();
    for (id, gs) in wanted {
        let Some(slot) = m.slot(&id) else { continue };
        let fill = match slot.kind {
            SlotKind::Text => SlotFill::Text(gs.into_iter().filter_map(|g| if let Given::Text(t) = g { Some(t) } else { None }).collect()),
            SlotKind::Captions => SlotFill::Captions(gs.into_iter().filter_map(|g| if let Given::Cue(c) = g { Some(c) } else { None }).collect()),
            _ => {
                let mut list = Vec::new();
                for g in gs {
                    match g {
                        Given::Item { item, source_in, focus, by_variant } => list.push(MediaFill { item, source_in, focus, focus_by_variant: by_variant }),
                        Given::Path { path, .. } => return Err(bad(cmd, format!("slot `{id}`: {path} is a file; import it first and give its item id"))),
                        Given::Text(_) | Given::Cue(_) => {}
                    }
                }
                SlotFill::Media(list)
            }
        };
        fills.insert(id, fill);
    }
    Ok(fills)
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
    let wanted = read_slots(&m, p, CMD)?;
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
    let with_sounds = bool_p(p, "sounds").unwrap_or(true);
    let with_captions = bool_p(p, "captions").unwrap_or(true);
    let given_beats = music_beats_p(p, CMD)?;
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
            if slot.kind == SlotKind::Captions {
                let cues = gs.into_iter().filter_map(|g| if let Given::Cue(c) = g { Some(c) } else { None }).collect();
                fills.insert(id, SlotFill::Captions(cues));
                continue;
            }
            let mut list = Vec::new();
            for g in gs {
                let (item, source_in, focus, by_variant) = match g {
                    Given::Path { path, source_in, focus, by_variant } => {
                        (import_once(s, &path, &mut cache, &mut imported, CMD)?, source_in, focus, by_variant)
                    }
                    Given::Item { item, source_in, focus, by_variant } => (item, source_in, focus, by_variant),
                    Given::Text(_) | Given::Cue(_) => continue,
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
                    // the caller's beats (a bed it generated: exact), else measured
                    let known = given_beats.as_ref().map(|b| {
                        let beats: Vec<Tick> = b.iter().filter(|t| **t >= from).map(|t| *t - from).collect();
                        let bpm = match (beats.first(), beats.last()) {
                            (Some(a), Some(z)) if beats.len() >= 2 && z > a => 60.0 * (beats.len() - 1) as f64 / (*z - *a).seconds(),
                            _ => 0.0,
                        };
                        (bpm, beats)
                    });
                    let measured = match known.filter(|(_, b)| b.len() >= 2) {
                        Some(k) => Ok(k),
                        None => crate::remix::beat_ticks(s, f.item, from, dur),
                    };
                    match measured {
                        Ok((bpm, beats)) => {
                            music = json!({"item": f.item.0, "generated": false, "bpm": bpm, "beats": beats.len()});
                            music_beats = Some(MusicBeats { beats, mode: None });
                        }
                        Err(e) => {
                            warnings.push(format!("the music's beats could not be measured ({e}); the cuts stay on the template's grid"));
                            music = json!({"item": f.item.0, "generated": false});
                        }
                    }
                }
            } else if let (true, Some(SlotDefault::Generate { mood, bpm, seed, genre })) = (generate_music, &ms.default) {
                let secs = m.duration.seconds() + 0.5;
                let mut gp = json!({"kind": "music", "duration": secs, "mood": mood, "bpm": bpm, "seed": seed, "place": false});
                if let Some(g) = genre {
                    gp["genre"] = json!(g);
                }
                match s.execute("audio.generate", gp) {
                    Ok(r) => {
                        if let Some(id) = r["item"].as_u64() {
                            fills.insert(ms.id.clone(), SlotFill::Media(vec![MediaFill::new(ItemId(id))]));
                            music = json!({"item": id, "generated": true, "bpm": bpm, "mood": mood, "genre": genre});
                            // the bed's exact beats (from its start): the cuts move onto them, the length stays
                            let beats: Vec<Tick> =
                                r["beats"].as_array().map(Vec::as_slice).unwrap_or_default().iter().filter_map(Value::as_i64).map(Tick).collect();
                            if beats.len() >= 2 {
                                music_beats = Some(MusicBeats { beats, mode: Some(QuantizeMode::Nearest) });
                            }
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
        // the Elements sounds: synthesized, imported (once) and swapped for their stand-ins
        let sounds = if with_sounds { swap_in_sounds(s, &applied, &label, &mut warnings) } else { Vec::new() };
        // the captions slot: transcribe what its sources say
        let captions = if with_captions && applied.captions_pending { caption_sources(s, &m, &applied, &label, &mut warnings) } else { Value::Null };
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
            "sounds": sounds,
            "captions": captions,
            "warnings": warnings,
        }))
    })
}

/// `musicBeats`: the filled music's beats, ticks from its start, increasing (at most 20 000).
fn music_beats_p(p: &Value, cmd: &str) -> Result<Option<Vec<Tick>>> {
    let Some(v) = p.get("musicBeats").filter(|v| !v.is_null()) else { return Ok(None) };
    let a = v
        .as_array()
        .filter(|a| a.len() <= filmcraft_templates::manifest::MAX_BEATS)
        .ok_or_else(|| bad(cmd, "`musicBeats` is a list of at most 20000 ticks"))?;
    let mut out: Vec<Tick> = Vec::with_capacity(a.len());
    for t in a {
        let t = t
            .as_i64()
            .filter(|t| (0..=MAX_TEMPLATE_TICKS.saturating_mul(4)).contains(t))
            .ok_or_else(|| bad(cmd, "`musicBeats` are ticks from the music's start"))?;
        if out.last().is_some_and(|l| l.0 >= t) {
            return Err(bad(cmd, "`musicBeats` must increase"));
        }
        out.push(Tick(t));
    }
    Ok(Some(out))
}

/// Synthesize the template's Elements sounds and swap them in for their stand-ins (part of the
/// apply's undo step). Returns what was swapped.
fn swap_in_sounds(s: &mut Session, applied: &tapply::Applied, label: &str, warnings: &mut Vec<String>) -> Vec<Value> {
    let mut out = Vec::new();
    let stand_ins: Vec<(ItemId, String)> = applied
        .items
        .iter()
        .filter_map(|i| s.project.item(*i).and_then(|it| it.metadata.get(filmcraft_templates::ELEMENT_METADATA).map(|e| (*i, e.clone()))))
        .collect();
    for (stand_in, id) in stand_ins {
        let Some(snd) = crate::elements::sound(&id) else {
            warnings.push(format!("the template names an Elements sound `{id}` this build does not have; it stays silent"));
            continue;
        };
        let audio = match snd.render() {
            Ok(a) => a,
            Err(e) => {
                warnings.push(format!("the sound `{id}` could not be made ({e}); it stays silent"));
                continue;
            }
        };
        let bytes = crate::audio_generate::write_wav_f32_stereo(&audio.left, &audio.right, crate::elements::RATE);
        let real = match crate::elements::sound_item(s, &json!({}), snd, bytes) {
            Ok((item, _, _)) => item,
            Err(e) => {
                warnings.push(format!("the sound `{id}` could not be written ({e}); it stays silent"));
                continue;
            }
        };
        let seq = applied.sequence;
        let e = id.clone();
        let r = s.edit(label, move |pr, _| {
            if let Some(it) = pr.item_mut(real) {
                it.metadata.insert(filmcraft_templates::ELEMENT_METADATA.into(), e);
            }
            tapply::swap_item(pr, seq, stand_in, real).map_err(|err| EngineError::Other(err.to_string()))
        });
        match r {
            Ok(n) => out.push(json!({"element": id, "item": real.0, "clips": n})),
            Err(e) => warnings.push(format!("the sound `{id}` was not placed: {e}")),
        }
    }
    out
}

/// Transcribe the sources of the template's captions slot and write their captions (part of the
/// apply's undo step). Without speech-to-text, a warning says how to get it.
fn caption_sources(s: &mut Session, m: &Manifest, applied: &tapply::Applied, label: &str, warnings: &mut Vec<String>) -> Value {
    let Some(spec) = m.captions_slot().and_then(|c| c.captions.clone()) else { return Value::Null };
    let seq_id = applied.sequence;
    let Some(seq) = s.project.sequence(seq_id).cloned() else { return Value::Null };
    let mut items: Vec<ItemId> = Vec::new();
    for c in seq.audio_tracks.iter().flat_map(|t| t.items.iter()).filter(|c| applied.caption_clips.contains(&c.id)) {
        if let Some((root, _, _)) = s.project.resolve_media(c.item)
            && !items.contains(&root)
        {
            items.push(root);
        }
    }
    if items.is_empty() {
        warnings.push("auto-captions: the captioned media has no sound".into());
        return json!({"captions": 0});
    }
    let missing: Vec<u64> = items.iter().filter(|i| !s.project.transcripts.contains_key(i)).map(|i| i.0).collect();
    if !missing.is_empty() {
        if let Err(why) = crate::transcript::can_transcribe(s) {
            warnings.push(format!("auto-captions: {why} (Transcribe… and the captions slot can be filled later)"));
            return json!({"captions": 0, "transcribed": false});
        }
        if let Err(e) = s.execute("transcript.generate", json!({"items": missing})) {
            warnings.push(format!("auto-captions: transcribing failed ({e})"));
            return json!({"captions": 0, "transcribed": false});
        }
    }
    // the words of the captioned clips only, on the timeline
    let mut only = seq.clone();
    for t in only.audio_tracks.iter_mut() {
        t.items.retain(|c| applied.caption_clips.contains(&c.id));
    }
    let words = filmcraft_edit::transcript::sequence_words(&only, &s.project.transcripts);
    let rules = filmcraft_edit::transcript::CaptionRules {
        max_chars: spec.max_chars as usize,
        lines: spec.lines as usize,
        ..filmcraft_edit::transcript::CaptionRules::default()
    };
    let blocks = filmcraft_edit::transcript::caption_blocks(&words, &rules, seq.settings.frame_rate);
    let cues: Vec<Cue> = blocks.iter().map(|b| Cue { start: b.start, end: b.end, text: b.text.clone(), speaker: b.speaker.clone() }).collect();
    let style = m.caption_style.clone();
    match s.edit(label, move |pr, _| tapply::set_captions(pr, seq_id, &cues, style.as_ref()).map_err(|e| EngineError::Other(e.to_string()))) {
        Ok(n) => json!({"captions": n, "transcribed": true, "words": words.len()}),
        Err(e) => {
            warnings.push(format!("auto-captions: {e}"));
            json!({"captions": 0, "transcribed": true})
        }
    }
}

// ------------------------------------------------------------------------------------------------
// Save and slot flags
// ------------------------------------------------------------------------------------------------

/// Largest file `template.save` bundles, and all of them together.
const MAX_BUNDLE_FILE: u64 = 128 * 1024 * 1024;
const MAX_BUNDLE_TOTAL: u64 = 256 * 1024 * 1024;

/// What a `<file>.attribution` sidecar (AGENTS.md §1.3's format) says: title, author, source,
/// licence.
pub fn read_attribution(text: &str) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for line in text.lines().take(200) {
        if let Some((k, v)) = line.split_once(':') {
            let k = k.trim().to_ascii_lowercase();
            if matches!(k.as_str(), "title" | "author" | "source" | "license" | "licence") && !out.contains_key(&k) {
                out.insert(if k == "licence" { "license".into() } else { k }, v.trim().chars().take(400).collect());
            }
        }
    }
    out
}

/// The slot ids flagged on a sequence's clips, with their kinds.
fn flagged_slots(q: &filmcraft_project::Sequence) -> BTreeMap<String, SlotKind> {
    let mut out = BTreeMap::new();
    for t in q.video_tracks.iter().chain(q.audio_tracks.iter()) {
        for c in &t.items {
            for f in c.markers.iter().filter_map(|m| parse_slot_marker(&m.name)) {
                out.entry(f.slot).or_insert(f.kind);
            }
        }
    }
    out
}

/// `template.save`'s authoring options (everything but the files to bundle).
fn save_options(
    s: &Session,
    p: &Value,
    q: &filmcraft_project::Sequence,
    seq_id: ItemId,
    name: &str,
    warnings: &mut Vec<String>,
) -> Result<filmcraft_templates::save::SaveOptions> {
    use filmcraft_templates::save::{CaptionsChoice, CaptionsEdit, SaveOptions};
    const CMD: &str = "template.save";
    let text = |k: &str, max: usize| str_p(p, k).unwrap_or("").chars().take(max).collect::<String>();
    // the template the sequence was made from (an earlier apply), unless `base: "none"`
    let base = match str_p(p, "base") {
        Some("none") => None,
        Some(key) => Some(find_template(s, key, CMD)?.package),
        None => match s.project.item(seq_id).and_then(|i| i.metadata.get("Template")).cloned() {
            Some(id) => match find_template(s, &id, CMD) {
                Ok(e) => Some(e.package),
                Err(_) => {
                    warnings.push(format!("the template this sequence was made from (`{id}`) is not installed; its slot details are not carried over"));
                    None
                }
            },
            None => None,
        },
    };
    let flagged = flagged_slots(q);
    let mut slots = BTreeMap::new();
    if let Some(v) = p.get("slots").filter(|v| !v.is_null()) {
        let o = v.as_object().ok_or_else(|| bad(CMD, "`slots` is an object: {\"<slot id>\": {\"name\", \"prompt\", …}}"))?;
        if o.len() > filmcraft_templates::manifest::MAX_SLOTS {
            return Err(bad(CMD, "too many slots in `slots`"));
        }
        for (id, edit) in o {
            let kind = flagged.get(id).copied().ok_or_else(|| {
                let ids: Vec<&str> = flagged.keys().map(String::as_str).collect();
                bad(CMD, format!("no clip is flagged as slot `{id}` (flagged: {}; template.markSlot flags clips)", ids.join(", ")))
            })?;
            let e = filmcraft_templates::manifest::read_slot_edit(edit, kind, &format!("slots.{id}")).map_err(|e| bad(CMD, e.to_string()))?;
            slots.insert(id.clone(), e);
        }
    }
    let captions = match p.get("captions") {
        None | Some(Value::Null) => CaptionsChoice::Auto,
        Some(Value::Bool(false)) => CaptionsChoice::Off,
        Some(Value::Bool(true)) => CaptionsChoice::Slot(CaptionsEdit::default()),
        Some(Value::Object(o)) => {
            let mut c = CaptionsEdit::default();
            if let Some(id) = o.get("id").and_then(Value::as_str) {
                c.id = id.chars().take(64).collect();
            }
            if let Some(n) = o.get("name").and_then(Value::as_str) {
                c.name = n.chars().take(200).collect();
            }
            if let Some(a) = o.get("source") {
                let a = a.as_array().ok_or_else(|| bad(CMD, "`captions.source` is a list of slot ids"))?;
                c.source = a.iter().take(16).filter_map(Value::as_str).map(str::to_string).collect();
            }
            if let Some(n) = o.get("maxChars").and_then(Value::as_u64) {
                c.max_chars = u32::try_from(n.clamp(8, 200)).unwrap_or(32);
            }
            if let Some(n) = o.get("lines").and_then(Value::as_u64) {
                c.lines = u32::try_from(n.clamp(1, 2)).unwrap_or(1);
            }
            CaptionsChoice::Slot(c)
        }
        Some(v) => return Err(bad(CMD, format!("`captions` is true, false or {{\"source\": [slot ids], \"maxChars\", \"lines\"}}, not {v}"))),
    };
    let variants: Vec<String> = match p.get("variants") {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(a)) if a.len() <= 16 => a.iter().filter_map(Value::as_str).map(|v| v.chars().take(16).collect()).collect(),
        Some(_) => return Err(bad(CMD, "`variants` is a list of layouts such as [\"16:9\", \"9:16\", \"1:1\", \"4:5\"]")),
    };
    let max = Tick(filmcraft_templates::manifest::MAX_TEMPLATE_TICKS);
    let poster = match (p.get("poster").and_then(Value::as_i64), crate::commands::f64_p(p, "posterSeconds")) {
        (Some(t), _) if (0..=max.0).contains(&t) => Some(Tick(t)),
        (Some(_), _) => return Err(bad(CMD, "`poster` is ticks from the start of the template")),
        (None, Some(x)) if x.is_finite() && (0.0..=max.seconds()).contains(&x) => Some(Tick::from_seconds_f64(x)),
        (None, Some(_)) => return Err(bad(CMD, "`posterSeconds` is seconds from the start of the template")),
        (None, None) => None,
    };
    let look = match p.get("look") {
        None | Some(Value::Null) => None,
        Some(v) => match (v.get("lumetri").and_then(Value::as_str), v.get("preset").and_then(Value::as_str)) {
            (Some(n), None) if n.len() <= 200 => Some(Look::Lumetri(n.to_string())),
            (None, Some(n)) if n.len() <= 200 => Some(Look::Preset(n.to_string())),
            _ => return Err(bad(CMD, "`look` is {\"lumetri\": name} or {\"preset\": name}")),
        },
    };
    let tags: Vec<String> = match p.get("tags") {
        Some(Value::Array(a)) => a.iter().take(32).filter_map(Value::as_str).map(|t| t.chars().take(64).collect()).collect(),
        _ => Vec::new(),
    };
    Ok(SaveOptions {
        id: text("id", 128),
        name: name.to_string(),
        description: text("description", 4000),
        author: text("author", 200),
        license: str_p(p, "license").filter(|l| !l.trim().is_empty()).unwrap_or("MIT OR Apache-2.0").chars().take(200).collect(),
        category: text("category", 200),
        version: text("version", 32),
        tags,
        base,
        slots,
        captions,
        variants,
        poster,
        look,
        export_preset: str_p(p, "exportPreset").map(|e| e.chars().take(200).collect()),
        bundle: Vec::new(),
    })
}

/// The user's own media of a sequence that no slot uses: (item, file path).
pub fn own_media(s: &Session, q: &filmcraft_project::Sequence) -> Vec<(ItemId, String)> {
    let mut out: Vec<(ItemId, String)> = Vec::new();
    for t in q.video_tracks.iter().chain(q.audio_tracks.iter()) {
        for c in &t.items {
            if c.markers.iter().any(|m| parse_slot_marker(&m.name).is_some()) || out.iter().any(|(i, _)| *i == c.item) {
                continue;
            }
            let Some(it) = s.project.item(c.item) else { continue };
            if it.metadata.contains_key(filmcraft_templates::ELEMENT_METADATA) {
                continue;
            }
            if let Some(m) = it.as_media()
                && let filmcraft_project::MediaRef::File { path } = &m.media
            {
                out.push((c.item, path.clone()));
            }
        }
    }
    out
}

/// The files `bundle` names, read with their attribution (the sidecar's, else the author's own
/// work under the template's licence).
fn bundle_files(
    s: &Session,
    p: &Value,
    q: &filmcraft_project::Sequence,
    opts: &filmcraft_templates::save::SaveOptions,
) -> Result<Vec<filmcraft_templates::save::BundleFile>> {
    const CMD: &str = "template.save";
    let own = own_media(s, q);
    let wanted: Vec<(ItemId, String)> = match p.get("bundle") {
        None | Some(Value::Null) | Some(Value::Bool(false)) => return Ok(Vec::new()),
        Some(Value::Bool(true)) => own,
        Some(Value::Array(a)) => {
            let mut v = Vec::new();
            for x in a.iter().take(filmcraft_templates::manifest::MAX_ASSETS + 1) {
                let id = x.as_u64().map(ItemId).ok_or_else(|| bad(CMD, "`bundle` is true or a list of item ids"))?;
                let f = own
                    .iter()
                    .find(|(i, _)| *i == id)
                    .cloned()
                    .ok_or_else(|| bad(CMD, format!("item {} is not your own media in this sequence (outside the slots)", id.0)))?;
                v.push(f);
            }
            v
        }
        Some(_) => return Err(bad(CMD, "`bundle` is true or a list of item ids")),
    };
    if wanted.len() > filmcraft_templates::manifest::MAX_ASSETS {
        return Err(bad(CMD, format!("at most {} bundled files", filmcraft_templates::manifest::MAX_ASSETS)));
    }
    let mut total: u64 = 0;
    let mut out = Vec::new();
    for (item, path) in wanted {
        let size = s.services.file_size(&path).map_err(|e| bad(CMD, format!("bundling {path}: {e}")))?;
        total = total.saturating_add(size);
        if size > MAX_BUNDLE_FILE || total > MAX_BUNDLE_TOTAL {
            return Err(bad(CMD, format!("bundling {path}: at most {MAX_BUNDLE_FILE} bytes a file and {MAX_BUNDLE_TOTAL} in all")));
        }
        let bytes = s.services.read_file(&path).map_err(|e| bad(CMD, format!("bundling {path}: {e}")))?;
        let file = std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "asset".into());
        let item_name = s.project.item(item).map(|i| i.name.clone()).unwrap_or_else(|| file.clone());
        let sidecar =
            s.services.read_file(&format!("{path}.attribution")).ok().filter(|b| b.len() <= 64 * 1024).map(|b| read_attribution(&String::from_utf8_lossy(&b)));
        let author = if opts.author.trim().is_empty() { "the template's author".to_string() } else { opts.author.clone() };
        let (title, author, source, license) = match sidecar {
            Some(a) if a.contains_key("license") => (
                a.get("title").cloned().unwrap_or(item_name),
                a.get("author").cloned().unwrap_or(author),
                a.get("source").cloned().unwrap_or_else(|| "original work".into()),
                a.get("license").cloned().unwrap_or_default(),
            ),
            // the author's own file: theirs, under the template's licence
            _ => (item_name, author, "original work".to_string(), opts.license.clone()),
        };
        if filmcraft_templates::manifest::licence_refused(&license) {
            return Err(bad(CMD, format!("{path} is licensed `{license}`, which a template cannot bundle (no NC / ND or proprietary files)")));
        }
        out.push(filmcraft_templates::save::BundleFile { item, bytes, name: file, title, author, source, license });
    }
    Ok(out)
}

fn save(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.save";
    let seq_id = crate::commands::item_p(p, "sequence").or(s.state.active_sequence).ok_or_else(|| bad(CMD, "no sequence: open one or pass `sequence`"))?;
    let q = s.project.sequence(seq_id).ok_or_else(|| bad(CMD, "no such sequence"))?.clone();
    let seq_name = s.project.item(seq_id).map(|i| i.name.clone()).unwrap_or_default();
    let name = str_p(p, "name").map(str::to_string).filter(|n| !n.trim().is_empty()).unwrap_or(seq_name);
    if name.len() > 256 {
        return Err(bad(CMD, "`name` is longer than 256 bytes"));
    }
    let zip_default = bool_p(p, "zip").unwrap_or(false);
    // where: the path given, else the user templates folder
    let path = match str_p(p, "path") {
        Some(x) if x.trim().is_empty() || x.len() > 4096 || x.contains('\0') => return Err(bad(CMD, "`path` must be a .fcvideo-template folder or a .zip")),
        Some(x) => crate::export_tools::expand_home(x),
        None => {
            let dir = user_dir(s).ok_or_else(|| bad(CMD, "need `path` (a .fcvideo-template folder or a .zip): there is no data directory"))?;
            let stem = filmcraft_templates::slug(&name);
            let ext = if zip_default { "zip".to_string() } else { EXTENSION.to_string() };
            let mut out = dir.join(format!("{stem}.{ext}"));
            let mut n = 2;
            while (s.services.file_size(&out.to_string_lossy()).is_ok() || is_dir(s, &out.to_string_lossy()))
                && !bool_p(p, "overwrite").unwrap_or(false)
                && n < 1000
            {
                out = dir.join(format!("{stem}-{n}.{ext}"));
                n += 1;
            }
            out.to_string_lossy().to_string()
        }
    };
    let mut warnings = Vec::new();
    let mut opts = save_options(s, p, &q, seq_id, &name, &mut warnings)?;
    opts.bundle = bundle_files(s, p, &q, &opts)?;
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
    let saved = filmcraft_templates::save::save(&s.project, seq_id, &opts, grid).map_err(|e| bad(CMD, e.to_string()))?;
    warnings.extend(saved.warnings.iter().cloned());
    let overwrite = bool_p(p, "overwrite").unwrap_or(false);
    let zip = path.to_ascii_lowercase().ends_with(".zip");
    let out = if zip || path.to_ascii_lowercase().ends_with(&format!(".{EXTENSION}")) { path.clone() } else { format!("{path}.{EXTENSION}") };
    if !overwrite && (s.services.file_size(&out).is_ok() || is_dir(s, &out)) {
        return Err(bad(CMD, format!("{out} exists (pass `overwrite: true` to replace it)")));
    }
    if zip {
        let bytes = saved.package.to_zip().map_err(|e| bad(CMD, e.to_string()))?;
        if !cfg!(target_arch = "wasm32")
            && let Some(parent) = std::path::Path::new(&out).parent().filter(|d| !d.as_os_str().is_empty())
        {
            std::fs::create_dir_all(parent).map_err(|e| EngineError::Other(format!("{}: {e}", parent.display())))?;
        }
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
    bump_library();
    let in_library = user_dir(s).is_some_and(|d| std::path::Path::new(&out).parent() == Some(d.as_path()));
    let assets: Vec<Value> = saved
        .package
        .manifest
        .assets
        .iter()
        .map(|a| json!({"path": a.path, "sha256": a.sha256, "title": a.title, "author": a.author, "source": a.source, "license": a.license}))
        .collect();
    let e = Entry { package: saved.package, source: if in_library { "user" } else { "file" }, path: Some(out.clone()), asset_dir: None };
    Ok(json!({"path": out, "inLibrary": in_library, "template": summary(&e), "assets": assets, "warnings": warnings}))
}

fn mark_slot(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.markSlot";
    let clear = bool_p(p, "clear").unwrap_or(false);
    let kind = match str_p(p, "kind") {
        Some(k) => Some(SlotKind::from_id(k).ok_or_else(|| bad(CMD, format!("unknown kind `{k}` (video, photo, text, logo, music)")))?),
        None if clear => None,
        None => return Err(bad(CMD, "need `kind` (video, photo, text, logo or music)")),
    };
    if kind == Some(SlotKind::Captions) {
        return Err(bad(CMD, "a captions slot has no clips: it captions the slots it names (author it in the manifest)"));
    }
    let slot = str_p(p, "slot").map(str::to_string).or_else(|| kind.map(|k| k.id().to_string())).unwrap_or_default();
    if !clear && !filmcraft_templates::manifest::is_slot_id(&slot) {
        return Err(bad(CMD, format!("`{slot}`: slot ids use lower-case letters, digits, `_` and `-`")));
    }
    let clips: Vec<ClipId> = crate::commands::clips_p(s, p);
    if clips.is_empty() {
        return Err(bad(CMD, "select clips or pass `clips`"));
    }
    let seq = s.active_sequence().ok_or(EngineError::NoSequence)?;
    // the clips of the selection that fit the kind (a linked selection holds both halves of a
    // clip: a picture slot takes the picture, a music slot the sound)
    let mut targets: Vec<ClipId> = Vec::new();
    let mut refused: Option<String> = None;
    for c in &clips {
        let (tid, it) = seq.find_item(*c).ok_or_else(|| bad(CMD, format!("no clip {} in the active sequence", c.0)))?;
        let track_kind = seq.track(tid).map(|t| t.kind).unwrap_or_default();
        let why = match kind {
            Some(SlotKind::Text) if !matches!(s.project.item(it.item).map(|i| &i.kind), Some(filmcraft_project::ItemKind::Graphic { .. })) => {
                Some(format!("“{}” is not a graphic: text slots are graphic clips with a text layer", it.name))
            }
            Some(SlotKind::Music) if track_kind != TrackKind::Audio => Some(format!("“{}” is not on an audio track", it.name)),
            Some(k) if k.is_visual() && track_kind != TrackKind::Video => Some(format!("“{}” is not on a video track", it.name)),
            _ => None,
        };
        match why {
            None => targets.push(*c),
            Some(w) => {
                refused.get_or_insert(w);
            }
        }
    }
    let fitted = targets.len();
    // a picture slot's clips bring their linked sound along, as an applied template's do
    if kind.is_some_and(SlotKind::is_visual) && !clear {
        let links: Vec<u64> = targets.iter().filter_map(|c| seq.find_item(*c).and_then(|(_, it)| it.link)).collect();
        for it in seq.audio_tracks.iter().flat_map(|t| t.items.iter()) {
            if it.link.is_some_and(|l| links.contains(&l)) && !targets.contains(&it.id) {
                targets.push(it.id);
            }
        }
    }
    if targets.is_empty() {
        return Err(bad(CMD, refused.unwrap_or_else(|| "no clip fits".into())));
    }
    let clips = targets;
    let n: usize = s.edit_sequence(if clear { "Clear Template Slot" } else { "Mark Template Slot" }, |q, ctx, _| {
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
    let linked = n.saturating_sub(fitted);
    Ok(json!({"clips": n - linked, "linked": linked, "kind": kind.map(|k| k.id()), "slot": if clear { Value::Null } else { json!(slot) }}))
}
