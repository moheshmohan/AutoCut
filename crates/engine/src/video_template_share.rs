//! Sharing video templates as plain files (no server, no account) and their timelines as
//! OpenTimelineIO.
//!
//! | command | does |
//! |---|---|
//! | `template.export` | a template (built-in, saved or a file) as one `.zip` package at `path` (a folder: `<name>.zip` in it) |
//! | `template.install` | a `.zip` package from `path`, or from an https `url` (a job: downloaded through the host's fetcher, at most [`MAX_INSTALL_BYTES`]); checked whole in memory — the ZIP's caps, names that leave the package, the strict manifest, the sequence — before anything is written; then written, re-encoded from what was read, into the user templates folder, where the browser shows it. Nothing in a package is ever run |
//! | `template.qr` | a QR code of `url` (an install link): a PNG at `path`, and its modules |
//! | `template.exportOtio` | the template's timeline as OpenTimelineIO (`variant`: laid out for that layout), its slots in `metadata.filmcraft` (the timeline's `template`, each placement clip's `slot`) |
//! | `sequence.exportOtio` | a sequence as OpenTimelineIO, the slot flags of its clips (`template.markSlot`, an applied template's) in the clips' `metadata.filmcraft.slot` |
//!
//! OTIO has no slots: they ride in the metadata, and OTIO readers carry metadata through.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use filmcraft_project::{ItemId, Project, Sequence};
use filmcraft_templates::apply::{ApplyOptions, parse_slot_marker};
use filmcraft_templates::qr::{EcLevel, QrCode};
use filmcraft_templates::{Package, TemplateError};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad, bool_p, has_seq, str_p, u64_p};
use crate::video_templates::{self as vt, find_template};
use crate::{EngineError, Result, Session};

/// Largest package `template.install` takes (a file or a download).
pub const MAX_INSTALL_BYTES: u64 = 128 * 1024 * 1024;
/// Longest install link.
pub const MAX_URL: usize = 2048;
/// Longest text a QR code holds here (an install link).
const MAX_QR_TEXT: usize = 2048;

/// Downloads a package for `template.install {url}`, installed by the host (the desktop app, the
/// CLI and the MCP server: pure-Rust HTTPS). None: installing from a link is not available.
pub trait TemplateFetcher: Send + Sync {
    /// GET `url` (https only, redirects to http refused) and return its body; a body larger than
    /// `max_bytes` is an error, not a cut. Blocking (called on a job's thread); `cancelled` says
    /// when to stop.
    fn fetch(&self, url: &str, max_bytes: u64, cancelled: &dyn Fn() -> bool) -> std::result::Result<Vec<u8>, String>;
}

/// Sharing in a session: the host's fetcher.
#[derive(Default)]
pub struct Sharing {
    pub fetcher: Option<Arc<dyn TemplateFetcher>>,
}

type Run = fn(&mut Session, &Value) -> Result<Value>;

fn spec(id: &'static str, label: &'static str, params: &'static str, enabled: fn(&Session) -> std::result::Result<(), String>, run: Run) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled, run, journal: false }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec("template.export", "Export Video Template", r#"{"template":id|name|path,"path":str (.zip, or a folder),"overwrite":bool=false}"#, always, export),
        spec(
            "template.install",
            "Install Video Template",
            r#"{"path":str (.zip)?,"url":"https://…"?,"overwrite":bool=false,"wait":bool=false}"#,
            always,
            install,
        ),
        spec(
            "template.qr",
            "Video Template QR Code",
            r#"{"url":str,"path":str (.png)?,"scale":1..64=8,"border":0..16=4,"ec":"L"|"M"|"Q"|"H"="M"}"#,
            always,
            qr,
        ),
        spec(
            "template.exportOtio",
            "Export Video Template as OpenTimelineIO",
            r#"{"template":id|name|path,"path":str (.otio),"variant":"16:9"|"9:16"|…?,"overwrite":bool=false}"#,
            always,
            export_otio,
        ),
        spec(
            "sequence.exportOtio",
            "Export Sequence as OpenTimelineIO",
            r#"{"path":str (.otio),"sequence":id?,"overwrite":bool=false}"#,
            has_seq,
            sequence_otio,
        ),
    ]
}

fn exists(s: &Session, path: &str) -> bool {
    s.services.file_size(path).is_ok()
}

fn is_folder(s: &Session, path: &str) -> bool {
    path.ends_with('/') || path.ends_with('\\') || (s.services.file_size(path).is_err() && s.services.list_entries(path).is_some_and(|r| r.is_ok()))
}

fn join(dir: &str, name: &str) -> String {
    let sep = if dir.ends_with('/') || dir.ends_with('\\') { "" } else { "/" };
    format!("{dir}{sep}{name}")
}

/// A path parameter: given, not empty, without NUL, at most 4096 bytes, `~` expanded.
fn path_param(p: &Value, key: &str, cmd: &str) -> Result<String> {
    let v = str_p(p, key).ok_or_else(|| bad(cmd, format!("need `{key}`")))?;
    if v.trim().is_empty() || v.len() > 4096 || v.contains('\0') {
        return Err(bad(cmd, format!("`{key}` must be a path")));
    }
    Ok(crate::export_tools::expand_home(v))
}

fn write(s: &Session, path: &str, bytes: &[u8]) -> Result<()> {
    if !cfg!(target_arch = "wasm32")
        && let Some(parent) = std::path::Path::new(path).parent().filter(|d| !d.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|e| EngineError::Other(format!("{}: {e}", parent.display())))?;
    }
    s.services.write_file(path, bytes).map_err(|e| EngineError::Other(format!("{path}: {e}")))
}

// ------------------------------------------------------------------------------------------------
// Export and install
// ------------------------------------------------------------------------------------------------

fn export(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.export";
    let key = str_p(p, "template").or_else(|| str_p(p, "id")).unwrap_or("");
    let e = find_template(s, key, CMD)?;
    let mut path = path_param(p, "path", CMD)?;
    if is_folder(s, &path) {
        path = join(&path, &format!("{}.zip", filmcraft_templates::slug(&e.package.manifest.name)));
    } else if !path.to_ascii_lowercase().ends_with(".zip") {
        path.push_str(".zip");
    }
    if exists(s, &path) && !bool_p(p, "overwrite").unwrap_or(false) {
        return Err(bad(CMD, format!("{path} exists (pass `overwrite: true` to replace it)")));
    }
    let bytes = e.package.to_zip().map_err(|err| bad(CMD, err.to_string()))?;
    write(s, &path, &bytes)?;
    Ok(
        json!({"path": path, "bytes": bytes.len(), "template": e.package.manifest.id, "name": e.package.manifest.name, "sha256": filmcraft_templates::sha256_hex(&bytes)}),
    )
}

/// Whether `url` is an install link this build fetches: `https://host/…`, no credentials, no
/// spaces or control characters, at most [`MAX_URL`] bytes.
pub fn check_url(url: &str) -> std::result::Result<(), String> {
    if url.len() > MAX_URL {
        return Err(format!("the link is longer than {MAX_URL} bytes"));
    }
    let Some(rest) = url.strip_prefix("https://") else { return Err("only https:// links are installed".into()) };
    if url.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err("the link has spaces or control characters".into());
    }
    let host = rest.split(['/', '?', '#']).next().unwrap_or("");
    if host.is_empty() || host.starts_with(':') {
        return Err("the link has no host".into());
    }
    if host.contains('@') {
        return Err("links with a user name or password are not installed".into());
    }
    Ok(())
}

/// What an install did.
#[derive(Clone, Debug)]
pub struct Installed {
    pub path: String,
    pub package: Package,
    pub replaced: bool,
    pub warnings: Vec<String>,
}

/// Check `bytes` as a package and write it into the user templates folder `dir` (re-encoded
/// from what was read, so only checked content is written). `existing` are the user templates
/// already there (id → path).
pub fn install_bytes(
    services: &dyn crate::Services,
    dir: &str,
    bytes: &[u8],
    existing: &HashMap<String, String>,
    overwrite: bool,
) -> std::result::Result<Installed, String> {
    if bytes.len() as u64 > MAX_INSTALL_BYTES {
        return Err(format!("the package is {} bytes (at most {MAX_INSTALL_BYTES})", bytes.len()));
    }
    if !filmcraft_templates::zip::is_zip(bytes) {
        return Err("not a template package (a .zip)".into());
    }
    let package = Package::from_zip(bytes).map_err(|e| e.to_string())?;
    let id = package.manifest.id.clone();
    if id.starts_with("builtin:") || filmcraft_templates::builtin::find(&id).is_some() {
        return Err(format!("`{id}` is a built-in template's id; a package cannot replace a built-in template"));
    }
    let zip = package.to_zip().map_err(|e| e.to_string())?;
    let (path, replaced) = match existing.get(&id) {
        Some(_) if !overwrite => return Err(format!("a template `{id}` is installed already (pass `overwrite: true` to replace it)")),
        Some(p) if p.to_ascii_lowercase().ends_with(".zip") => (p.clone(), true),
        Some(p) => return Err(format!("`{id}` is installed as the folder {p}; remove it first to install this package")),
        None => {
            let stem = filmcraft_templates::slug(id.split_once(':').map_or(id.as_str(), |(_, r)| r));
            let mut path = join(dir, &format!("{stem}.zip"));
            let mut n = 2;
            while services.file_size(&path).is_ok() && n < 1000 {
                path = join(dir, &format!("{stem}-{n}.zip"));
                n += 1;
            }
            (path, false)
        }
    };
    if !cfg!(target_arch = "wasm32") {
        std::fs::create_dir_all(dir).map_err(|e| format!("{dir}: {e}"))?;
    }
    services.write_file(&path, &zip).map_err(|e| format!("{path}: {e}"))?;
    vt::bump_library();
    Ok(Installed { path, package, replaced, warnings: Vec::new() })
}

fn installed_json(s: &Session, i: &Installed) -> Value {
    let mut warnings = i.warnings.clone();
    warnings.extend(vt::package_checks(s, &i.package).into_iter().map(|(f, m)| format!("{f}: {m}")));
    json!({
        "path": i.path,
        "template": i.package.manifest.id,
        "name": i.package.manifest.name,
        "replaced": i.replaced,
        "warnings": warnings,
    })
}

fn existing_user_templates(s: &Session) -> HashMap<String, String> {
    vt::library(s).0.into_iter().filter(|e| e.source == "user").filter_map(|e| Some((e.package.manifest.id.clone(), e.path?))).collect()
}

fn install(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.install";
    let dir = vt::user_dir(s).ok_or_else(|| bad(CMD, "no data directory for user templates"))?.to_string_lossy().to_string();
    let overwrite = bool_p(p, "overwrite").unwrap_or(false);
    match (str_p(p, "path"), str_p(p, "url")) {
        (Some(_), Some(_)) => Err(bad(CMD, "give `path` or `url`, not both")),
        (None, None) => Err(bad(CMD, "need `path` (a .zip package) or `url` (an https:// link to one)")),
        (Some(_), None) => {
            let path = path_param(p, "path", CMD)?;
            let size = s.services.file_size(&path).map_err(|e| bad(CMD, format!("{path}: {e}")))?;
            if size > MAX_INSTALL_BYTES {
                return Err(bad(CMD, format!("{path} is {size} bytes (at most {MAX_INSTALL_BYTES})")));
            }
            let bytes = s.services.read_file(&path).map_err(|e| bad(CMD, format!("{path}: {e}")))?;
            let existing = existing_user_templates(s);
            let i = install_bytes(s.services.as_ref(), &dir, &bytes, &existing, overwrite).map_err(|e| bad(CMD, format!("{path}: {e}")))?;
            Ok(installed_json(s, &i))
        }
        (None, Some(url)) => {
            let url = url.trim().to_string();
            check_url(&url).map_err(|e| bad(CMD, e))?;
            let fetcher = s
                .sharing
                .fetcher
                .clone()
                .ok_or_else(|| bad(CMD, "installing from a link is not available in this app; download the .zip and install the file"))?;
            if cfg!(target_arch = "wasm32") {
                return Err(bad(CMD, "installing from a link is not available on the web"));
            }
            let wait = bool_p(p, "wait").unwrap_or(false);
            let existing = existing_user_templates(s);
            let services = s.services.clone();
            let job = crate::narration::new_job(s, format!("Installing a template from {}", url.chars().take(80).collect::<String>()));
            *job.progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = "Downloading…".into();
            let (prog, res) = (job.progress.clone(), job.result.clone());
            let id = crate::narration::spawn(s, job, wait, "filmcraft-template-install", move || {
                let t0 = web_time::Instant::now();
                let cancelled = || prog.cancel.load(Ordering::Relaxed);
                let r = fetcher
                    .fetch(&url, MAX_INSTALL_BYTES, &cancelled)
                    .and_then(|bytes| if cancelled() { Err("cancelled".into()) } else { Ok(bytes) })
                    .and_then(|bytes| install_bytes(services.as_ref(), &dir, &bytes, &existing, overwrite));
                let r = match r {
                    Ok(i) => {
                        let path = i.path.clone();
                        *prog.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = format!("Installed “{}”", i.package.manifest.name);
                        prog.finished.store(true, Ordering::Relaxed);
                        Ok(filmcraft_export::Report { path, frames: 0, seconds: t0.elapsed().as_secs_f64(), bytes: 0, render_fps: 0.0, ..Default::default() })
                    }
                    Err(e) => {
                        *prog.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = e.clone();
                        *prog.error.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(e.clone());
                        prog.finished.store(true, Ordering::Relaxed);
                        Err(e)
                    }
                };
                *res.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(r);
            })?;
            let mut out = json!({"job": id});
            if wait && let Some(j) = s.jobs.iter().find(|j| j.id == id) {
                let v = j.to_json();
                if let Some(e) = v["result"]["error"].as_str() {
                    return Err(bad(CMD, e.to_string()));
                }
                out["path"] = v["result"]["path"].clone();
            }
            Ok(out)
        }
    }
}

// ------------------------------------------------------------------------------------------------
// QR codes
// ------------------------------------------------------------------------------------------------

/// A QR code of `text` at level `ec`.
pub fn qr_code(text: &str, ec: EcLevel) -> std::result::Result<QrCode, String> {
    if text.len() > MAX_QR_TEXT {
        return Err(format!("at most {MAX_QR_TEXT} bytes"));
    }
    QrCode::encode_text(text, ec).map_err(|e| e.to_string())
}

fn qr(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.qr";
    let text = str_p(p, "url").or_else(|| str_p(p, "text")).ok_or_else(|| bad(CMD, "need `url` (the link to put in the code)"))?;
    let ec = match str_p(p, "ec") {
        Some(e) => EcLevel::from_id(e).ok_or_else(|| bad(CMD, format!("`ec` is L, M, Q or H, not `{e}`")))?,
        None => EcLevel::M,
    };
    let code = qr_code(text, ec).map_err(|e| bad(CMD, e))?;
    let scale = u64_p(p, "scale").unwrap_or(8).clamp(1, 64) as u32;
    let border = u64_p(p, "border").unwrap_or(4).min(16) as u32;
    let mut out = json!({"version": code.version, "size": code.size, "ec": format!("{:?}", code.ec), "mask": code.mask, "modules": code.rows()});
    if str_p(p, "path").is_some() {
        let mut path = path_param(p, "path", CMD)?;
        if !path.to_ascii_lowercase().ends_with(".png") {
            path.push_str(".png");
        }
        let (w, h, px) = code.to_rgba(scale, border);
        let png = filmcraft_export::encode_png(px, w, h).map_err(|e| bad(CMD, e.to_string()))?;
        write(s, &path, &png)?;
        out["path"] = json!(path);
        out["width"] = json!(w);
        out["height"] = json!(h);
    }
    Ok(out)
}

// ------------------------------------------------------------------------------------------------
// OpenTimelineIO
// ------------------------------------------------------------------------------------------------

/// Put each clip's slot (by clip id) in its OTIO clip's `metadata.filmcraft.slot`. The writer
/// emits a track's items in order, with gaps and transitions between them; a track whose items
/// cannot be matched one to one is left as it is.
fn annotate_clips(doc: &mut Value, seq: &Sequence, slot_of: &HashMap<u64, Value>) -> usize {
    let mut n = 0;
    let Some(tracks) = doc.pointer_mut("/tracks/children").and_then(Value::as_array_mut) else { return 0 };
    let ours = seq.video_tracks.iter().chain(seq.audio_tracks.iter());
    for (tj, t) in tracks.iter_mut().zip(ours) {
        let Some(children) = tj.get_mut("children").and_then(Value::as_array_mut) else { continue };
        let mut items: Vec<&mut Value> = children
            .iter_mut()
            .filter(|c| !matches!(c.get("OTIO_SCHEMA").and_then(Value::as_str), Some(s) if s.starts_with("Gap.") || s.starts_with("Transition.")))
            .collect();
        if items.len() != t.items.len() {
            continue;
        }
        for (cj, c) in items.iter_mut().zip(&t.items) {
            if let Some(slot) = slot_of.get(&c.id.0) {
                if !cj.get("metadata").is_some_and(Value::is_object) {
                    cj["metadata"] = json!({});
                }
                if !cj["metadata"].get("filmcraft").is_some_and(Value::is_object) {
                    cj["metadata"]["filmcraft"] = json!({});
                }
                cj["metadata"]["filmcraft"]["slot"] = slot.clone();
                n += 1;
            }
        }
    }
    n
}

/// The slot flags of a sequence's clips (`template.markSlot`, an applied template's).
fn flags_of(seq: &Sequence) -> HashMap<u64, Value> {
    let mut out = HashMap::new();
    for t in seq.video_tracks.iter().chain(seq.audio_tracks.iter()) {
        for c in &t.items {
            let flags: Vec<Value> = c
                .markers
                .iter()
                .filter_map(|m| parse_slot_marker(&m.name))
                .take(16)
                .map(|f| json!({"kind": f.kind.id(), "id": f.slot, "placement": f.placement, "layer": f.layer}))
                .collect();
            match flags.len() {
                0 => {}
                1 => {
                    out.insert(c.id.0, flags.into_iter().next().unwrap_or(Value::Null));
                }
                _ => {
                    out.insert(c.id.0, Value::Array(flags));
                }
            }
        }
    }
    out
}

fn otio_bytes(
    project: &Project,
    seq_id: ItemId,
    name: &str,
    slot_of: &HashMap<u64, Value>,
    template: Option<Value>,
    cmd: &str,
) -> Result<(Vec<u8>, usize, Vec<String>)> {
    let seq = project.sequence(seq_id).ok_or_else(|| bad(cmd, "no such sequence"))?;
    let opts = filmcraft_interchange::ExportOptions { name: Some(name.to_string()), ..Default::default() };
    let (bytes, report) = filmcraft_interchange::export(project, seq_id, filmcraft_interchange::Format::Otio, &opts).map_err(|e| bad(cmd, e.to_string()))?;
    let mut doc: Value = serde_json::from_slice(&bytes).map_err(|e| bad(cmd, format!("the OTIO writer wrote no JSON: {e}")))?;
    let n = annotate_clips(&mut doc, seq, slot_of);
    if let Some(t) = template {
        if !doc.get("metadata").is_some_and(Value::is_object) {
            doc["metadata"] = json!({});
        }
        if !doc["metadata"].get("filmcraft").is_some_and(Value::is_object) {
            doc["metadata"]["filmcraft"] = json!({});
        }
        doc["metadata"]["filmcraft"]["template"] = t;
    }
    let out = serde_json::to_vec_pretty(&doc).map_err(|e| bad(cmd, e.to_string()))?;
    let notes = report.entries.iter().map(|e| e.message.clone()).collect();
    Ok((out, n, notes))
}

fn otio_path(s: &Session, p: &Value, cmd: &str) -> Result<String> {
    let mut path = path_param(p, "path", cmd)?;
    if !path.to_ascii_lowercase().ends_with(".otio") {
        path.push_str(".otio");
    }
    if exists(s, &path) && !bool_p(p, "overwrite").unwrap_or(false) {
        return Err(bad(cmd, format!("{path} exists (pass `overwrite: true` to replace it)")));
    }
    Ok(path)
}

fn export_otio(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "template.exportOtio";
    let key = str_p(p, "template").or_else(|| str_p(p, "id")).unwrap_or("");
    let e = find_template(s, key, CMD)?;
    let path = otio_path(s, p, CMD)?;
    let m = &e.package.manifest;
    let variant = str_p(p, "variant");
    if let Some(v) = variant
        && m.variant(Some(v)).is_none()
    {
        let ids: Vec<&str> = m.variants.iter().map(|v| v.id.as_str()).collect();
        return Err(bad(CMD, format!("the template has no variant `{v}` (its variants: {})", ids.join(", "))));
    }
    let design = m.variant(None).map(|v| v.id.clone()).unwrap_or_default();
    let laid_out = variant.is_some_and(|v| !v.eq_ignore_ascii_case(&design));
    let template = json!({
        "id": m.id, "name": m.name, "version": m.version, "author": m.author, "license": m.license,
        "variant": variant.unwrap_or(&design), "manifest": serde_json::to_value(m).unwrap_or_default(),
    });
    let (bytes, n, notes) = if laid_out {
        // the layout as applied (placeholders), its clips flagged by the apply
        let mut scratch = Project::new("OTIO");
        let opts = ApplyOptions { variant: variant.map(str::to_string), ..Default::default() };
        let a = filmcraft_templates::apply::apply(&mut scratch, &e.package, &opts).map_err(|err: TemplateError| bad(CMD, err.to_string()))?;
        let seq = scratch.sequence(a.sequence).ok_or_else(|| bad(CMD, "the template has no sequence"))?;
        let flags = flags_of(seq);
        otio_bytes(&scratch, a.sequence, &m.name, &flags, Some(template), CMD)?
    } else {
        let mut slot_of: HashMap<u64, Value> = HashMap::new();
        for sl in &m.slots {
            for (i, pl) in sl.placements.iter().enumerate() {
                let v = json!({"kind": sl.kind.id(), "id": sl.id, "name": sl.name, "placement": i, "media": pl.media, "layer": pl.layer, "box": pl.area});
                match slot_of.get_mut(&pl.clip) {
                    // two text slots on one graphic
                    Some(Value::Array(a)) => a.push(v),
                    Some(one) => *one = Value::Array(vec![one.clone(), v]),
                    None => {
                        slot_of.insert(pl.clip, v);
                    }
                }
            }
        }
        otio_bytes(&e.package.fragment, ItemId(m.sequence), &m.name, &slot_of, Some(template), CMD)?
    };
    write(s, &path, &bytes)?;
    Ok(json!({"path": path, "bytes": bytes.len(), "slotClips": n, "variant": variant.unwrap_or(&design), "report": notes}))
}

fn sequence_otio(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "sequence.exportOtio";
    let seq_id = crate::commands::item_p(p, "sequence").or(s.state.active_sequence).ok_or(EngineError::NoSequence)?;
    let seq = s.project.sequence(seq_id).ok_or_else(|| bad(CMD, "no such sequence"))?;
    let path = otio_path(s, p, CMD)?;
    let flags = flags_of(seq);
    let it = s.project.item(seq_id);
    let name = it.map(|i| i.name.clone()).unwrap_or_default();
    let template = it.and_then(|i| i.metadata.get("Template")).map(|t| json!({"id": t, "variant": it.and_then(|i| i.metadata.get("Template Variant"))}));
    let (bytes, n, notes) = otio_bytes(&s.project, seq_id, &name, &flags, template, CMD)?;
    write(s, &path, &bytes)?;
    Ok(json!({"path": path, "bytes": bytes.len(), "slotClips": n, "report": notes}))
}
