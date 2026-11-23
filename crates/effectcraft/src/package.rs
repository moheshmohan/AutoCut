//! The `.ectemplate` package: a ZIP (read by [`filmcraft_templates::zip::read`], which caps the
//! entries, their sizes and what they inflate to, and refuses names leaving the package) holding
//! `manifest.json`, `project.ecproj`, `media/<item>/<file>` and `poster.png`. This part needs no
//! EffectCraft: it is the same in every build.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::{Map, Value};

use crate::limits::{MAX_MANIFEST_BYTES, MAX_PROJECT_BYTES, MAX_TEMPLATE_BYTES};
use crate::{Error, Result};

/// The manifest's `format`.
pub const FORMAT: &str = "effectcraft-template";
/// The newest manifest `version` understood.
pub const VERSION: u64 = 1;

/// A template package, read and checked.
#[derive(Clone, Debug, PartialEq)]
pub struct Package {
    pub manifest: Map<String, Value>,
    /// The id of the comp the template renders (the manifest's `comp`).
    pub comp: u64,
    /// The template's name (the manifest's `name`, else `Template`).
    pub name: String,
    /// `project.ecproj`, as text.
    pub project: String,
    /// The media it carries: `media/<item>/<file>` → bytes.
    pub media: BTreeMap<String, Vec<u8>>,
    pub poster: Option<Vec<u8>>,
    /// Entries it holds that a template does not use.
    pub ignored: Vec<String>,
}

fn bad(m: impl Into<String>) -> Error {
    Error::Template(m.into())
}

/// Read and check a template package.
pub fn read(bytes: &[u8]) -> Result<Package> {
    if bytes.len() as u64 > MAX_TEMPLATE_BYTES {
        return Err(Error::TooLarge(format!("the template is {} MiB; at most {} MiB is read", bytes.len() >> 20, MAX_TEMPLATE_BYTES >> 20)));
    }
    let mut entries = filmcraft_templates::zip::read(bytes).map_err(|e| bad(format!("not a readable template (.ectemplate is a ZIP): {e}")))?;
    let manifest = entries.remove("manifest.json").ok_or_else(|| bad("not an EffectCraft template: it has no manifest.json"))?;
    if manifest.len() > MAX_MANIFEST_BYTES {
        return Err(Error::TooLarge(format!("manifest.json is {} KiB; at most {} KiB", manifest.len() >> 10, MAX_MANIFEST_BYTES >> 10)));
    }
    let manifest: Value = serde_json::from_slice(&manifest).map_err(|e| bad(format!("manifest.json is not valid JSON: {e}")))?;
    let Value::Object(manifest) = manifest else { return Err(bad("manifest.json is not a JSON object")) };
    if manifest.get("format").and_then(Value::as_str) != Some(FORMAT) {
        return Err(bad(format!("not an EffectCraft template: manifest.json's `format` is not `{FORMAT}`")));
    }
    match manifest.get("version") {
        None | Some(Value::Null) => {}
        Some(v) => match v.as_u64() {
            Some(n) if n <= VERSION => {}
            Some(n) => return Err(bad(format!("the template is version {n}, made by a newer EffectCraft (version {VERSION} is read)"))),
            None => return Err(bad("manifest.json's `version` is not a whole number")),
        },
    }
    let comp = manifest.get("comp").and_then(Value::as_u64).ok_or_else(|| bad("manifest.json names no `comp` (the composition's id)"))?;
    let name = manifest
        .get("name")
        .and_then(Value::as_str)
        .map(|n| n.chars().take(256).collect::<String>())
        .filter(|n| !n.trim().is_empty())
        .unwrap_or_else(|| "Template".into());
    let project = entries.remove("project.ecproj").ok_or_else(|| bad("the template has no project.ecproj"))?;
    if project.len() > MAX_PROJECT_BYTES {
        return Err(Error::TooLarge(format!("project.ecproj is {} MiB; at most {} MiB", project.len() >> 20, MAX_PROJECT_BYTES >> 20)));
    }
    let project = String::from_utf8(project).map_err(|_| bad("project.ecproj is not UTF-8 text"))?;
    let poster = entries.remove("poster.png");
    let mut media = BTreeMap::new();
    let mut ignored = Vec::new();
    for (k, v) in entries {
        if media_entry(&k).is_some() {
            media.insert(k, v);
        } else {
            ignored.push(k);
        }
    }
    Ok(Package { manifest, comp, name, project, media, poster, ignored })
}

/// `media/<item>/<file>` split: the item's id and a plain file name (the ZIP reader already
/// refused names that leave the package).
pub fn media_entry(entry: &str) -> Option<(u64, &str)> {
    let (item, file) = entry.strip_prefix("media/")?.split_once('/')?;
    if item.is_empty() || item.len() > 19 || !item.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let bare = !file.is_empty() && file != "." && file != ".." && !file.contains(['/', '\\', '\0', ':']) && file.len() <= 255;
    if !bare {
        return None;
    }
    Some((item.parse().ok()?, file))
}

/// Where media entry `entry` is written under `dir`: `<dir>/<item>/<file>`.
pub fn media_path(dir: &Path, entry: &str) -> Option<PathBuf> {
    let (item, file) = media_entry(entry)?;
    Some(dir.join(item.to_string()).join(file))
}

/// A PNG's size from its header.
pub fn png_size(png: &[u8]) -> Option<(u32, u32)> {
    if png.get(..8)? != b"\x89PNG\r\n\x1a\n" || png.get(12..16)? != b"IHDR" {
        return None;
    }
    let be = |at: usize| -> Option<u32> { Some(u32::from_be_bytes(png.get(at..at + 4)?.try_into().ok()?)) };
    Some((be(16)?, be(20)?))
}
