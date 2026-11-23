//! EffectCraft templates in process (PX4): `.ectemplate` packages (Essential Graphics templates)
//! read and rendered by EffectCraft's own renderer, linked with the cargo feature `effectcraft`
//! ([`filmcraft_effectcraft`]; the packaged AutoCut app has it, workspace builds and the web do
//! not, and then every command here says so).
//!
//! | command | does |
//! |---|---|
//! | `effectcraft.templateInfo` | what a template is: its comp (size, rate, duration, frames), its controls (name, type, default, range or options), the media it carries or needs, its poster, whether it can be rendered here |
//!
//! The template file is read whole, at most [`filmcraft_effectcraft::limits::MAX_TEMPLATE_BYTES`];
//! everything in it is checked before EffectCraft sees it ([`filmcraft_effectcraft`]'s docs).

use std::path::Path;

use serde_json::{Value, json};

use crate::commands::{CommandSpec, always, bad};
use crate::motion_graphics::{path_p, str_of};
use crate::{Result, Session};

const INFO: &str = "effectcraft.templateInfo";

fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>, journal: bool) -> CommandSpec {
    CommandSpec { id, label, menu: &[], shortcut: None, params, enabled: always, run, journal }
}

pub fn commands() -> Vec<CommandSpec> {
    vec![spec(INFO, "EffectCraft Template Info", r#"{"template":path (an absolute .ectemplate)}"#, template_info, false)]
}

/// The `template` parameter: an absolute path to a file.
fn template_p(cmd: &str, p: &Value) -> Result<std::path::PathBuf> {
    let t = path_p(cmd, p, "template")?.ok_or_else(|| bad(cmd, "`template` is needed: the absolute path of an .ectemplate"))?;
    Ok(t)
}

/// The template's bytes: refused past the cap before anything is read.
fn read_template(cmd: &str, path: &Path) -> Result<Vec<u8>> {
    use std::io::Read as _;
    let max = filmcraft_effectcraft::limits::MAX_TEMPLATE_BYTES;
    let f = std::fs::File::open(path).map_err(|e| bad(cmd, format!("{}: {e}", path.display())))?;
    let len = f.metadata().map(|m| m.len()).unwrap_or(0);
    if len > max {
        return Err(bad(cmd, format!("{} is {} MiB; a template is at most {} MiB", path.display(), len >> 20, max >> 20)));
    }
    let mut bytes = Vec::new();
    f.take(max.saturating_add(1)).read_to_end(&mut bytes).map_err(|e| bad(cmd, format!("{}: {e}", path.display())))?;
    if bytes.len() as u64 > max {
        return Err(bad(cmd, format!("{} grew past {} MiB while it was read", path.display(), max >> 20)));
    }
    Ok(bytes)
}

/// Refused up front in a build without EffectCraft.
fn linked(cmd: &str) -> Result<()> {
    if filmcraft_effectcraft::available() { Ok(()) } else { Err(bad(cmd, filmcraft_effectcraft::Error::NotBuilt.to_string())) }
}

fn template_info(_: &mut Session, p: &Value) -> Result<Value> {
    let path = template_p(INFO, p)?;
    linked(INFO)?;
    let bytes = read_template(INFO, &path)?;
    let mut v = filmcraft_effectcraft::template_info(&bytes).map_err(|e| bad(INFO, format!("{}: {e}", path.display())))?;
    v["path"] = json!(str_of(&path));
    v["bytes"] = json!(bytes.len());
    Ok(v)
}

#[cfg(test)]
#[path = "effectcraft_tests.rs"]
mod tests;
