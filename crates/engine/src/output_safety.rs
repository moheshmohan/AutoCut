//! Prevent generated artifacts from replacing the open project or imported originals.
use crate::{Result, Session, commands::bad};
use filmcraft_project::{ItemKind, MediaRef, Project};
use std::path::{Component, Path, PathBuf};

pub(crate) fn normalized(path: &str) -> PathBuf {
    let path = Path::new(path);
    let absolute = if path.is_absolute() { path.to_path_buf() } else { std::env::current_dir().unwrap_or_default().join(path) };
    let mut result = PathBuf::new();
    for part in absolute.components() {
        match part {
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            _ => result.push(part.as_os_str()),
        }
    }
    result
}

pub(crate) fn aliases(a: &str, b: &str) -> bool {
    if normalized(a) == normalized(b) {
        return true;
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        same_file::is_same_file(a, b).unwrap_or(false)
    }
    #[cfg(target_arch = "wasm32")]
    {
        false
    }
}

pub(crate) fn protect(s: &Session, project: &Project, path: &str, command: &str) -> Result<()> {
    if s.path.as_deref().is_some_and(|saved| aliases(path, saved)) {
        return Err(bad(command, "output would replace the saved project; choose an export/report path"));
    }
    for item in project.items.values() {
        if let ItemKind::Media(media) = &item.kind
            && let MediaRef::File { path: source } = &media.media
            && aliases(path, source)
        {
            return Err(bad(command, format!("output would replace imported source {source}; choose a new destination")));
        }
    }
    Ok(())
}
