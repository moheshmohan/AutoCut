//! One saving run at a time per project file.
//!
//! A `--save` run reads the project, edits it and writes the whole file back, so two at once would
//! silently drop the edits of whichever saved first (#3924). Each run holds an OS lock from before
//! its read until it exits; the next one waits, then reads what the first one saved.
//!
//! The lock is on a file next to the project, not on the project: a save replaces the project's
//! inode, and on Windows a locked file can't be read by other processes. The OS drops the lock when
//! the process ends, however it ends, so a crash leaves nothing stale. The file itself stays, since
//! removing it would let one run wait on the old file while another locks a new one.

use crate::args::Args;
use std::fs::{File, TryLockError};
use std::path::{Path, PathBuf};

/// Most bytes of the project's file name kept in the lock's name; the rest of the 255-byte limit
/// is left for the leading dot and `.lock`.
const MAX_NAME_BYTES: usize = 200;

fn lock_path(project: &Path) -> PathBuf {
    let name = project.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    project.with_file_name(format!(".{}.lock", &name[..name.floor_char_boundary(MAX_NAME_BYTES)]))
}

/// Do two paths name one existing file?
fn same_file(a: &str, b: &str) -> bool {
    a == b || matches!((std::fs::canonicalize(a), std::fs::canonicalize(b)), (Ok(a), Ok(b)) if a == b)
}

/// Lock the path being replaced, keeping AutoCut's create-if-missing `.fcproj` workflow.
/// A new project is locked before creation too: another creator then opens its saved edits.
fn saved_in_place(a: &Args) -> Result<Option<PathBuf>, String> {
    let Some(project) = a.opt("--project").filter(|_| a.opt("--bridge").is_none()) else { return Ok(None) };
    let full = PathBuf::from(filmcraft_engine::export_tools::expand_home(project));
    if crate::missing_project(project).is_some() {
        return Ok(Some(full));
    }
    match full.metadata() {
        Ok(meta) if meta.is_file() => {}
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(format!("{}: {e}", full.display())),
        _ => return Ok(None),
    }
    let target = a.opt("--save-as").map(filmcraft_engine::export_tools::expand_home);
    match target {
        Some(target) if same_file(&target, &full.to_string_lossy()) => Ok(Some(PathBuf::from(target))),
        None if a.flag("--save") => Ok(Some(full)),
        _ => Ok(None),
    }
}

/// Lock `project`, calling `waiting` first if another run holds the lock.
fn lock(project: &Path, waiting: impl FnOnce()) -> std::io::Result<File> {
    let file = File::options().create(true).truncate(false).read(true).write(true).open(lock_path(project))?;
    match file.try_lock() {
        Ok(()) => {}
        Err(TryLockError::WouldBlock) => {
            waiting();
            file.lock()?;
        }
        Err(TryLockError::Error(e)) => return Err(e),
    }
    Ok(file)
}

/// Take the project's lock before reading it and keep it through the save.
/// A lock failure is an error: continuing could silently overwrite another run's edits.
pub fn for_save(a: &Args) -> Result<Option<File>, String> {
    let Some(project) = saved_in_place(a)? else { return Ok(None) };
    if crate::missing_project(&project.to_string_lossy()).is_some()
        && let Some(parent) = project.parent().filter(|p| !p.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let shown = project.display();
    let waiting = || crate::diag(format_args!("filmcraft-cli: waiting for another run saving {shown}"));
    lock(&project, waiting).map(Some).map_err(|e| format!("can't lock {shown}: {e}; no edits were made"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Args {
        Args::parse(list.iter().map(|s| s.to_string()))
    }

    /// The moment between this check and `Backend::open` can't be hit on demand, so this pins the
    /// decision that closes it: a run never goes on to open a project that wasn't there, however
    /// its paths are spelt and whatever it saves.
    #[test]
    fn a_missing_project_is_locked_before_automatic_creation() {
        let dir = std::env::temp_dir().join(format!("fc-cli-new-dir-{}", std::process::id()));
        let project = dir.join("nested/p.fcproj");
        let shown = project.to_str().unwrap();
        for extra in [&["--save"][..], &["--save-as", shown], &[]] {
            let list = [&["--project", shown][..], extra, &["exec"]].concat();
            assert_eq!(saved_in_place(&args(&list)).unwrap(), Some(project.clone()));
        }
        let typo = dir.join("p.fcprj");
        assert!(saved_in_place(&args(&["--project", typo.to_str().unwrap(), "--save", "exec"])).unwrap().is_none());
        assert!(saved_in_place(&args(&["--bridge", "127.0.0.1:1", "--project", shown, "--save", "exec"])).unwrap().is_none());
    }

    #[test]
    fn the_lock_sits_next_to_the_project() {
        assert_eq!(lock_path(Path::new("/a/b/p.fcproj")), Path::new("/a/b/.p.fcproj.lock"));
        assert_eq!(lock_path(Path::new("p.fcproj")), Path::new(".p.fcproj.lock"));
    }

    #[test]
    fn a_long_project_name_still_gets_a_lock_name_the_file_system_takes() {
        for name in ["a".repeat(255), "é".repeat(127)] {
            let lock = lock_path(Path::new(&name));
            let lock = lock.to_str().unwrap();
            assert!(lock.len() <= 255, "{} bytes", lock.len());
            assert!(lock.ends_with(".lock"));
        }
    }
}
