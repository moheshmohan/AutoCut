//! Insta360 footage for the editor.
//!
//! - [`library`]: the user's recordings on disk — where Insta360 Studio downloads its cloud files
//!   (`~/Downloads/Insta360 Download`) and every mounted camera card — with their `.lrv` proxies
//!   and whether each has finished downloading.
//! - [`studio`]: downloads from the user's Insta360 cloud through Insta360 Studio, which the user
//!   signed in to: the local screen agent reads Studio's Cloud Media list and clicks its Download
//!   menu, as a person would (the cloud has no public API).
//!
//! [`engine_host`] puts both behind the engine's `insta360.*` commands
//! (`filmcraft_engine::insta360`), which the desktop app, the CLI and the MCP server install.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

pub mod library;
pub mod studio;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::PoisonError;
use std::time::Duration;

use filmcraft_engine::insta360::{Downloaded, Insta360Host, Recording};

/// How long one cloud download may take to finish.
pub const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(60 * 60);
/// How long the agent waits for the mouse (another agent holding the `chrome` lock).
pub const LOCK_WAIT: Duration = Duration::from_secs(10 * 60);

/// The engine's view of a found recording.
fn engine_recording(r: &library::Recording) -> Recording {
    Recording {
        stem: r.stem.clone(),
        kind: match r.kind {
            library::Kind::Photo => "photo".into(),
            _ => "video".into(),
        },
        shot: r.shot.format("%Y-%m-%dT%H:%M:%S").to_string(),
        path: r.path.to_string_lossy().into_owned(),
        bytes: r.bytes,
        proxy: r.proxy.as_ref().map(|p| p.to_string_lossy().into_owned()),
        second_lens: r.second_lens.as_ref().map(|p| p.to_string_lossy().into_owned()),
        complete: r.complete,
        camera: r.camera.clone(),
    }
}

/// The desktop host: files through `std::fs`, Studio through the screen agent.
struct Host;

/// Studio's download folder (its default; a folder chosen in Studio is passed as `folders`).
pub fn download_folder() -> PathBuf {
    std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join("Downloads").join("Insta360 Download")
}

impl Insta360Host for Host {
    fn default_folders(&self) -> Vec<String> {
        library::default_folders().into_iter().map(|p| p.to_string_lossy().into_owned()).collect()
    }

    fn scan(&self, folders: &[String]) -> Vec<Recording> {
        let dirs: Vec<PathBuf> = folders.iter().map(PathBuf::from).collect();
        library::scan(&dirs).iter().map(engine_recording).collect()
    }

    fn studio_ready(&self) -> Result<(), String> {
        studio::ready()
    }

    fn download(&self, names: &[String], wait: bool, progress: &filmcraft_export::Progress) -> Result<Vec<Downloaded>, String> {
        use std::sync::atomic::Ordering;
        let set = |s: &str| *progress.status.lock().unwrap_or_else(PoisonError::into_inner) = s.to_string();
        let cancelled = || progress.cancel.load(Ordering::Relaxed);
        let mut said = false;
        let _mouse =
            filmcraft_locks::acquire(filmcraft_locks::Resource::Chrome, "Insta360 Studio download agent (the mouse and keyboard)", LOCK_WAIT, &mut |e| {
                if !said {
                    set(&format!("waiting for {e}"));
                    said = true;
                }
            })
            .map_err(|e| format!("{e}; try again when it is free (`filmcraft-cli locks` shows who holds what)"))?;
        let mut on = |s: &str| set(s);
        let outcomes = studio::download(names, wait, &download_folder(), DOWNLOAD_TIMEOUT, &mut on, &cancelled).map_err(|e| e.to_string())?;
        Ok(outcomes
            .into_iter()
            .map(|o| Downloaded { name: o.name, requested: o.requested, path: o.path.map(|p| p.to_string_lossy().into_owned()), why: o.why })
            .collect())
    }
}

/// The host the app installs as `Session::insta360`.
pub fn engine_host() -> Arc<dyn Insta360Host> {
    Arc::new(Host)
}
