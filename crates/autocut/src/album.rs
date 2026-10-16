//! A Google Photos album as the seed of a showcase: the user adds the clips to an album in
//! Google Photos, and a showcase job given `album` (`autocut.analyze` / `autocut.makeShowcase`,
//! `filmcraft-cli showcase --album`) picks all of it, imports it — reusing what the project
//! already holds — and analyses exactly those clips.
//!
//! Why the picker: Google lets an app read back only the albums and items it created itself
//! (the Library API's read scopes for everything else were retired on 2025-03-31), so clips a
//! user adds to an album by hand are reachable only through Google's picker, where the user — or
//! FilmCraft's screen agent, opening the album and selecting all of it — chooses them.
//!
//! The picker and the screen agent live in crates this one does not depend on, so the host is
//! given an [`AlbumSource`] by whoever installs it (`filmcraft_automation::photos` for the CLI,
//! the MCP server and the app); tests use [`FixedAlbum`].

use std::sync::{Mutex, PoisonError};

use serde::Serialize;
use serde_json::{Value, json};

use crate::{AutoCutError, Executor};

/// What a showcase job asks an [`AlbumSource`] for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AlbumPick {
    /// The album's exact title in Google Photos.
    pub title: String,
    /// Download only its videos (a showcase is built from video clips).
    pub videos_only: bool,
    /// Skip items created before this date (`YYYY-MM-DD`): they are not downloaded.
    pub since: Option<String>,
}

/// An album, picked and on disk.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct AlbumFetched {
    /// The downloaded (or already cached) files, in the album's order.
    pub paths: Vec<String>,
    /// How many of them were downloaded now; the rest were in the cache.
    pub downloaded: usize,
    pub cached: usize,
    /// `"<file>: <why>"` for items that could not be downloaded.
    pub failed: Vec<String>,
    /// `"<file>: <why>"` for items left out (photos, items before `since`).
    pub skipped: Vec<String>,
    /// What the agent that drove the picker said, if one did.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
}

/// Picks a whole Google Photos album and downloads it.
pub trait AlbumSource: Send + Sync {
    /// Whether a pick can start now; `Err` says why not (not connected, …). Cheap: no network.
    fn ready(&self) -> Result<(), String> {
        Ok(())
    }

    /// Pick every item of album `req.title` and bring it to disk. Blocking (HTTP, the screen
    /// agent, the download); `progress` gets a line per step; stops with an error soon after
    /// `cancelled()` says so.
    fn pick(&self, req: &AlbumPick, progress: &mut dyn FnMut(&str), cancelled: &dyn Fn() -> bool) -> Result<AlbumFetched, String>;
}

/// An album's files in the project.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Imported {
    /// The media items, in the files' order: imported now or already there.
    pub items: Vec<u64>,
    /// Those the project already held (not imported again).
    pub reused: Vec<u64>,
    /// What could not be imported, and why.
    pub errors: Vec<String>,
}

/// Import `paths` (`file.import` with `reuse`: a file the project already holds is not imported
/// twice, its item is taken).
pub async fn import(ex: &dyn Executor, paths: &[String]) -> Result<Imported, AutoCutError> {
    if paths.is_empty() {
        return Err(AutoCutError::Invalid("nothing to import".into()));
    }
    let r = ex.execute("file.import", json!({"paths": paths, "reuse": true})).await.map_err(AutoCutError::Executor)?;
    let ids = |k: &str| r[k].as_array().map(|a| a.iter().filter_map(Value::as_u64).collect::<Vec<_>>()).unwrap_or_default();
    let errors = r["errors"].as_array().map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect()).unwrap_or_default();
    Ok(Imported { items: ids("items"), reused: ids("reused"), errors })
}

/// `YYYY-MM-DD` of a Unix time (UTC): the date a pick's `since` takes.
pub fn day_of(unix: i64) -> String {
    let (y, m, d) = crate::showcase::ymd(unix);
    format!("{y:04}-{m:02}-{d:02}")
}

/// An [`AlbumSource`] that "picks" the same files every time, and records what it was asked:
/// for tests and demos without Google Photos.
pub struct FixedAlbum {
    pub fetched: AlbumFetched,
    /// Every pick asked for.
    pub asked: Mutex<Vec<AlbumPick>>,
}

impl FixedAlbum {
    pub fn new(paths: Vec<String>) -> Self {
        let n = paths.len();
        Self { fetched: AlbumFetched { paths, downloaded: n, ..AlbumFetched::default() }, asked: Mutex::new(Vec::new()) }
    }

    /// The picks asked for so far.
    pub fn asked(&self) -> Vec<AlbumPick> {
        self.asked.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }
}

impl AlbumSource for FixedAlbum {
    fn pick(&self, req: &AlbumPick, progress: &mut dyn FnMut(&str), cancelled: &dyn Fn() -> bool) -> Result<AlbumFetched, String> {
        self.asked.lock().unwrap_or_else(PoisonError::into_inner).push(req.clone());
        if cancelled() {
            return Err("cancelled".into());
        }
        progress(&format!("{} item(s) picked from \"{}\"", self.fetched.paths.len(), req.title));
        Ok(self.fetched.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn days_for_the_picker() {
        assert_eq!(day_of(1_590_969_600), "2020-06-01");
        assert_eq!(day_of(0), "1970-01-01");
    }

    #[test]
    fn a_fixed_album_records_and_honours_cancel() {
        let a = FixedAlbum::new(vec!["/x/a.mp4".into()]);
        let req = AlbumPick { title: "Robots".into(), videos_only: true, since: None };
        let mut lines = Vec::new();
        let got = a.pick(&req, &mut |l| lines.push(l.to_string()), &|| false).unwrap();
        assert_eq!((got.paths.len(), got.downloaded), (1, 1));
        assert_eq!(lines, ["1 item(s) picked from \"Robots\""]);
        assert!(a.pick(&req, &mut |_| {}, &|| true).is_err());
        assert_eq!(a.asked().len(), 2);
    }
}
