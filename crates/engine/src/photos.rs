//! Upload to Google Photos: `photos.upload {"path", "description"?, "album"?}` sends a file
//! (typically a finished export) to the user's Google Photos library — and, with `album`, into
//! that album — as a background job, with byte progress and cancellation like an export
//! (`jobs.list`, `jobs.cancel`).
//!
//! Albums FilmCraft makes: `photos.createAlbum {"title", "description"?}` (an album FilmCraft
//! made earlier with that exact title is reused), `photos.albums` (the albums FilmCraft made;
//! Google shows an app no others) and `photos.addToAlbum {"album", "mediaItemIds"}` (items
//! FilmCraft uploaded). These are short network calls that answer directly, not jobs.
//!
//! The engine only runs them. The calls themselves (OAuth, Google's protocols) live in
//! `filmcraft-gphotos`, a layer above, which the host (desktop app, CLI, MCP server) installs as
//! [`Session::photos`]. Without one (the web build, tests) the commands are disabled and say so.
//!
//! A finished upload's media item (`mediaItemId`, `productUrl`, `album`, …) is in its
//! `jobs.list` entry as `upload`, and in the command's answer with `wait: true`.

use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p};
use crate::{EngineError, Result, Session};

/// Google's limit on a media item's description, in characters.
pub const MAX_DESCRIPTION_CHARS: usize = 1000;
/// Google's limit on an album title, in characters.
pub const MAX_ALBUM_TITLE_CHARS: usize = 500;
/// Google's limit on the items in one album: the most ids one `photos.addToAlbum` takes.
pub const MAX_ALBUM_ITEMS: usize = 20_000;
/// The longest media item id taken (Google's are well under it).
const MAX_ID_CHARS: usize = 512;

/// What `photos.upload` says without an uploader.
const NOT_HERE: &str = "Google Photos uploads are not available in this app";
/// What the album commands say when the host's uploader does no albums.
const NO_ALBUMS: &str = "Google Photos albums are not available in this app";

/// Uploads to Google Photos, installed by the host in [`Session::photos`].
pub trait PhotosUploader: Send + Sync {
    /// Whether an upload can start now: Google Photos is connected and the upload permission
    /// granted. `Err` says why not and what fixes it. Called while drawing, so it must be cheap
    /// (no network).
    fn ready(&self) -> std::result::Result<(), String>;
    /// Whether Google Photos takes this file (its type); `Err` says why not. Checked before the
    /// job starts so a wrong file fails at once.
    fn accepts(&self, _path: &str) -> std::result::Result<(), String> {
        Ok(())
    }
    /// Upload `path` (blocking). Report bytes in `progress.done` / `progress.total` and a line
    /// in `progress.status`; stop with an error once `progress.cancel` is set. `Ok` is the new
    /// media item as JSON.
    fn upload(&self, path: &str, description: Option<&str>, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String>;
    /// Whether the album commands can run now: connected, with the album permissions. `Err`
    /// says why not and what fixes it. Cheap (no network): it decides whether they are enabled.
    fn albums_ready(&self) -> std::result::Result<(), String> {
        Err(NO_ALBUMS.into())
    }
    /// [`Self::upload`] into `album` as well: an album FilmCraft made with that exact title (or
    /// that id), created when there is none.
    fn upload_to_album(
        &self,
        _path: &str,
        _description: Option<&str>,
        _album: &str,
        _progress: &filmcraft_export::Progress,
    ) -> std::result::Result<Value, String> {
        Err(NO_ALBUMS.into())
    }
    /// The album FilmCraft made with title `title`, created (with `description` as a note at
    /// its top) when there is none. `Ok` is the album as JSON (`id`, `title`, `productUrl`,
    /// `created`, …). Blocking (one or two short requests).
    fn create_album(&self, _title: &str, _description: Option<&str>) -> std::result::Result<Value, String> {
        Err(NO_ALBUMS.into())
    }
    /// The albums FilmCraft made, as JSON (`{"albums": [...], "complete"}`). Blocking.
    fn albums(&self) -> std::result::Result<Value, String> {
        Err(NO_ALBUMS.into())
    }
    /// Add media items FilmCraft uploaded to the album FilmCraft made with title (or id)
    /// `album`. Blocking.
    fn add_to_album(&self, _album: &str, _media_item_ids: &[String]) -> std::result::Result<Value, String> {
        Err(NO_ALBUMS.into())
    }
}

/// An upload job's media item, filled when it finishes.
pub type UploadOutcome = Arc<Mutex<Option<Value>>>;

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "photos.upload",
            label: "Upload to Google Photos",
            menu: &[],
            shortcut: None,
            params: r#"{"path":str,"description":str?,"album":str?,"wait":bool=false}"#,
            enabled: can_upload,
            run: upload,
            journal: false,
        },
        CommandSpec {
            id: "photos.createAlbum",
            label: "Create Google Photos Album",
            menu: &[],
            shortcut: None,
            params: r#"{"title":str,"description":str?}"#,
            enabled: can_use_albums,
            run: create_album,
            journal: false,
        },
        CommandSpec {
            id: "photos.albums",
            label: "List Google Photos Albums",
            menu: &[],
            shortcut: None,
            params: "{}",
            enabled: can_use_albums,
            run: list_albums,
            journal: false,
        },
        CommandSpec {
            id: "photos.addToAlbum",
            label: "Add to Google Photos Album",
            menu: &[],
            shortcut: None,
            params: r#"{"album":str,"mediaItemIds":[str]}"#,
            enabled: can_use_albums,
            run: add_to_album,
            journal: false,
        },
    ]
}

/// `photos.upload` can run: an uploader is installed and ready.
pub fn can_upload(s: &Session) -> std::result::Result<(), String> {
    s.photos.as_ref().ok_or_else(|| NOT_HERE.to_string())?.ready()
}

/// The album commands can run: an uploader is installed and has the album permissions.
pub fn can_use_albums(s: &Session) -> std::result::Result<(), String> {
    s.photos.as_ref().ok_or_else(|| NO_ALBUMS.to_string())?.albums_ready()
}

/// An optional string parameter, trimmed (empty is absent), at most `max` characters.
fn opt_text(cmd: &str, p: &Value, key: &str, max: usize, what: &str) -> Result<Option<String>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(d)) => {
            let d = d.trim();
            let n = d.chars().count();
            if n > max {
                return Err(bad(cmd, format!("`{key}` is {n} characters; {what} takes at most {max}")));
            }
            Ok((!d.is_empty()).then(|| d.to_string()))
        }
        Some(_) => Err(bad(cmd, format!("`{key}` must be a string"))),
    }
}

/// An album title (or id) parameter: required, at most [`MAX_ALBUM_TITLE_CHARS`].
fn album_param(cmd: &str, p: &Value, key: &str) -> Result<String> {
    opt_text(cmd, p, key, MAX_ALBUM_TITLE_CHARS, "a Google Photos album title")?.ok_or_else(|| bad(cmd, format!("need `{key}`: the album's title (or id)")))
}

fn albums_of(s: &Session, cmd: &str) -> Result<Arc<dyn PhotosUploader>> {
    let up = s.photos.clone().ok_or_else(|| bad(cmd, NO_ALBUMS))?;
    up.albums_ready().map_err(|e| bad(cmd, e))?;
    Ok(up)
}

fn create_album(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "photos.createAlbum";
    let title = album_param(cmd, p, "title")?;
    let description = opt_text(cmd, p, "description", MAX_DESCRIPTION_CHARS, "an album description")?;
    let up = albums_of(s, cmd)?;
    up.create_album(&title, description.as_deref()).map_err(|e| EngineError::Other(format!("{cmd}: {e}")))
}

fn list_albums(s: &mut Session, _p: &Value) -> Result<Value> {
    let cmd = "photos.albums";
    let up = albums_of(s, cmd)?;
    up.albums().map_err(|e| EngineError::Other(format!("{cmd}: {e}")))
}

fn add_to_album(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "photos.addToAlbum";
    let album = album_param(cmd, p, "album")?;
    let ids = match p.get("mediaItemIds") {
        Some(Value::Array(a)) if a.is_empty() => return Err(bad(cmd, "`mediaItemIds` is empty: pass the ids `photos.upload` returned")),
        Some(Value::Array(a)) if a.len() > MAX_ALBUM_ITEMS => {
            return Err(bad(cmd, format!("{} media item ids; an album holds at most {MAX_ALBUM_ITEMS}", a.len())));
        }
        Some(Value::Array(a)) => a
            .iter()
            .map(|v| match v.as_str().map(str::trim) {
                Some(id) if !id.is_empty() && id.chars().count() <= MAX_ID_CHARS => Ok(id.to_string()),
                _ => Err(bad(cmd, "every `mediaItemIds` entry must be a media item id (a non-empty string)")),
            })
            .collect::<Result<Vec<String>>>()?,
        None | Some(Value::Null) => return Err(bad(cmd, "need `mediaItemIds`: the media items to add (ids from `photos.upload`)")),
        Some(_) => return Err(bad(cmd, "`mediaItemIds` must be a list of media item ids")),
    };
    let up = albums_of(s, cmd)?;
    up.add_to_album(&album, &ids).map_err(|e| EngineError::Other(format!("{cmd}: {e}")))
}

fn upload(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "photos.upload";
    let up = s.photos.clone().ok_or_else(|| bad(cmd, NOT_HERE))?;
    let path = match p.get("path") {
        Some(Value::String(x)) if !x.trim().is_empty() => x.trim().to_string(),
        Some(Value::String(_)) | None | Some(Value::Null) => return Err(bad(cmd, "need `path`: the file to upload")),
        Some(_) => return Err(bad(cmd, "`path` must be a string")),
    };
    let description = opt_text(cmd, p, "description", MAX_DESCRIPTION_CHARS, "Google Photos")?;
    let album = opt_text(cmd, p, "album", MAX_ALBUM_TITLE_CHARS, "a Google Photos album title")?;
    if album.is_some() {
        up.albums_ready().map_err(|e| bad(cmd, e))?;
    }
    let bytes = s.services.file_size(&path).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    if bytes == 0 {
        return Err(bad(cmd, format!("{path} is empty")));
    }
    up.accepts(&path).map_err(|e| bad(cmd, e))?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let name = std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.clone());
    let label = match album.as_deref() {
        Some(a) => format!("Uploading {name} to the Google Photos album {a}"),
        None => format!("Uploading {name} to Google Photos"),
    };
    let job = crate::narration::new_job(s, label);
    job.progress.total.store(bytes, std::sync::atomic::Ordering::Relaxed);
    *job.progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Waiting to upload".into();
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome = UploadOutcome::default();
    s.photo_uploads.insert(job.id, outcome.clone());
    let file = path.clone();
    let id = crate::narration::spawn(s, job, wait, "filmcraft-photos-upload", move || {
        let t0 = web_time::Instant::now();
        let r = match album.as_deref() {
            Some(a) => up.upload_to_album(&file, description.as_deref(), a, &prog),
            None => up.upload(&file, description.as_deref(), &prog),
        };
        finish(&prog, &res, &outcome, &file, bytes, r, t0);
    })?;
    let mut out = json!({"job": id, "path": path});
    if wait && let Some(j) = s.jobs.iter().find(|j| j.id == id) {
        let v = j.to_json();
        if let Some(e) = v["result"]["error"].as_str() {
            return Err(EngineError::Other(format!("upload failed: {e}")));
        }
        out["result"] = v["result"].clone();
        out["upload"] = upload_json(s, id);
    }
    Ok(out)
}

/// End an upload job: the media item first, then the result that marks it finished.
fn finish(
    prog: &filmcraft_export::Progress,
    res: &crate::narration::JobResult,
    outcome: &UploadOutcome,
    path: &str,
    bytes: u64,
    r: std::result::Result<Value, String>,
    t0: web_time::Instant,
) {
    use std::sync::atomic::Ordering;
    let r = match r {
        Ok(item) => {
            *outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(item);
            prog.done.store(bytes, Ordering::Relaxed);
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = "Uploaded to Google Photos".into();
            let seconds = t0.elapsed().as_secs_f64();
            Ok(filmcraft_export::Report { path: path.to_string(), frames: 0, seconds, bytes, render_fps: 0.0, ..Default::default() })
        }
        Err(e) => {
            *prog.status.lock().unwrap_or_else(PoisonError::into_inner) = e.clone();
            *prog.error.lock().unwrap_or_else(PoisonError::into_inner) = Some(e.clone());
            Err(e)
        }
    };
    prog.finished.store(true, Ordering::Relaxed);
    *res.lock().unwrap_or_else(PoisonError::into_inner) = Some(r);
}

/// Upload job `job`'s media item (`null` while it runs, after a failure, or for another job).
pub fn upload_json(s: &Session, job: u64) -> Value {
    s.photo_uploads.get(&job).and_then(|o| o.lock().unwrap_or_else(PoisonError::into_inner).clone()).unwrap_or(Value::Null)
}

/// Whether job `job` is an upload.
pub fn is_upload(s: &Session, job: u64) -> bool {
    s.photo_uploads.contains_key(&job)
}

/// The file the most recent successful media export wrote (what "Upload to Google Photos"
/// offers after an export), if it is still there.
pub fn last_export(s: &Session) -> Option<String> {
    s.jobs.iter().rev().filter(|j| s.export_offline.contains_key(&j.id)).find_map(|j| {
        let r = j.result.lock().unwrap_or_else(PoisonError::into_inner);
        match &*r {
            Some(Ok(report)) if !report.path.is_empty() && s.services.file_size(&report.path).is_ok_and(|n| n > 0) => Some(report.path.clone()),
            _ => None,
        }
    })
}
