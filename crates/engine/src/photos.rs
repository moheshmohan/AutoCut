//! Upload to Google Photos: `photos.upload {"path", "description"?}` sends a file (typically a
//! finished export) to the user's Google Photos library as a background job, with byte progress
//! and cancellation like an export (`jobs.list`, `jobs.cancel`).
//!
//! The engine only runs the job. The upload itself (OAuth, Google's resumable protocol) lives in
//! `filmcraft-gphotos`, a layer above, which the host (desktop app, CLI, MCP server) installs as
//! [`Session::photos`]. Without one (the web build, tests) the command is disabled and says so.
//!
//! A finished upload's media item (`mediaItemId`, `productUrl`, …) is in its `jobs.list` entry
//! as `upload`, and in the command's answer with `wait: true`.

use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use crate::commands::{CommandSpec, bad, bool_p};
use crate::{EngineError, Result, Session};

/// Google's limit on a media item's description, in characters.
pub const MAX_DESCRIPTION_CHARS: usize = 1000;

/// What `photos.upload` says without an uploader.
const NOT_HERE: &str = "Google Photos uploads are not available in this app";

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
}

/// An upload job's media item, filled when it finishes.
pub type UploadOutcome = Arc<Mutex<Option<Value>>>;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "photos.upload",
        label: "Upload to Google Photos",
        menu: &[],
        shortcut: None,
        params: r#"{"path":str,"description":str?,"wait":bool=false}"#,
        enabled: can_upload,
        run: upload,
        journal: false,
    }]
}

/// `photos.upload` can run: an uploader is installed and ready.
pub fn can_upload(s: &Session) -> std::result::Result<(), String> {
    s.photos.as_ref().ok_or_else(|| NOT_HERE.to_string())?.ready()
}

fn upload(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "photos.upload";
    let up = s.photos.clone().ok_or_else(|| bad(cmd, NOT_HERE))?;
    let path = match p.get("path") {
        Some(Value::String(x)) if !x.trim().is_empty() => x.trim().to_string(),
        Some(Value::String(_)) | None | Some(Value::Null) => return Err(bad(cmd, "need `path`: the file to upload")),
        Some(_) => return Err(bad(cmd, "`path` must be a string")),
    };
    let description = match p.get("description") {
        None | Some(Value::Null) => None,
        Some(Value::String(d)) => {
            let d = d.trim();
            let n = d.chars().count();
            if n > MAX_DESCRIPTION_CHARS {
                return Err(bad(cmd, format!("`description` is {n} characters; Google Photos takes at most {MAX_DESCRIPTION_CHARS}")));
            }
            (!d.is_empty()).then(|| d.to_string())
        }
        Some(_) => return Err(bad(cmd, "`description` must be a string")),
    };
    let bytes = s.services.file_size(&path).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    if bytes == 0 {
        return Err(bad(cmd, format!("{path} is empty")));
    }
    up.accepts(&path).map_err(|e| bad(cmd, e))?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let name = std::path::Path::new(&path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.clone());
    let job = crate::narration::new_job(s, format!("Uploading {name} to Google Photos"));
    job.progress.total.store(bytes, std::sync::atomic::Ordering::Relaxed);
    *job.progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Waiting to upload".into();
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome = UploadOutcome::default();
    s.photo_uploads.insert(job.id, outcome.clone());
    let file = path.clone();
    let id = crate::narration::spawn(s, job, wait, "filmcraft-photos-upload", move || {
        let t0 = web_time::Instant::now();
        let r = up.upload(&file, description.as_deref(), &prog);
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
            Ok(filmcraft_export::Report { path: path.to_string(), frames: 0, seconds, bytes, render_fps: 0.0, extra_files: Vec::new() })
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
