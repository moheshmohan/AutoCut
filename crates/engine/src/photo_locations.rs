//! Where clips were filmed, from Google Photos.
//!
//! Google strips the location from what its API and picker hand out, so a downloaded clip has
//! none; the Google Photos page still shows it in each item's Info panel. Two commands:
//!
//! | command | does |
//! |---|---|
//! | `photos.locations` | read an album's places from the Google Photos page — a background job the host runs ([`PhotoLocator`]: the screen agent driving Chrome with the local vision model, under the `chrome` lock) — into a JSON file `[{fileName, date, gmtOffset, place, estimated}]`; with `wait` the places are then stored with the project's items |
//! | `photos.applyLocations` | store places (`locations`, or the file `path` a read wrote) with the project's media items, matched by file name: the item's `Location` field (and `Location Estimated: yes` when Google marked it estimated), one undo step |
//!
//! A showcase plan reads an entry's `place` from its first shot whose item has a `Location`
//! (`filmcraft_autocut::places`). Matching ignores case and a photo or video extension
//! (Google Photos sometimes shows a name without one): `VID_20221011_132939` is
//! `VID_20221011_132939.mp4`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use serde_json::{Value, json};

use filmcraft_project::{ItemId, ItemKind, MediaRef};

use crate::commands::{CommandSpec, always, bad, bool_p};
use crate::{EngineError, Result, Session};

/// The item field a place is stored in.
pub const LOCATION_FIELD: &str = "Location";
/// Set to `yes` when Google marked the place estimated.
pub const ESTIMATED_FIELD: &str = "Location Estimated";
/// Most locations one call takes, and the longest file name and place.
pub const MAX_LOCATIONS: usize = 20_000;
const MAX_NAME: usize = 255;
const MAX_PLACE: usize = 160;
/// The largest locations file read.
const MAX_FILE_BYTES: u64 = 32 << 20;

/// What `photos.locations` says without a locator.
const NOT_HERE: &str = "reading places from Google Photos is not available in this app (it drives Chrome on macOS)";

/// Reads an album's places from the Google Photos page, installed by the host in
/// [`Session::photo_locations`].
pub trait PhotoLocator: Send + Sync {
    /// Whether a read can start now; `Err` says why not and what fixes it. Cheap (no network,
    /// no browser): it decides whether the command is enabled.
    fn ready(&self) -> std::result::Result<(), String>;
    /// Read the places (blocking): a line in `progress.status` per item, items in
    /// `progress.done` / `progress.total`; stop once `progress.cancel` is set. `Ok` is the
    /// report: `{albumUrl, out, locations, read, withoutPlace, failed, stopped}`.
    fn read(&self, request: &LocateRequest, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String>;
}

/// What to read.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LocateRequest {
    /// The album's exact title (found on the albums page, then remembered).
    pub album: Option<String>,
    /// Or its page (`https://photos.google.com/album/…`).
    pub album_url: Option<String>,
    pub max_items: Option<u64>,
    pub per_item_seconds: Option<u64>,
    /// Go on from the items already in `out`.
    pub resume: bool,
    /// Where the JSON goes (the locator's default place when `None`).
    pub out: Option<String>,
}

/// The host's locator and what its jobs answered.
#[derive(Default)]
pub struct LocationsState {
    pub locator: Option<Arc<dyn PhotoLocator>>,
    pub outcomes: HashMap<u64, Arc<Mutex<Option<Value>>>>,
}

pub fn commands() -> Vec<CommandSpec> {
    vec![
        CommandSpec {
            id: "photos.locations",
            label: "Read Places from Google Photos",
            menu: &[],
            shortcut: None,
            params: r#"{"album":str? (exact title),"albumUrl":"https://photos.google.com/album/…"?,"maxItems":1..5000?,"perItem":5..600 (seconds)?,"resume":bool?,"out":str? (JSON file),"apply":bool=true (with wait: store the places with the project's items),"wait":bool=false}"#,
            enabled: can_locate,
            run: locate,
            journal: false,
        },
        CommandSpec {
            id: "photos.applyLocations",
            label: "Apply Places to Project Items",
            menu: &[],
            shortcut: None,
            params: r#"{"locations":[{"fileName":str,"place":str|null,"estimated":bool?}]? | "path":str? (a photos.locations file),"items":[id]? (only these)}"#,
            enabled: always,
            run: apply,
            journal: true,
        },
    ]
}

/// `photos.locations` can run: a locator is installed and ready.
pub fn can_locate(s: &Session) -> std::result::Result<(), String> {
    s.photo_locations.locator.as_ref().ok_or_else(|| NOT_HERE.to_string())?.ready()
}

/// One place to store: a file name's key, the place, estimated.
#[derive(Clone, Debug, PartialEq)]
pub struct Place {
    pub file_name: String,
    pub place: String,
    pub estimated: bool,
}

/// File extensions Google Photos may leave off a name.
const MEDIA_EXT: [&str; 22] = [
    "mp4", "mov", "m4v", "3gp", "mkv", "webm", "avi", "mts", "m2ts", "wmv", "mpg", "jpg", "jpeg", "heic", "heif", "png", "gif", "webp", "dng", "arw", "cr2",
    "nef",
];

/// A file name as names are compared: its base name (no folders), without a photo or video
/// extension, lowercased.
pub fn name_key(name: &str) -> String {
    let base = name.rsplit(['/', '\\']).next().unwrap_or(name).trim();
    let stem = match base.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() && MEDIA_EXT.contains(&ext.to_ascii_lowercase().as_str()) => stem,
        _ => base,
    };
    stem.to_lowercase()
}

/// The places in `v`: an array of `{fileName, place, estimated}` (or `{"locations": [...]}`),
/// each with a file name and a place (those without a place are left out). Hostile values are
/// errors; overlong names and places are refused.
pub fn places_from(cmd: &str, v: &Value) -> Result<Vec<Place>> {
    let list = match v {
        Value::Array(a) => a,
        Value::Object(o) => {
            o.get("locations").and_then(Value::as_array).ok_or_else(|| bad(cmd, "`locations` must be a list of {fileName, place, estimated}"))?
        }
        _ => return Err(bad(cmd, "`locations` must be a list of {fileName, place, estimated}")),
    };
    if list.len() > MAX_LOCATIONS {
        return Err(bad(cmd, format!("{} locations; at most {MAX_LOCATIONS}", list.len())));
    }
    let mut out = Vec::new();
    for (k, l) in list.iter().enumerate() {
        let Some(o) = l.as_object() else { return Err(bad(cmd, format!("location {}: not an object", k + 1))) };
        let name = match o.get("fileName") {
            Some(Value::String(n)) => n.trim(),
            None | Some(Value::Null) => continue, // an item that could not be read
            Some(_) => return Err(bad(cmd, format!("location {}: `fileName` must be a string", k + 1))),
        };
        let place = match o.get("place") {
            Some(Value::String(p)) => p.trim(),
            None | Some(Value::Null) => continue,
            Some(_) => return Err(bad(cmd, format!("location {}: `place` must be a string or null", k + 1))),
        };
        if name.is_empty() || place.is_empty() {
            continue;
        }
        if name.chars().count() > MAX_NAME || place.chars().count() > MAX_PLACE || name.chars().chain(place.chars()).any(char::is_control) {
            return Err(bad(cmd, format!("location {}: the file name or place is too long or has control characters", k + 1)));
        }
        let estimated = match o.get("estimated") {
            Some(Value::Bool(b)) => *b,
            None | Some(Value::Null) => false,
            Some(_) => return Err(bad(cmd, format!("location {}: `estimated` must be true or false", k + 1))),
        };
        out.push(Place { file_name: name.to_string(), place: place.to_string(), estimated });
    }
    Ok(out)
}

/// The names an item is known by: its name and its media file's.
fn item_keys(s: &Session, id: ItemId) -> Vec<String> {
    let Some(it) = s.project.item(id) else { return Vec::new() };
    let mut keys = vec![name_key(&it.name)];
    if let ItemKind::Media(m) = &it.kind
        && let MediaRef::File { path } = &m.media
    {
        let k = name_key(path);
        if !keys.contains(&k) {
            keys.push(k);
        }
    }
    keys
}

/// Which project media items `places` name: (item, place) pairs, the first place of a name.
pub fn matches(s: &Session, places: &[Place], only: Option<&[ItemId]>) -> Vec<(ItemId, Place)> {
    let mut by_key: HashMap<String, &Place> = HashMap::new();
    for p in places {
        by_key.entry(name_key(&p.file_name)).or_insert(p);
    }
    let mut out = Vec::new();
    for it in s.project.items.values().filter(|it| matches!(it.kind, ItemKind::Media(_))) {
        if only.is_some_and(|o| !o.contains(&it.id)) {
            continue;
        }
        if let Some(p) = item_keys(s, it.id).iter().find_map(|k| by_key.get(k)) {
            out.push((it.id, (*p).clone()));
        }
    }
    out
}

/// Store `found` with the items, one undo step; how many changed.
fn store(s: &mut Session, found: &[(ItemId, Place)]) -> Result<usize> {
    let changed: Vec<(ItemId, Place)> = found
        .iter()
        .filter(|(id, p)| {
            s.project.item(*id).is_some_and(|it| {
                it.metadata.get(LOCATION_FIELD) != Some(&p.place) || it.metadata.get(ESTIMATED_FIELD).is_some_and(|e| e == "yes") != p.estimated
            })
        })
        .cloned()
        .collect();
    if changed.is_empty() {
        return Ok(0);
    }
    s.edit("Apply Places", |pr, _| {
        for (id, p) in &changed {
            let it = pr.item_mut(*id).ok_or_else(|| bad("photos.applyLocations", "no such item"))?;
            it.metadata.insert(LOCATION_FIELD.into(), p.place.clone());
            if p.estimated {
                it.metadata.insert(ESTIMATED_FIELD.into(), "yes".into());
            } else {
                it.metadata.remove(ESTIMATED_FIELD);
            }
        }
        Ok(())
    })?;
    Ok(changed.len())
}

/// Read a locations file (capped), as JSON.
fn read_file(cmd: &str, path: &str) -> Result<Value> {
    let meta = std::fs::metadata(path).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    if meta.len() > MAX_FILE_BYTES {
        return Err(bad(cmd, format!("{path}: larger than {} MB; not a locations file", MAX_FILE_BYTES >> 20)));
    }
    let bytes = std::fs::read(path).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    serde_json::from_slice(&bytes).map_err(|e| bad(cmd, format!("{path}: not JSON ({e})")))
}

/// Store places with items and say what matched.
fn apply_places(s: &mut Session, cmd: &str, v: &Value, only: Option<&[ItemId]>) -> Result<Value> {
    let places = places_from(cmd, v)?;
    let found = matches(s, &places, only);
    let changed = store(s, &found)?;
    let matched: std::collections::HashSet<String> = found.iter().map(|(_, p)| name_key(&p.file_name)).collect();
    let unmatched: Vec<&str> = places.iter().filter(|p| !matched.contains(&name_key(&p.file_name))).map(|p| p.file_name.as_str()).take(200).collect();
    let items: Vec<Value> = found
        .iter()
        .map(|(id, p)| json!({"item": id.0, "name": s.project.item(*id).map(|it| it.name.clone()), "place": p.place, "estimated": p.estimated}))
        .collect();
    Ok(json!({"places": places.len(), "matched": found.len(), "changed": changed, "items": items, "unmatched": unmatched}))
}

fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "photos.applyLocations";
    let only: Option<Vec<ItemId>> = match p.get("items") {
        None | Some(Value::Null) => None,
        Some(Value::Array(a)) => Some(a.iter().map(|v| v.as_u64().map(ItemId).ok_or_else(|| bad(cmd, "`items` must be item ids"))).collect::<Result<_>>()?),
        Some(_) => return Err(bad(cmd, "`items` must be a list of item ids")),
    };
    let v = match (p.get("locations"), p.get("path")) {
        (Some(l), None) => l.clone(),
        (None, Some(Value::String(path))) if !path.trim().is_empty() => read_file(cmd, path.trim())?,
        (None, Some(_)) => return Err(bad(cmd, "`path` must be the file `photos.locations` wrote")),
        (Some(_), Some(_)) => return Err(bad(cmd, "give `locations` or `path`, not both")),
        (None, None) => return Err(bad(cmd, "need `locations` or `path` (the file `photos.locations` wrote)")),
    };
    apply_places(s, cmd, &v, only.as_deref())
}

fn opt_str(cmd: &str, p: &Value, key: &str, max: usize) -> Result<Option<String>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(v)) => {
            let v = v.trim();
            if v.chars().count() > max || v.chars().any(char::is_control) {
                return Err(bad(cmd, format!("`{key}` is too long or has control characters")));
            }
            Ok((!v.is_empty()).then(|| v.to_string()))
        }
        Some(_) => Err(bad(cmd, format!("`{key}` must be a string"))),
    }
}

fn opt_count(cmd: &str, p: &Value, key: &str, lo: u64, hi: u64) -> Result<Option<u64>> {
    match p.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => match v.as_u64() {
            Some(n) if (lo..=hi).contains(&n) => Ok(Some(n)),
            _ => Err(bad(cmd, format!("`{key}` must be a whole number from {lo} to {hi}"))),
        },
    }
}

/// The request in `photos.locations`'s params, checked.
pub fn request_from(p: &Value) -> Result<LocateRequest> {
    let cmd = "photos.locations";
    let album = opt_str(cmd, p, "album", 500)?;
    let album_url = opt_str(cmd, p, "albumUrl", 2048)?;
    if let Some(u) = &album_url
        && !(u.starts_with("https://photos.google.com/album/")
            || u.starts_with("https://photos.google.com/u/")
            || u.starts_with("https://photos.google.com/share/"))
    {
        return Err(bad(cmd, "`albumUrl` must be a Google Photos album address (https://photos.google.com/album/…)"));
    }
    if album.is_none() && album_url.is_none() {
        return Err(bad(cmd, "need `album` (its exact title) or `albumUrl`"));
    }
    Ok(LocateRequest {
        album,
        album_url,
        max_items: opt_count(cmd, p, "maxItems", 1, 5000)?,
        per_item_seconds: opt_count(cmd, p, "perItem", 5, 600)?,
        resume: bool_p(p, "resume").unwrap_or(false),
        out: opt_str(cmd, p, "out", 4096)?,
    })
}

fn locate(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "photos.locations";
    let req = request_from(p)?;
    let locator = s.photo_locations.locator.clone().ok_or_else(|| bad(cmd, NOT_HERE))?;
    locator.ready().map_err(|e| bad(cmd, e))?;
    let wait = bool_p(p, "wait").unwrap_or(false);
    let apply_after = bool_p(p, "apply").unwrap_or(true);
    let what = req.album.clone().or_else(|| req.album_url.clone()).unwrap_or_default();
    let job = crate::narration::new_job(s, format!("Reading places of the Google Photos album {what}"));
    *job.progress.status.lock().unwrap_or_else(PoisonError::into_inner) = "Waiting for the browser".into();
    let (prog, res) = (job.progress.clone(), job.result.clone());
    let outcome: Arc<Mutex<Option<Value>>> = Arc::default();
    s.photo_locations.outcomes.insert(job.id, outcome.clone());
    let id = crate::narration::spawn(s, job, wait, "filmcraft-photo-locations", move || {
        let t0 = web_time::Instant::now();
        let r = locator.read(&req, &prog);
        let r = match r {
            Ok(v) => {
                let status = format!("Read {} item(s); {} without a place", v["read"].as_u64().unwrap_or(0), v["withoutPlace"].as_u64().unwrap_or(0));
                *outcome.lock().unwrap_or_else(PoisonError::into_inner) = Some(v);
                Ok(status)
            }
            Err(e) => Err(e),
        };
        crate::narration::finish(&prog, &res, r, t0);
    })?;
    let mut out = json!({"job": id});
    if wait {
        let v = outcome_json(s, id);
        if let Some(e) = s.jobs.iter().find(|j| j.id == id).and_then(|j| j.to_json()["result"]["error"].as_str().map(str::to_string)) {
            return Err(EngineError::Other(format!("{cmd}: {e}")));
        }
        if apply_after && v["locations"].is_array() {
            out["applied"] = apply_places(s, "photos.applyLocations", &v["locations"], None)?;
        }
        out["locations"] = v;
    }
    Ok(out)
}

/// A `photos.locations` job's report (`null` while it runs, after a failure, or for another job).
pub fn outcome_json(s: &Session, job: u64) -> Value {
    s.photo_locations.outcomes.get(&job).and_then(|o| o.lock().unwrap_or_else(PoisonError::into_inner).clone()).unwrap_or(Value::Null)
}

/// Whether job `job` reads places.
pub fn is_job(s: &Session, job: u64) -> bool {
    s.photo_locations.outcomes.contains_key(&job)
}
