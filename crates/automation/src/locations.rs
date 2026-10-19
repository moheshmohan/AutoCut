//! Where an album's items were taken, read from the Google Photos page — `filmcraft-cli photos
//! locations`, the engine's `photos.locations` ([`locator`]) and so MCP — in one place: the
//! `chrome` lock held while the screen agent drives the browser, the album's address (given,
//! remembered from an earlier run, else found on the albums page by its exact title), a
//! resumable JSON file written after every item, and the report.
//!
//! The file is `[{fileName, date, gmtOffset, place, estimated, …}]`
//! ([`filmcraft_screenagent::locations::Location`]); `photos.applyLocations` stores its places
//! with the project's items and a showcase fills its entries' `place` from them.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use serde_json::{Value, json};

use filmcraft_engine::photo_locations::{LocateRequest, PhotoLocator};
use filmcraft_screenagent::locations::{self as loc, Location};

/// How long a read waits for the browser to be free.
const BROWSER_WAIT: Duration = Duration::from_secs(15 * 60);
/// The largest locations file resumed from.
const MAX_FILE: u64 = 32 << 20;

/// What to read.
#[derive(Clone, Debug)]
pub struct Run {
    /// The album's exact title.
    pub album: Option<String>,
    /// Or its page.
    pub album_url: Option<String>,
    pub max_items: usize,
    pub per_item: Duration,
    pub budget: Duration,
    /// Where the JSON goes ([`default_out`] when `None`).
    pub out: Option<PathBuf>,
    /// Go on from the items already in `out`.
    pub resume: bool,
}

impl Default for Run {
    fn default() -> Self {
        Self {
            album: None,
            album_url: None,
            max_items: 1000,
            per_item: Duration::from_secs(90),
            budget: Duration::from_secs(3 * 3600),
            out: None,
            resume: false,
        }
    }
}

/// Where locations are kept: `<data>/google-photos/locations`.
pub fn dir() -> PathBuf {
    filmcraft_gphotos::data_dir().join("locations")
}

/// A file-name-safe form of an album's title or address.
fn slug(s: &str) -> String {
    let t: String = s.chars().map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { '-' }).collect();
    let t = t.split('-').filter(|w| !w.is_empty()).collect::<Vec<_>>().join("-");
    let t: String = t.chars().take(80).collect();
    if t.is_empty() { "album".into() } else { t }
}

/// The file a read of `album` (a title or an address) writes when none is given.
pub fn default_out(album: &str) -> PathBuf {
    let key = loc::album_url(album).and_then(|u| u.rsplit('/').next().map(str::to_string)).unwrap_or_else(|| album.to_string());
    dir().join(format!("{}.json", slug(&key)))
}

/// The album addresses found before, by title (`<dir>/albums.json`).
fn remembered_path() -> PathBuf {
    dir().join("albums.json")
}

fn remembered(title: &str) -> Option<String> {
    let bytes = std::fs::read(remembered_path()).ok().filter(|b| b.len() < 1 << 20)?;
    let v: Value = serde_json::from_slice(&bytes).ok()?;
    v.get(title.trim()).and_then(Value::as_str).and_then(loc::album_url)
}

fn remember(title: &str, url: &str) {
    let path = remembered_path();
    let mut v: serde_json::Map<String, Value> =
        std::fs::read(&path).ok().filter(|b| b.len() < 1 << 20).and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    if v.len() > 1000 {
        v.clear();
    }
    v.insert(title.trim().to_string(), json!(url));
    let _ = write_atomic(&path, &serde_json::to_vec_pretty(&Value::Object(v)).unwrap_or_default());
}

/// Write `bytes` to `path` through a temporary file, so a reader never sees half a file.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent).map_err(|e| format!("{}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension("json.part");
    std::fs::write(&tmp, bytes).map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {e}", path.display()))
}

/// The locations already in `path` (a resumed run), or none.
pub fn load(path: &Path) -> Result<Vec<Location>, String> {
    let Ok(meta) = std::fs::metadata(path) else { return Ok(Vec::new()) };
    if meta.len() > MAX_FILE {
        return Err(format!("{}: too large for a locations file", path.display()));
    }
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_slice(&bytes).map_err(|e| format!("{}: not JSON ({e})", path.display()))?;
    let list = match &v {
        Value::Object(o) => o.get("locations").cloned().unwrap_or(Value::Null),
        other => other.clone(),
    };
    serde_json::from_value(list).map_err(|e| format!("{}: not a locations file ({e})", path.display()))
}

/// The model's eyes, keeping every capture it is shown and every answer it gives in `dump`
/// when one is set (`AUTOCUT_LOCATIONS_DEBUG=<dir>`: for finding out why a panel did not read).
struct Eyes {
    model: filmcraft_screenagent::ground::QwenGrounder,
    dump: Option<PathBuf>,
    seen: std::cell::Cell<usize>,
}

impl Eyes {
    fn keep(&self, png: &[u8], what: &str, answer: &str) {
        let Some(dir) = &self.dump else { return };
        let n = self.seen.get() + 1;
        self.seen.set(n);
        let _ = std::fs::create_dir_all(dir);
        let _ = std::fs::write(dir.join(format!("{n:04}-{what}.png")), png);
        let _ = std::fs::write(dir.join(format!("{n:04}-{what}.txt")), answer);
    }
}

impl loc::PanelEyes for Eyes {
    fn read(&self, png: &[u8], ask: &str) -> Result<String, filmcraft_screenagent::ScreenError> {
        let r = loc::PanelEyes::read(&self.model, png, ask);
        self.keep(png, "read", &format!("{ask}\n---\n{}", r.as_ref().map_or_else(|e| format!("error: {e}"), String::clone)));
        r
    }

    fn find(&self, png: &[u8], target: &str) -> Result<Option<filmcraft_screenagent::ground::Element>, filmcraft_screenagent::ScreenError> {
        let r = loc::PanelEyes::find(&self.model, png, target);
        self.keep(png, "find", &format!("{target}\n---\n{r:?}"));
        r
    }
}

/// Whether this build can drive the browser for a read (macOS: Chrome through AppleScript).
pub fn readiness() -> Result<(), String> {
    if cfg!(target_os = "macos") { Ok(()) } else { Err("reading places from Google Photos drives Chrome on macOS; this platform cannot".into()) }
}

/// Read the places of `run`'s album (blocking: the browser, the model). `progress` hears a
/// line per step; `go_on` is asked after each item (false: stop). The JSON file is written
/// after every item. The report: `{album, albumUrl, out, locations, read, withoutPlace,
/// failed, stopped}`.
pub fn read(run: &Run, progress: &mut dyn FnMut(&str), go_on: &dyn Fn() -> bool) -> Result<Value, String> {
    readiness()?;
    let what = run.album.clone().or_else(|| run.album_url.clone()).ok_or("give the album's exact title (--album) or its address (--album-url)")?;
    if let Some(u) = &run.album_url
        && loc::album_url(u).is_none()
    {
        return Err(format!("not a Google Photos album address: {}", u.chars().take(120).collect::<String>()));
    }
    let out = run.out.clone().unwrap_or_else(|| default_out(&what));
    let done = if run.resume { load(&out)? } else { Vec::new() };
    if run.resume && !done.is_empty() {
        progress(&format!("locations: resuming from {} item(s) in {}", done.len(), out.display()));
    }
    // the model first: a run that cannot read is told before it drives the browser
    let cfg = filmcraft_screenagent::Config::from_env();
    let eyes = filmcraft_screenagent::ground::QwenGrounder::from_config(&filmcraft_screenagent::Config {
        http_timeout: run.per_item.clamp(Duration::from_secs(20), Duration::from_secs(600)),
        ..cfg.clone()
    });
    match eyes.model_available() {
        Ok(true) => {}
        Ok(false) => return Err(format!("the vision model {} is not installed: `ollama pull {}`", cfg.model, cfg.model)),
        Err(e) => return Err(e.to_string()),
    }
    let dump = std::env::var_os("AUTOCUT_LOCATIONS_DEBUG").map(PathBuf::from).filter(|d| !d.as_os_str().is_empty());
    let eyes = Eyes { model: eyes, dump, seen: std::cell::Cell::new(0) };
    // one driver of the browser at a time
    let mut said = false;
    let _browser = filmcraft_locks::acquire(filmcraft_locks::Resource::Chrome, &format!("reading places of {what}"), BROWSER_WAIT, &mut |e| {
        if !said {
            progress(&format!("locations: waiting for {e}"));
            said = true;
        }
    })
    .map_err(|e| format!("{e}; try again when it is free (`filmcraft-cli locks` shows who holds what)"))?;
    let mut viewer = loc::ChromeViewer::new();
    let album_url = match (&run.album_url, &run.album) {
        (Some(u), _) => loc::album_url(u).unwrap_or_default(),
        (None, Some(title)) => match remembered(title) {
            Some(u) => {
                progress(&format!("locations: the album \"{title}\" is {u} (remembered)"));
                u
            }
            None => {
                let u = loc::find_album(&mut viewer, &eyes, title, progress).map_err(|e| e.to_string())?;
                remember(title, &u);
                u
            }
        },
        (None, None) => return Err("give the album's exact title or its address".into()),
    };
    let mut all: Vec<Location> = done.clone();
    let mut write_err: Option<String> = None;
    let req = loc::Request { album_url: album_url.clone(), max_items: run.max_items, per_item: run.per_item, budget: run.budget, done };
    let report = loc::read_album(&mut viewer, &eyes, &req, progress, &mut |l| {
        match all.iter_mut().find(|x| x.error.is_some() && x.id.is_some() && x.id == l.id) {
            Some(slot) => *slot = l.clone(),
            None => all.push(l.clone()),
        }
        if let Err(e) = write_atomic(&out, &serde_json::to_vec_pretty(&all).unwrap_or_default()) {
            write_err = Some(e);
            return false;
        }
        go_on()
    })
    .map_err(|e| e.to_string())?;
    if let Some(e) = write_err {
        return Err(format!("could not keep the locations: {e}"));
    }
    write_atomic(&out, &serde_json::to_vec_pretty(&report.locations).unwrap_or_default())?;
    Ok(json!({
        "album": run.album,
        "albumUrl": album_url,
        "out": out.to_string_lossy(),
        "locations": report.locations,
        "read": report.read,
        "withoutPlace": report.without_place,
        "failed": report.failed,
        "stopped": report.stopped,
    }))
}

/// The engine's `photos.locations` through [`read`].
pub struct ChromeLocator;

impl PhotoLocator for ChromeLocator {
    fn ready(&self) -> Result<(), String> {
        readiness()
    }

    fn read(&self, request: &LocateRequest, progress: &filmcraft_engine::export::Progress) -> Result<Value, String> {
        let run = Run {
            album: request.album.clone(),
            album_url: request.album_url.clone(),
            max_items: request.max_items.map_or(1000, |n| usize::try_from(n).unwrap_or(usize::MAX)),
            per_item: request.per_item_seconds.map_or(Duration::from_secs(90), Duration::from_secs),
            out: request.out.as_ref().map(PathBuf::from),
            resume: request.resume,
            ..Run::default()
        };
        if let Some(n) = request.max_items {
            progress.total.store(n, Ordering::Relaxed);
        }
        let mut status = |line: &str| {
            *progress.status.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = line.trim_start_matches("locations: ").to_string();
        };
        let go_on = || {
            progress.done.fetch_add(1, Ordering::Relaxed);
            !progress.cancel.load(Ordering::Relaxed)
        };
        read(&run, &mut status, &go_on)
    }
}

/// The locator hosts install in `Session::photo_locations`.
pub fn locator() -> Arc<dyn PhotoLocator> {
    Arc::new(ChromeLocator)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_files_are_named_after_the_album() {
        assert!(default_out("Every Robot I've Met").ends_with("locations/every-robot-i-ve-met.json"));
        assert!(default_out("https://photos.google.com/album/AF1QipXYZ").ends_with("locations/af1qipxyz.json"));
        assert!(default_out("///").ends_with("locations/album.json"));
        assert!(default_out(&"é".repeat(500)).file_name().is_some_and(|n| n.len() < 400));
    }

    #[test]
    fn a_locations_file_loads_and_hostile_ones_are_errors() {
        let d = std::env::temp_dir().join(format!("filmcraft-automation-locations-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("l.json");
        assert!(load(&p).unwrap().is_empty(), "no file: nothing done yet");
        std::fs::write(&p, r#"[{"fileName": "a.mp4", "date": "2021-10-21", "gmtOffset": "+04:00", "place": "Dubai", "estimated": false, "id": "P1"}]"#)
            .unwrap();
        let l = load(&p).unwrap();
        assert_eq!((l[0].file_name.as_str(), l[0].place.as_deref(), l[0].id.as_deref()), ("a.mp4", Some("Dubai"), Some("P1")));
        std::fs::write(&p, r#"{"locations": [{"fileName": "b.mp4", "place": null}]}"#).unwrap();
        assert_eq!(load(&p).unwrap()[0].file_name, "b.mp4");
        for bad in ["{oops", "[5]", "\"x\"", r#"[{"place": "no name"}]"#] {
            std::fs::write(&p, bad).unwrap();
            assert!(load(&p).is_err(), "{bad}");
        }
        write_atomic(&p, b"[]").unwrap();
        assert!(load(&p).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_read_without_an_album_or_with_a_foreign_address_is_refused() {
        let mut lines = Vec::new();
        let e = read(&Run::default(), &mut |l| lines.push(l.to_string()), &|| true).unwrap_err();
        assert!(e.contains("--album") || e.contains("macOS"), "{e}");
        if cfg!(target_os = "macos") {
            let e = read(&Run { album_url: Some("https://example.com/album/A".into()), ..Run::default() }, &mut |_| {}, &|| true).unwrap_err();
            assert!(e.contains("not a Google Photos album"), "{e}");
        }
    }
}
