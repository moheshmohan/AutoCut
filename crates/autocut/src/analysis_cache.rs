//! Analyse once: what a clip analysis found, kept by what the clip *is*, so going over a library
//! again re-analyses nothing it has seen.
//!
//! Locating a subject frame by frame with a local vision model is the slow part of a showcase (a
//! minute or two a clip). An entry is keyed by
//! - the media file's identity: its Google Photos media id when the file came from the picker's
//!   download cache (its sidecar's `googlePhotosId`, or the cache folder it sits in), else its
//!   size and a fingerprint of its first and last MiB (the engine's relink fingerprint, so a moved
//!   or renamed file is still the same clip);
//! - the subject, the vision model, and the frames/screen settings;
//! - [`VERSION`], bumped whenever the analysis itself changes.
//!
//! Entries live in `<data dir>/autocut/analysis-cache/<hash>.json`, written atomically (a
//! temporary file renamed into place), at most [`MAX_ENTRY_BYTES`] each and [`MAX_ENTRIES`] in all
//! (the oldest go first). A damaged or foreign entry is ignored — it is only a cache. An entry
//! keeps the clip's facts, not its place in a project: the item id, name and capture date come
//! from the project each time.

use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use serde_json::{Value, json};

use crate::analyze::ClipFacts;

/// Bumped whenever the analysis changes (its prompts, sampling or rules), so entries made by an
/// older analysis no longer match.
pub const VERSION: u32 = 1;
/// The largest entry read.
pub const MAX_ENTRY_BYTES: u64 = 4 << 20;
/// Entries kept at most; the oldest are removed past it.
pub const MAX_ENTRIES: usize = 20_000;
/// The largest metadata sidecar read.
const MAX_SIDECAR_BYTES: u64 = 1 << 20;
/// Bytes fingerprinted at each end of a file.
const SPAN: usize = filmcraft_engine::relink::FINGERPRINT_SPAN;

/// The analysis cache: where it is (none: off), whether to bypass it, and what it did.
#[derive(Debug, Default)]
pub struct AnalysisCache {
    dir: Option<PathBuf>,
    fresh: bool,
    hits: AtomicUsize,
    stored: AtomicUsize,
    pruned: AtomicBool,
}

impl AnalysisCache {
    /// No cache: every clip is analysed, nothing is kept.
    pub fn off() -> Self {
        Self::default()
    }

    /// A cache in the directory `dir` itself.
    pub fn at(dir: PathBuf) -> Self {
        Self { dir: Some(dir), ..Self::default() }
    }

    /// The cache of the data directory `data`: `<data>/autocut/analysis-cache`.
    pub fn in_data_dir(data: &Path) -> Self {
        Self::at(data.join("autocut").join("analysis-cache"))
    }

    /// The cache of the app's data directory (off when there is none).
    pub fn default_location() -> Self {
        filmcraft_engine::autosave::default_data_dir().map_or_else(Self::off, |d| Self::in_data_dir(&d))
    }

    /// With `fresh`, nothing is read from the cache: every clip is analysed again, and what it
    /// finds replaces what was kept.
    pub fn fresh(mut self, fresh: bool) -> Self {
        self.fresh = fresh;
        self
    }

    pub fn enabled(&self) -> bool {
        self.dir.is_some()
    }

    /// Clips taken from the cache so far.
    pub fn hits(&self) -> usize {
        self.hits.load(Ordering::Relaxed)
    }

    /// Clips written to the cache so far.
    pub fn stored(&self) -> usize {
        self.stored.load(Ordering::Relaxed)
    }

    /// The key of a clip's analysis: what the file is, and everything the analysis depends on.
    pub fn key(identity: &str, subject: &str, model: &str, frames: usize, screen: usize) -> String {
        let subject = subject.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
        format!("v{VERSION}|{identity}|subject={subject}|model={model}|frames={frames}|screen={screen}")
    }

    fn path_of(&self, key: &str) -> Option<PathBuf> {
        let hash = filmcraft_engine::relink::fingerprint(key.len() as u64, key.as_bytes(), &[]);
        Some(self.dir.as_ref()?.join(format!("{hash:016x}.json")))
    }

    /// The analysis kept under `key`, if there is a sound one (and the cache is not bypassed).
    pub fn get(&self, key: &str) -> Option<ClipFacts> {
        if self.fresh {
            return None;
        }
        let path = self.path_of(key)?;
        let file = std::fs::File::open(&path).ok()?;
        if file.metadata().ok()?.len() > MAX_ENTRY_BYTES {
            return None;
        }
        let mut text = String::new();
        file.take(MAX_ENTRY_BYTES).read_to_string(&mut text).ok()?;
        let v: Value = serde_json::from_str(&text).ok()?;
        // another key that hashed the same, an older analysis, or a damaged entry: not ours
        if v["version"].as_u64() != Some(u64::from(VERSION)) || v["key"].as_str() != Some(key) {
            return None;
        }
        let facts: ClipFacts = serde_json::from_value(v.get("facts")?.clone()).ok()?;
        self.hits.fetch_add(1, Ordering::Relaxed);
        Some(facts)
    }

    /// Keep `facts` under `key`: written to a temporary file and renamed into place, so a reader
    /// never sees half an entry. Best effort: a cache that cannot be written only costs time.
    pub fn put(&self, key: &str, facts: &ClipFacts) {
        let (Some(dir), Some(path)) = (self.dir.as_ref(), self.path_of(key)) else { return };
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }
        if !self.pruned.swap(true, Ordering::Relaxed) {
            prune(dir, MAX_ENTRIES);
        }
        let stored_at = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let entry = json!({"version": VERSION, "key": key, "storedAt": stored_at, "facts": facts});
        let Ok(bytes) = serde_json::to_vec(&entry) else { return };
        if bytes.len() as u64 > MAX_ENTRY_BYTES {
            return;
        }
        let nanos = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
        let tmp = dir.join(format!(".{}.{}.{nanos}.tmp", path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(), std::process::id()));
        if std::fs::write(&tmp, &bytes).is_ok() && std::fs::rename(&tmp, &path).is_ok() {
            self.stored.fetch_add(1, Ordering::Relaxed);
        } else {
            let _ = std::fs::remove_file(&tmp);
        }
    }
}

/// Keep at most `max` entries in `dir`: past it, the least recently written go (down to nine
/// tenths of `max`, so pruning is rare). The walk itself is bounded.
fn prune(dir: &Path, max: usize) {
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<(std::time::SystemTime, PathBuf)> = rd
        .flatten()
        .take(max.saturating_mul(2).saturating_add(1000))
        .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    if entries.len() <= max {
        return;
    }
    entries.sort_by_key(|(t, _)| *t);
    let keep = max.saturating_sub(max / 10);
    let excess = entries.len().saturating_sub(keep);
    for (_, p) in entries.into_iter().take(excess) {
        let _ = std::fs::remove_file(p);
    }
}

/// What a media file is, for the cache: `google:<media id>` for an original from the Google
/// Photos download cache, else `file:<size>:<fingerprint>` of its content; `None` when it cannot
/// be read.
pub fn media_identity(path: &Path) -> Option<String> {
    google_id(path).map(|id| format!("google:{id}")).or_else(|| content_identity(path))
}

/// A Google Photos media id: URL-safe characters only, bounded.
fn clean_id(id: &str) -> Option<String> {
    let id = id.trim();
    (!id.is_empty() && id.len() <= 256 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')).then(|| id.to_string())
}

/// The Google Photos media id of a picked original: its `<file>.json` sidecar's `googlePhotosId`,
/// else the folder it was downloaded into (`…/google-photos/cache/<media id>/<file>`).
fn google_id(path: &Path) -> Option<String> {
    let mut sidecar = path.as_os_str().to_os_string();
    sidecar.push(".json");
    let from_sidecar = std::fs::metadata(&sidecar)
        .ok()
        .filter(|m| m.is_file() && m.len() <= MAX_SIDECAR_BYTES)
        .and_then(|_| std::fs::read(&sidecar).ok())
        .and_then(|b| serde_json::from_slice::<Value>(&b).ok())
        .and_then(|v| v.get("googlePhotosId").and_then(Value::as_str).and_then(clean_id));
    from_sidecar.or_else(|| {
        let folder = path.parent()?;
        let cache = folder.parent()?;
        let root = cache.parent()?;
        (cache.file_name()? == "cache" && root.file_name()? == "google-photos").then(|| clean_id(&folder.file_name()?.to_string_lossy())).flatten()
    })
}

/// `file:<size>:<fingerprint>`: the size and the engine's fingerprint of the first and last MiB
/// (as `media.status` reports a file's identity).
pub fn content_identity(path: &Path) -> Option<String> {
    let mut f = std::fs::File::open(path).ok()?;
    let size = f.metadata().ok().filter(std::fs::Metadata::is_file)?.len();
    let span = SPAN as u64;
    let mut head = Vec::with_capacity(SPAN.min(usize::try_from(size).unwrap_or(SPAN)));
    (&mut f).take(span).read_to_end(&mut head).ok()?;
    let mut tail = Vec::new();
    if size > span.saturating_mul(2) {
        f.seek(SeekFrom::Start(size - span)).ok()?;
    }
    f.take(span).read_to_end(&mut tail).ok()?;
    Some(format!("file:{size}:{:016x}", filmcraft_engine::relink::fingerprint(size, &head, &tail)))
}

/// The identity `media.status` reports for an item (`{"size", "fingerprint"}`), in the form of
/// [`content_identity`].
pub fn reported_identity(v: &Value) -> Option<String> {
    let size = v.get("size")?.as_u64()?;
    let fp = u64::from_str_radix(v.get("fingerprint")?.as_str()?, 16).ok()?;
    Some(format!("file:{size}:{fp:016x}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("autocut-analysis-cache-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn facts(label: &str) -> ClipFacts {
        ClipFacts { item: 7, name: "c7.mp4".into(), seconds: 12.0, present: true, coverage: 0.5, label: label.into(), best: 3.0, ..ClipFacts::default() }
    }

    #[test]
    fn entries_round_trip_by_key_and_fresh_bypasses_them() {
        let dir = temp("roundtrip");
        let cache = AnalysisCache::in_data_dir(&dir);
        let key = AnalysisCache::key("file:10:00000000000000ab", "  Robot ", "ollama:qwen2.5vl:7b", 6, 0);
        assert_eq!(key, AnalysisCache::key("file:10:00000000000000ab", "robot", "ollama:qwen2.5vl:7b", 6, 0), "the subject in any case and spacing");
        assert!(cache.get(&key).is_none());
        cache.put(&key, &facts("Pepper humanoid robot"));
        assert_eq!(cache.stored(), 1);
        let got = cache.get(&key).unwrap();
        assert_eq!((got.label.as_str(), got.best, cache.hits()), ("Pepper humanoid robot", 3.0, 1));
        assert!(dir.join("autocut/analysis-cache").is_dir());
        // anything the analysis depends on makes another key
        for other in [
            AnalysisCache::key("file:11:00000000000000ab", "robot", "ollama:qwen2.5vl:7b", 6, 0),
            AnalysisCache::key("file:10:00000000000000ab", "drone", "ollama:qwen2.5vl:7b", 6, 0),
            AnalysisCache::key("file:10:00000000000000ab", "robot", "ollama:qwen2.5vl:3b", 6, 0),
            AnalysisCache::key("file:10:00000000000000ab", "robot", "ollama:qwen2.5vl:7b", 8, 0),
            AnalysisCache::key("file:10:00000000000000ab", "robot", "ollama:qwen2.5vl:7b", 6, 3),
        ] {
            assert!(cache.get(&other).is_none(), "{other}");
        }
        // --fresh: nothing read, and what it finds replaces the entry
        let fresh = AnalysisCache::in_data_dir(&dir).fresh(true);
        assert!(fresh.get(&key).is_none());
        fresh.put(&key, &facts("NAO robot"));
        assert_eq!(AnalysisCache::in_data_dir(&dir).get(&key).unwrap().label, "NAO robot");
        // off: nothing kept, nothing read
        let off = AnalysisCache::off();
        off.put(&key, &facts("x"));
        assert!(off.get(&key).is_none() && !off.enabled() && off.stored() == 0);
        // no temporary files are left behind
        let names: Vec<String> =
            std::fs::read_dir(dir.join("autocut/analysis-cache")).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        assert!(names.iter().all(|n| n.ends_with(".json") && !n.starts_with('.')), "{names:?}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn damaged_foreign_and_old_entries_are_ignored() {
        let dir = temp("damaged");
        let cache = AnalysisCache::at(dir.clone());
        let key = AnalysisCache::key("google:AF1Qip", "robot", "m", 6, 0);
        let path = cache.path_of(&key).unwrap();
        for junk in [
            "".to_string(),
            "{not json".to_string(),
            "[1,2]".to_string(),
            json!({"version": VERSION, "key": "another key", "facts": facts("x")}).to_string(),
            json!({"version": VERSION + 1, "key": key, "facts": facts("x")}).to_string(),
            json!({"version": VERSION, "key": key, "facts": {"item": "seven"}}).to_string(),
            json!({"version": VERSION, "key": key}).to_string(),
        ] {
            std::fs::write(&path, &junk).unwrap();
            assert!(cache.get(&key).is_none(), "{junk}");
        }
        // a huge file is not read
        std::fs::write(&path, vec![b' '; (MAX_ENTRY_BYTES + 1) as usize]).unwrap();
        assert!(cache.get(&key).is_none());
        // and is replaced by the next analysis
        cache.put(&key, &facts("robot arm"));
        assert_eq!(cache.get(&key).unwrap().label, "robot arm");
        assert_eq!(cache.hits(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_oldest_entries_go_past_the_cap() {
        let dir = temp("prune");
        for i in 0..30 {
            std::fs::write(dir.join(format!("{i:016x}.json")), "{}").unwrap();
        }
        std::fs::write(dir.join("notes.txt"), "kept").unwrap();
        prune(&dir, 20);
        let left = std::fs::read_dir(&dir).unwrap().flatten().filter(|e| e.path().extension().is_some_and(|x| x == "json")).count();
        assert_eq!(left, 18, "down to nine tenths of the cap");
        assert!(dir.join("notes.txt").is_file(), "only entries are pruned");
        prune(&dir, 20);
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 19, "under the cap nothing goes");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_file_is_known_by_its_google_id_or_its_content() {
        let dir = temp("identity");
        // a picked original: by its sidecar, or by the cache folder it is in
        let picked = dir.join("google-photos/cache/AF1QipABC-_1/PXL_1.mp4");
        std::fs::create_dir_all(picked.parent().unwrap()).unwrap();
        std::fs::write(&picked, b"video").unwrap();
        assert_eq!(media_identity(&picked).as_deref(), Some("google:AF1QipABC-_1"));
        std::fs::write(dir.join("google-photos/cache/AF1QipABC-_1/PXL_1.mp4.json"), r#"{"googlePhotosId": "AF1QipFromSidecar"}"#).unwrap();
        assert_eq!(media_identity(&picked).as_deref(), Some("google:AF1QipFromSidecar"));
        // a hostile sidecar id is not used; the folder still is
        std::fs::write(dir.join("google-photos/cache/AF1QipABC-_1/PXL_1.mp4.json"), r#"{"googlePhotosId": "../../etc"}"#).unwrap();
        assert_eq!(media_identity(&picked).as_deref(), Some("google:AF1QipABC-_1"));
        // any other file: its size and content, the same wherever it is and whatever it is called
        let a = dir.join("a.mov");
        let b = dir.join("moved/renamed.mov");
        std::fs::create_dir_all(b.parent().unwrap()).unwrap();
        let bytes: Vec<u8> = (0..3_000_000u32).map(|i| (i % 251) as u8).collect();
        std::fs::write(&a, &bytes).unwrap();
        std::fs::write(&b, &bytes).unwrap();
        let ia = media_identity(&a).unwrap();
        assert!(ia.starts_with("file:3000000:"), "{ia}");
        assert_eq!(Some(ia.clone()), media_identity(&b));
        // the engine's own fingerprint of the same bytes, as media.status reports it
        let eng = filmcraft_engine::relink::identity_of_bytes(&bytes);
        assert_eq!(reported_identity(&json!({"size": eng.size, "fingerprint": format!("{:016x}", eng.fingerprint)})), Some(ia.clone()));
        // a change at the end is another file
        let mut changed = bytes.clone();
        if let Some(last) = changed.last_mut() {
            *last ^= 1;
        }
        std::fs::write(&b, &changed).unwrap();
        assert_ne!(media_identity(&b), Some(ia));
        // small files, empty files, missing files and folders
        std::fs::write(&a, b"tiny").unwrap();
        assert!(media_identity(&a).unwrap().starts_with("file:4:"));
        std::fs::write(&a, b"").unwrap();
        assert!(media_identity(&a).unwrap().starts_with("file:0:"));
        assert_eq!(media_identity(&dir.join("missing.mp4")), None);
        assert_eq!(media_identity(&dir), None);
        assert_eq!(reported_identity(&json!({"size": 3, "fingerprint": "not hex"})), None);
        assert_eq!(reported_identity(&json!(null)), None);
        std::fs::remove_dir_all(&dir).ok();
    }
}
