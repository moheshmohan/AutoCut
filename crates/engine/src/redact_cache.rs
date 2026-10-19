//! The redaction cache: what `clip.autoRedact` learned about a media file, kept on disk, so
//! redacting the same footage again — a rebuilt showcase, the same show in another language
//! whose shots play other windows of the same clips — decodes and asks almost nothing.
//!
//! Three kinds of entry, each for one media file (its content identity: size and the engine's
//! fingerprint of its first and last MiB, plus its picture size, length and interpreted rate):
//!
//! - **detections**: the faces and plates found at each analysed media time, by the detector
//!   settings (kinds, scores, the picture size the detectors see) and the detectors' own files
//!   (their pinned SHA-256s; a host's own detector by its id);
//! - **verdicts**: the vision check's answer for a track's crop (its kind, media time and box),
//!   by the vision model;
//! - **looks**: the places the vision model found the named targets at each media time it
//!   looked, by the model, the targets and the picture it was shown.
//!
//! Analysis times are on a grid of media time (`clip.autoRedact`), so two windows of a clip
//! share the frames they overlap. An entry is JSON, `<data dir>/redact-cache/<hash>.json`,
//! carrying its full key (another key that hashes the same is a miss), written atomically (a
//! temporary file renamed into place). The cache is capped ([`DEFAULT_CAP_BYTES`]): after a
//! write the least recently used entries (modification time, refreshed on every hit) go first.
//! An entry that cannot be read, or a cache that cannot be written, only means analysing again.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, SystemTime};

use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value, json};

/// Bump when what an entry means changes (the analysis, the crops, the grid): older entries miss.
pub const VERSION: u32 = 1;
/// Directory name under the app data directory.
pub const DIR_NAME: &str = "redact-cache";
/// Default size cap of the cache directory.
pub const DEFAULT_CAP_BYTES: u64 = 512 * 1024 * 1024;
/// The largest entry read or written.
const MAX_ENTRY_BYTES: u64 = 64 * 1024 * 1024;
/// Files looked at when trimming.
const MAX_SCAN: usize = 100_000;
/// A temporary file older than this was left by a writer that died: removed when trimming.
const STALE_TMP: Duration = Duration::from_secs(3600);
const EXT: &str = "json";

/// Unique temporary names within this process.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// The detections of one media file at each analysed media time (ticks).
pub type Detections = BTreeMap<i64, Vec<filmcraft_vision::Detection>>;
/// The vision check's answers, by `kind|ticks` of the moment cropped: the box asked about, and
/// the answer.
pub type Verdicts = BTreeMap<String, Vec<([f64; 4], Option<bool>)>>;
/// Answers kept for one moment at most (different tracks at the same time).
const MAX_VERDICTS_AT: usize = 32;
/// A box overlapping a kept one this much (IoU) is the same thing.
const SAME_BOX: f64 = 0.5;
/// The named targets' places at each media time looked at (ticks): `(target, [x, y, w, h])`.
pub type Looks = BTreeMap<i64, Vec<(usize, [f64; 4])>>;

/// The redaction cache in one directory (see the module docs).
#[derive(Debug)]
pub struct RedactCache {
    dir: PathBuf,
    cap: u64,
    /// One read-merge-write at a time (workers analysing two windows of one file).
    write: Mutex<()>,
}

impl RedactCache {
    pub fn new(dir: impl Into<PathBuf>, cap_bytes: u64) -> Self {
        Self { dir: dir.into(), cap: cap_bytes, write: Mutex::new(()) }
    }

    /// The cache in `<data_dir>/redact-cache` with the default cap.
    pub fn in_data_dir(data_dir: &Path) -> Self {
        Self::new(data_dir.join(DIR_NAME), DEFAULT_CAP_BYTES)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn path(&self, key: &str) -> PathBuf {
        // FNV-1a over the key; the key itself is checked on reading
        let h = key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3));
        self.dir.join(format!("{h:016x}.{EXT}"))
    }

    /// The value kept under `key`, its last use refreshed; `None` for a miss, a damaged entry or
    /// another key's.
    pub fn load<T: DeserializeOwned>(&self, key: &str) -> Option<T> {
        if cfg!(target_arch = "wasm32") {
            return None;
        }
        let path = self.path(key);
        let file = std::fs::File::open(&path).ok()?;
        if file.metadata().ok()?.len() > MAX_ENTRY_BYTES {
            return None;
        }
        let mut text = String::new();
        file.take(MAX_ENTRY_BYTES).read_to_string(&mut text).ok()?;
        let v: Value = serde_json::from_str(&text).ok()?;
        if v["version"].as_u64() != Some(u64::from(VERSION)) || v["key"].as_str() != Some(key) {
            return None;
        }
        let value = serde_json::from_value(v.get("value")?.clone()).ok()?;
        touch(&path);
        Some(value)
    }

    /// Keep `value` under `key` (atomically), then trim the cache to its cap. Best effort.
    pub fn store<T: Serialize>(&self, key: &str, value: &T) {
        if cfg!(target_arch = "wasm32") {
            return;
        }
        let Ok(bytes) = serde_json::to_vec(&json!({"version": VERSION, "key": key, "value": value})) else { return };
        if bytes.len() as u64 > MAX_ENTRY_BYTES.min(self.cap) || std::fs::create_dir_all(&self.dir).is_err() {
            return;
        }
        let path = self.path(key);
        let tmp = self.dir.join(format!(
            ".{}.{}-{}.tmp",
            path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default(),
            std::process::id(),
            TMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        if std::fs::write(&tmp, &bytes).is_err() || std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        self.trim();
    }

    /// Read the value under `key` (or start from the default), change it, write it back: one
    /// at a time in this process.
    pub fn update<T: Default + Serialize + DeserializeOwned>(&self, key: &str, change: impl FnOnce(&mut T)) {
        let _g = self.write.lock().unwrap_or_else(PoisonError::into_inner);
        let mut v: T = self.load(key).unwrap_or_default();
        change(&mut v);
        self.store(key, &v);
    }

    /// Entries and their total size in bytes.
    pub fn usage(&self) -> (usize, u64) {
        let e = self.entries();
        (e.len(), e.iter().map(|e| e.1).fold(0u64, u64::saturating_add))
    }

    /// `(path, bytes, last use)` of every entry; stale temporary files are removed on the way.
    fn entries(&self) -> Vec<(PathBuf, u64, SystemTime)> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return Vec::new() };
        let now = SystemTime::now();
        let mut out = Vec::new();
        for e in rd.filter_map(|e| e.ok()).take(MAX_SCAN) {
            let path = e.path();
            let Ok(meta) = e.metadata() else { continue };
            let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            match path.extension().and_then(|x| x.to_str()) {
                Some(EXT) if meta.is_file() => out.push((path, meta.len(), modified)),
                Some("tmp") if now.duration_since(modified).is_ok_and(|age| age > STALE_TMP) => {
                    let _ = std::fs::remove_file(&path);
                }
                _ => {}
            }
        }
        out
    }

    /// Remove the least recently used entries until the cache fits its cap.
    fn trim(&self) {
        let mut entries = self.entries();
        let mut total: u64 = entries.iter().map(|e| e.1).fold(0u64, u64::saturating_add);
        if total <= self.cap {
            return;
        }
        entries.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
        for (path, len, _) in entries {
            if total <= self.cap {
                break;
            }
            if std::fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(len);
            }
        }
    }
}

/// Mark an entry as just used (LRU order; access times are unreliable, so the modification time).
fn touch(path: &Path) {
    if let Ok(f) = std::fs::File::options().write(true).open(path) {
        let _ = f.set_modified(SystemTime::now());
    }
}

/// A short digest of the downloaded detectors' pinned files: a new model is a new key.
pub fn detector_pins() -> String {
    let joined: String = filmcraft_vision::catalog::FILES.iter().map(|f| format!("{}={}", f.name, f.sha256)).collect::<Vec<_>>().join(",");
    let h = joined.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3));
    format!("{h:016x}")
}

/// The key of a vision verdict: the track's kind and the media time of its crop.
pub fn verdict_key(kind: filmcraft_vision::Kind, t: filmcraft_time::Tick) -> String {
    format!("{}|{}", kind.name(), t.0)
}

/// Overlap of two boxes `[x, y, w, h]`: intersection over union (0 for empty or hostile boxes).
fn iou(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    if !a.iter().chain(b).all(|v| v.is_finite()) {
        return 0.0;
    }
    let ix = ((a[0] + a[2]).min(b[0] + b[2]) - a[0].max(b[0])).max(0.0);
    let iy = ((a[1] + a[3]).min(b[1] + b[3]) - a[1].max(b[1])).max(0.0);
    let inter = ix * iy;
    let union = a[2] * a[3] + b[2] * b[3] - inter;
    if union > 0.0 && union.is_finite() { inter / union } else { 0.0 }
}

/// The answer kept for a track: any of its moments (`(ticks, box)`, best first) with a kept
/// answer about a box overlapping its own there. A track seen again in another window of the
/// clip is the same thing, whichever of its moments was cropped then.
pub fn verdict_for<'a>(
    kept: &Verdicts,
    kind: filmcraft_vision::Kind,
    moments: impl IntoIterator<Item = (filmcraft_time::Tick, &'a [f64; 4])>,
) -> Option<Option<bool>> {
    moments.into_iter().find_map(|(t, b)| kept.get(&verdict_key(kind, t))?.iter().find(|(k, _)| iou(k, b) >= SAME_BOX).map(|(_, a)| *a))
}

/// `new` answers merged into `kept` (an answer about the same box at the same moment replaced).
pub fn merge_verdicts(kept: &mut Verdicts, new: &Verdicts) {
    for (k, list) in new {
        let at = kept.entry(k.clone()).or_default();
        for (b, a) in list {
            at.retain(|(o, _)| iou(o, b) < SAME_BOX);
            at.push((*b, *a));
        }
        let excess = at.len().saturating_sub(MAX_VERDICTS_AT);
        at.drain(..excess);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-redact-cache-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn entries_round_trip_by_key_and_merge() {
        let dir = tmp("roundtrip");
        let c = RedactCache::in_data_dir(&dir);
        assert!(c.load::<Detections>("a").is_none());
        let d = filmcraft_vision::Detection { kind: filmcraft_vision::Kind::Face, score: 0.9, bbox: [0.1, 0.2, 0.3, 0.4], points: vec![[0.5, 0.5]] };
        c.update::<Detections>("a", |e| {
            e.insert(100, vec![d.clone()]);
        });
        c.update::<Detections>("a", |e| {
            e.insert(200, Vec::new());
        });
        let got = c.load::<Detections>("a").unwrap();
        assert_eq!(got.get(&100), Some(&vec![d]));
        assert_eq!(got.len(), 2);
        assert!(dir.join(DIR_NAME).is_dir());
        // another key is another entry; a value of another shape is a miss, not a panic
        assert!(c.load::<Detections>("b").is_none());
        assert!(c.load::<Verdicts>("a").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn damaged_or_foreign_entries_are_misses() {
        let dir = tmp("damaged");
        let c = RedactCache::new(&dir, DEFAULT_CAP_BYTES);
        c.store("k", &Verdicts::from([("face|1".to_string(), vec![([0.1, 0.1, 0.2, 0.2], Some(true))])]));
        let path = c.path("k");
        for bad in [&b"{not json"[..], b"", br#"{"version": 1, "key": "other", "value": {}}"#, br#"{"version": 99, "key": "k", "value": {}}"#] {
            std::fs::write(&path, bad).unwrap();
            assert!(c.load::<Verdicts>("k").is_none());
        }
        // an unwritable cache only means no cache
        let blocked = RedactCache::new(dir.join("file-not-dir"), DEFAULT_CAP_BYTES);
        std::fs::write(dir.join("file-not-dir"), b"x").unwrap();
        blocked.store("k", &1);
        assert!(blocked.load::<i32>("k").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_cap_removes_the_least_recently_used() {
        let dir = tmp("cap");
        let c = RedactCache::new(&dir, 2500);
        let big = vec![7u8; 300];
        for k in ["one", "two", "three"] {
            c.store(k, &big);
            std::thread::sleep(Duration::from_millis(20));
        }
        // "one" is used again: "two" is now the oldest
        assert!(c.load::<Vec<u8>>("one").is_some());
        std::thread::sleep(Duration::from_millis(20));
        c.store("four", &big);
        let (n, bytes) = c.usage();
        assert!(bytes <= 2500, "{n} entries, {bytes} bytes");
        assert!(c.load::<Vec<u8>>("two").is_none());
        assert!(c.load::<Vec<u8>>("one").is_some() && c.load::<Vec<u8>>("four").is_some());
        // an entry larger than the cap is not kept
        c.store("huge", &vec![1u8; 4000]);
        assert!(c.load::<Vec<u8>>("huge").is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_track_finds_its_answer_at_any_of_its_moments() {
        use filmcraft_time::Tick;
        use filmcraft_vision::Kind;
        assert_eq!(verdict_key(Kind::Plate, Tick(42)), "plate|42");
        assert_eq!(detector_pins().len(), 16);
        let mut kept = Verdicts::new();
        let a = [0.1, 0.1, 0.2, 0.2];
        merge_verdicts(&mut kept, &Verdicts::from([("face|10".to_string(), vec![(a, Some(false))])]));
        // the same face asked about at its best moment then; now its best is another moment
        let moved = [0.12, 0.1, 0.2, 0.2];
        assert_eq!(verdict_for(&kept, Kind::Face, [(Tick(30), &moved), (Tick(10), &moved)]), Some(Some(false)));
        // another box at that moment, another kind, another moment: not known
        assert_eq!(verdict_for(&kept, Kind::Face, [(Tick(10), &[0.6, 0.6, 0.2, 0.2])]), None);
        assert_eq!(verdict_for(&kept, Kind::Plate, [(Tick(10), &a)]), None);
        assert_eq!(verdict_for(&kept, Kind::Face, [(Tick(11), &a)]), None);
        // an answer about the same box again replaces the old one; many at one moment are capped
        merge_verdicts(&mut kept, &Verdicts::from([("face|10".to_string(), vec![(moved, None)])]));
        assert_eq!(kept["face|10"].len(), 1);
        assert_eq!(verdict_for(&kept, Kind::Face, [(Tick(10), &a)]), Some(None));
        let many: Vec<([f64; 4], Option<bool>)> = (0..100).map(|i| ([f64::from(i) * 0.01, 0.0, 0.005, 0.005], Some(true))).collect();
        merge_verdicts(&mut kept, &Verdicts::from([("face|10".to_string(), many)]));
        assert_eq!(kept["face|10"].len(), MAX_VERDICTS_AT);
        // hostile boxes overlap nothing
        assert_eq!(iou(&[f64::NAN, 0.0, 1.0, 1.0], &a), 0.0);
        assert_eq!(iou(&[0.0, 0.0, 0.0, 0.0], &[0.0, 0.0, 0.0, 0.0]), 0.0);
    }
}
