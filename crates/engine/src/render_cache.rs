//! Content-addressed chapter/VFX artifacts. Complete immutable entries are published by rename
//! under an OS advisory lock; projects and source media never live in this disposable cache.
use crate::commands::{CommandSpec, always, bad};
use crate::{Result, Session};
use filmcraft_project::{ItemId, ItemKind, MediaRef};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Write},
    path::{Path, PathBuf},
};
const FORMAT: &str = "autocut.renderCache";
const MAX_JSON: u64 = 16 << 20;
pub const ENGINE: &str = env!("AUTOCUT_RENDER_ENGINE_HASH");
fn spec(id: &'static str, label: &'static str, params: &'static str, run: fn(&mut Session, &Value) -> Result<Value>) -> CommandSpec {
    CommandSpec { id, label, params, run, enabled: always, menu: &[], shortcut: None, journal: false }
}
pub fn commands() -> Vec<CommandSpec> {
    vec![
        spec(
            "render.cachePlan",
            "Plan cached render",
            r#"{"sequence":id?,"start":ticks?,"end":ticks?,"settings":{…},"external":JSON?,"dependencies":[path]?,"rehash":bool=false}"#,
            plan,
        ),
        spec("render.cacheInspect", "Inspect cached artifact", r#"{"key":sha256}"#, inspect),
        spec("render.cacheCommit", "Commit complete render", r#"{"key":sha256,"path":path,"manifest":CachePlan}"#, commit),
    ]
}
pub fn digest_file(path: &Path) -> std::result::Result<(String, u64), String> {
    let mut f = filmcraft_media::reader::open_regular_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let before = f.metadata().map_err(|e| e.to_string())?;
    let mut hash = Sha256::new();
    let mut total = 0u64;
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf).map_err(|e| e.to_string())?;
        if n == 0 {
            break;
        }
        total = total.checked_add(n as u64).ok_or("file size overflows")?;
        if total > before.len() {
            return Err("source grew while hashing; retry when it is stable".into());
        }
        hash.update(buf.get(..n).ok_or("invalid read size")?);
    }
    let after = f.metadata().map_err(|e| e.to_string())?;
    if total != before.len() || before.modified().ok() != after.modified().ok() {
        return Err("source changed while hashing; retry".into());
    }
    Ok((format!("{:x}", hash.finalize()), total))
}
fn digest(v: &Value) -> Result<String> {
    Ok(format!("{:x}", Sha256::digest(serde_json::to_vec(v).map_err(|e| bad("render.cachePlan", e.to_string()))?)))
}
/// How long a file must have been left alone before its digest is remembered. A second change
/// inside this window can leave the same times behind (file systems keep them to between a
/// nanosecond and two seconds), so a file changed this recently is hashed again next time.
const SETTLE: std::time::Duration = std::time::Duration::from_secs(2);
const MEMO: &str = "source-digests.json";
const MEMO_FORMAT: &str = "autocut.sourceDigests";
const MEMO_MAX: usize = 20000;
/// What tells one state of a file from the next without reading it, and when it last changed
/// (nanoseconds since 1970): which file it is, its size, and when its bytes and its inode were
/// last changed. A program can put a file's modified time back, never its change time, so bytes
/// changed in place always show. `None` where the platform does not report all of this: there
/// every source is hashed every time.
#[cfg(unix)]
fn stamp(m: &std::fs::Metadata) -> Option<(String, i128)> {
    use std::os::unix::fs::MetadataExt;
    let at = |seconds: i64, nanos: i64| i128::from(seconds) * 1_000_000_000 + i128::from(nanos);
    let (modified, changed) = (at(m.mtime(), m.mtime_nsec()), at(m.ctime(), m.ctime_nsec()));
    m.is_file().then(|| (format!("{}:{}:{}:{modified}:{changed}", m.dev(), m.ino(), m.size()), modified.max(changed)))
}
#[cfg(not(unix))]
fn stamp(_: &std::fs::Metadata) -> Option<(String, i128)> {
    None
}
/// Source digests kept between plans in the cache folder, so a plan reads only the sources that
/// changed since the last one. Every plan used to read every source in full, and the commit
/// read them all again: 14 s each for a 101-source edit. A digest is used only while the file's
/// [`stamp`] is the one it was hashed under. All of this is an optimisation: a missing,
/// unreadable or damaged memo means hashing, never an error.
struct SourceDigests {
    file: Option<PathBuf>,
    /// Path → the stamp it was hashed under, its SHA-256 and its size.
    entries: BTreeMap<String, (String, String, u64)>,
    used: BTreeSet<String>,
    settle: std::time::Duration,
    trust: bool,
    changed: bool,
    hashed: usize,
    reused: usize,
}
impl SourceDigests {
    /// The memo of `folder` (none: nothing is kept). `trust: false` hashes every file anyway.
    fn open(folder: Option<&Path>, settle: std::time::Duration, trust: bool) -> Self {
        let file = folder.map(|d| d.join(MEMO));
        let entries = file
            .as_deref()
            .and_then(|f| read(f).ok())
            .filter(|v| v["format"] == MEMO_FORMAT && v["version"] == 1)
            .and_then(|v| serde_json::from_value(v["entries"].clone()).ok())
            .unwrap_or_default();
        Self { file, entries, used: BTreeSet::new(), settle, trust, changed: false, hashed: 0, reused: 0 }
    }
    fn digest(&mut self, path: &Path) -> std::result::Result<(String, u64), String> {
        let name = path.to_string_lossy().into_owned();
        let before = std::fs::metadata(path).ok().as_ref().and_then(stamp);
        if self.trust
            && let (Some((stamp, _)), Some((known, hash, bytes))) = (&before, self.entries.get(&name))
            && stamp == known
            && hash.len() == 64
            && hash.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
        {
            self.reused += 1;
            self.used.insert(name);
            return Ok((hash.clone(), *bytes));
        }
        let result = digest_file(path)?;
        self.hashed += 1;
        // Remembered only when the file is the same after the read as before it, and has been
        // left alone long enough for its times to show the next change.
        let after = std::fs::metadata(path).ok().as_ref().and_then(stamp);
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().and_then(|d| i128::try_from(d.as_nanos()).ok());
        let settle = i128::try_from(self.settle.as_nanos()).unwrap_or(i128::MAX);
        match (before, after, now) {
            (Some(a), Some(b), Some(now)) if a == b && now.saturating_sub(b.1) >= settle => {
                self.entries.insert(name.clone(), (b.0, result.0.clone(), result.1));
                self.used.insert(name);
                self.changed = true;
            }
            _ => self.changed |= self.entries.remove(&name).is_some(),
        }
        Ok(result)
    }
    /// Keep what was learned. Over [`MEMO_MAX`] files, only the ones this run asked about stay.
    fn save(&mut self) {
        let Some(file) = self.file.as_deref().filter(|_| self.changed) else {
            return;
        };
        if self.entries.len() > MEMO_MAX {
            let used = &self.used;
            self.entries.retain(|path, _| used.contains(path));
        }
        let Ok(bytes) = serde_json::to_vec(&json!({"format":MEMO_FORMAT,"version":1,"entries":self.entries})) else {
            return;
        };
        if self.entries.len() > MEMO_MAX || bytes.len() as u64 > MAX_JSON {
            return;
        }
        if let Some(folder) = file.parent() {
            let _ = std::fs::create_dir_all(folder);
        }
        if filmcraft_format::atomic_write(file, &bytes).is_ok() {
            self.changed = false;
        }
    }
}
/// One writer per cache key, by an OS advisory lock on `<key>.lock`. Its holder removes the file
/// when it is done (every committed key used to leave one behind for good), so a writer that
/// opened the file before that and locked it after holds a file nobody else can reach: it checks
/// that its file is still the one at the path, and takes the new one if not.
fn lock_entry(path: &Path, cmd: &str) -> Result<std::fs::File> {
    for _ in 0..8 {
        let file = std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(path).map_err(|e| bad(cmd, e.to_string()))?;
        file.try_lock().map_err(|e| bad(cmd, format!("another writer owns this cache entry: {e}")))?;
        if is_at(&file, path) {
            return Ok(file);
        }
    }
    Err(bad(cmd, "the lock of this cache entry kept changing hands; run it again"))
}
#[cfg(unix)]
fn is_at(file: &std::fs::File, path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    matches!((file.metadata(), std::fs::symlink_metadata(path)), (Ok(a), Ok(b)) if a.dev() == b.dev() && a.ino() == b.ino())
}
/// Elsewhere than Unix an open file cannot be told from the one at its path, so there the lock
/// file is never removed and this is always true.
#[cfg(not(unix))]
fn is_at(_: &std::fs::File, _: &Path) -> bool {
    true
}
/// Remove the lock file while it is still held, then let it go.
fn unlock_entry(lock: std::fs::File, path: &Path) {
    if cfg!(unix) {
        let _ = std::fs::remove_file(path);
    }
    drop(lock);
}
fn dependencies(v: &Value, out: &mut BTreeSet<String>, depth: usize) -> Result<()> {
    if depth > 64 {
        return Err(bad("render.cachePlan", "dependency nesting too deep"));
    }
    match v {
        Value::Object(map) => {
            for (key, value) in map {
                if ["path", "file", "fontPath", "second_lens", "secondLens", "proxy"].contains(&key.as_str())
                    && let Some(path) = value.as_str().filter(|p| !p.is_empty())
                {
                    out.insert(path.into());
                }
                dependencies(value, out, depth + 1)?;
            }
        }
        Value::Array(a) => {
            for value in a {
                dependencies(value, out, depth + 1)?;
            }
        }
        _ => {}
    }
    Ok(())
}
fn plan(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "render.cachePlan";
    let mut sources = BTreeSet::new();
    let mut selected = Vec::new();
    let mut range = Value::Null;
    let sequence = match p.get("sequence") {
        None => s.state.active_sequence,
        Some(value) => Some(ItemId(value.as_u64().ok_or_else(|| bad(cmd, "sequence must be an unsigned identity"))?)),
    };
    if let Some(id) = sequence {
        let q = s.project.sequence(id).ok_or_else(|| bad(cmd, "sequence is absent"))?;
        let read = |key: &str, default: i64| {
            p.get(key).map_or(Some(default), Value::as_i64).filter(|v| *v >= 0 && *v <= Tick::MAX.0).ok_or_else(|| bad(cmd, format!("invalid {key} ticks")))
        };
        let start = read("start", 0)?;
        let end = read("end", q.duration().0)?;
        if end <= start || end > q.duration().0 {
            return Err(bad(cmd, "chapter range outside sequence"));
        }
        let time = TimeRange::new(Tick(start), Tick(end - start));
        range = json!({"start":start,"end":end});
        let mut queue = vec![(id, 0usize)];
        let mut seen = BTreeSet::new();
        while let Some((id, depth)) = queue.pop() {
            if depth > 64 || seen.len() > 10000 {
                return Err(bad(cmd, "nested dependencies exceed limits"));
            }
            if !seen.insert(id) {
                continue;
            }
            let mut it = s.project.item(id).cloned().ok_or_else(|| bad(cmd, "missing referenced media"))?;
            if let Some(value) = it.metadata.get("frameatlas.dependencies") {
                let paths: Vec<String> = serde_json::from_str(value).map_err(|e| bad(cmd, format!("invalid source dependencies: {e}")))?;
                if paths.len() > 1000 {
                    return Err(bad(cmd, "too many related source files"));
                }
                sources.extend(paths);
            }
            it.metadata.clear();
            match &mut it.kind {
                ItemKind::Sequence(q) => {
                    let q = std::sync::Arc::make_mut(q);
                    let nested = depth > 0;
                    for tr in q.video_tracks.iter_mut().chain(q.audio_tracks.iter_mut()) {
                        if !nested && tr.kind == filmcraft_project::TrackKind::Video {
                            tr.transitions.retain(|c| c.start < time.end() && c.start + c.duration > time.start);
                            let handles: BTreeSet<_> = tr.transitions.iter().flat_map(|t| [t.from, t.to]).flatten().collect();
                            tr.items.retain(|c| c.start < time.end() && c.end() > time.start || handles.contains(&c.id));
                        }
                        // Audio processors have history, tails and lookahead; ducking may
                        // depend on a neighbouring clip. Keep complete audio dependencies.
                        for clip in &tr.items {
                            queue.push((clip.item, depth + 1));
                        }
                    }
                    if !nested {
                        for tr in &mut q.caption_tracks {
                            tr.captions.retain(|c| c.start < time.end() && c.start + c.duration > time.start);
                        }
                    }
                }
                ItemKind::Subclip { parent, .. } => queue.push((*parent, depth + 1)),
                ItemKind::Media(m) => {
                    // Cached chapters use original media, just like the export renderer.
                    // An absent editing proxy must not block or change that cache entry.
                    m.proxy = None;
                    if let MediaRef::File { path } = &m.media {
                        s.media.remove(id);
                        sources.insert(path.clone());
                        if m.info.kind == filmcraft_media::MediaKind::ImageSequence {
                            let frames = crate::media_pool::image_sequence_frames(path, s.services.as_ref()).map_err(|e| bad(cmd, e.to_string()))?;
                            if frames.len() > 10000 {
                                return Err(bad(cmd, "image sequence exceeds 10000 cache dependencies; split the sequence"));
                            }
                            for frame in frames {
                                sources.insert(frame.ok_or_else(|| bad(cmd, "image sequence has missing frames"))?);
                            }
                        }
                    }
                    if m.offline {
                        return Err(bad(cmd, "relink offline media before caching"));
                    }
                }
                _ => {}
            }
            let value = serde_json::to_value(it).map_err(|e| bad(cmd, e.to_string()))?;
            dependencies(&value, &mut sources, 0)?;
            // The item's digest, not the item: a long edit's items are tens of megabytes, and
            // the identity is kept in every manifest and passed back to `render.cacheCommit`.
            selected.push(json!(digest(&value)?));
        }
    }
    if let Some(v) = p.get("dependencies") {
        for path in v.as_array().filter(|a| a.len() <= 10000).ok_or_else(|| bad(cmd, "dependencies needs up to 10000 local files"))? {
            sources.insert(path.as_str().filter(|s| !s.is_empty() && s.len() <= 4096).ok_or_else(|| bad(cmd, "dependency must be a path"))?.into());
        }
    }
    if selected.is_empty() && sources.is_empty() {
        return Err(bad(cmd, "provide a sequence or external composition dependencies"));
    }
    let rehash = p.get("rehash").map_or(Ok(false), |v| v.as_bool().ok_or_else(|| bad(cmd, "rehash must be boolean")))?;
    let mut digests = SourceDigests::open(dir(s).as_deref(), SETTLE, !rehash);
    let hashed: std::result::Result<BTreeMap<String, (String, u64)>, String> =
        sources.into_iter().map(|path| digests.digest(Path::new(&path)).map(|d| (path, d))).collect();
    digests.save();
    let hashes = hashed.map_err(|e| bad(cmd, e))?;
    // These resources render outside ProjectItem: a library LUT can change pixels
    // without changing the clip's `lib:<id>` reference. Keep them conservatively.
    let resources = (!selected.is_empty())
        .then(|| {
            digest(&json!({"settings":s.project.settings,"luts":s.project.luts,
            "sourceGraphics":s.project.source_graphics,"motionTracks":s.project.motion_tracks}))
        })
        .transpose()?;
    let identity = json!({"format":FORMAT,"version":2,"engine":ENGINE,"range":range,"items":selected,"resources":resources,"sources":hashes,"settings":p.get("settings"),"external":p.get("external")});
    if serde_json::to_vec(&identity).map_err(|e| bad(cmd, e.to_string()))?.len() as u64 > MAX_JSON / 2 {
        return Err(bad(cmd, "cache identity exceeds 8 MiB; split into smaller chapters"));
    }
    let key = digest(&identity)?;
    let hit = lookup(s, &key).map_err(|e| bad(cmd, e))?;
    Ok(
        json!({"key":key,"identity":identity,"hit":hit.is_some(),"artifact":hit,"cacheAvailable":dir(s).is_some(),"sourceDigests":{"hashed":digests.hashed,"reused":digests.reused}}),
    )
}
fn key<'a>(p: &'a Value, cmd: &str) -> Result<&'a str> {
    p["key"]
        .as_str()
        .filter(|s| s.len() == 64 && s.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase()))
        .ok_or_else(|| bad(cmd, "key must be lowercase SHA-256"))
}
pub(crate) fn cache_root(s: &Session) -> Option<PathBuf> {
    // Headless CLI sessions intentionally do not install a writable preferences path. Their
    // narration cache already records the explicitly selected native data directory.
    s.media_cache_dir().or_else(|| s.tts_disk.as_ref().and_then(|c| c.dir().parent()).map(|p| p.join("Media Cache")))
}
fn dir(s: &Session) -> Option<PathBuf> {
    cache_root(s).map(|p| p.join("render-cache-v1"))
}
fn read(path: &Path) -> std::result::Result<Value, String> {
    let f = filmcraft_media::reader::open_regular_file(path).map_err(|e| e.to_string())?;
    if f.metadata().map_err(|e| e.to_string())?.len() > MAX_JSON {
        return Err("cache manifest is too large".into());
    }
    serde_json::from_reader(f.take(MAX_JSON + 1)).map_err(|e| e.to_string())
}
fn lookup(s: &Session, key: &str) -> std::result::Result<Option<Value>, String> {
    let Some(cache) = dir(s) else {
        return Ok(None);
    };
    let root = cache.join(key);
    if !root.exists() {
        return Ok(None);
    }
    let m = match read(&root.join("manifest.json")) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    if m["key"] != key || m["format"] != FORMAT || m["engine"] != ENGINE {
        return Ok(None);
    }
    if digest(&m["plan"]["identity"]).ok().as_deref() != Some(key) {
        return Ok(None);
    }
    let path = root.join("artifact");
    let (hash, bytes) = match digest_file(&path) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    if m["sha256"] != hash || m["bytes"].as_u64() != Some(bytes) {
        return Ok(None);
    }
    Ok(Some(json!({"path":path,"sha256":hash,"bytes":bytes,"manifest":m})))
}
fn inspect(s: &mut Session, p: &Value) -> Result<Value> {
    let k = key(p, "render.cacheInspect")?;
    let result = lookup(s, k).map_err(|e| bad("render.cacheInspect", e))?;
    Ok(json!({"key":k,"hit":result.is_some(),"artifact":result}))
}
fn commit(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "render.cacheCommit";
    let k = key(p, cmd)?;
    let manifest = &p["manifest"];
    if manifest["key"] != k || manifest["identity"]["engine"] != ENGINE || digest(&manifest["identity"])? != k {
        return Err(bad(cmd, "manifest identity does not match this build and key"));
    }
    if let Some(sources) = manifest["identity"]["sources"].as_object() {
        // a source the plan hashed is read again only if it changed since
        let mut digests = SourceDigests::open(dir(s).as_deref(), SETTLE, true);
        let unchanged = sources.iter().try_for_each(|(path, expected)| {
            let (hash, bytes) = digests.digest(Path::new(path)).map_err(|e| bad(cmd, e))?;
            if expected == &json!([hash, bytes]) { Ok(()) } else { Err(bad(cmd, "source changed during render; cache publication refused")) }
        });
        digests.save();
        unchanged?;
    }
    let path = Path::new(p["path"].as_str().ok_or_else(|| bad(cmd, "provide a completed render path"))?);
    let (hash, bytes) = digest_file(path).map_err(|e| bad(cmd, e))?;
    if bytes == 0 {
        return Err(bad(cmd, "empty render cannot be cached"));
    }
    let cache = dir(s).ok_or_else(|| bad(cmd, "configure a native cache directory first"))?;
    std::fs::create_dir_all(&cache).map_err(|e| bad(cmd, e.to_string()))?;
    let lock_path = cache.join(format!("{k}.lock"));
    let lock = lock_entry(&lock_path, cmd)?;
    let published = publish(s, k, &cache, manifest, path, (&hash, bytes));
    unlock_entry(lock, &lock_path);
    published
}
/// Publish a complete render as the entry of `k`, under that key's lock.
fn publish(s: &Session, k: &str, cache: &Path, manifest: &Value, path: &Path, (hash, bytes): (&str, u64)) -> Result<Value> {
    let cmd = "render.cacheCommit";
    if let Some(hit) = lookup(s, k).map_err(|e| bad(cmd, e))? {
        return Ok(json!({"key":k,"hit":true,"artifact":hit}));
    }
    let destination = cache.join(k);
    // Corrupt immutable entries remain available for diagnosis, away from the new publication.
    if destination.exists() {
        std::fs::rename(
            &destination,
            cache
                .join(format!("{k}.corrupt-{}", chrono::DateTime::<chrono::Utc>::from(std::time::SystemTime::now()).timestamp_nanos_opt().unwrap_or_default())),
        )
        .map_err(|e| bad(cmd, e.to_string()))?;
    }
    let staged = cache.join(format!("{k}.pending-{}", std::process::id()));
    std::fs::create_dir(&staged).map_err(|e| bad(cmd, format!("stale/in-progress publication {}: {e}", staged.display())))?;
    let result = (|| {
        let input = filmcraft_media::reader::open_regular_file(path).map_err(|e| bad(cmd, e.to_string()))?;
        let mut output = std::fs::File::create_new(staged.join("artifact")).map_err(|e| bad(cmd, e.to_string()))?;
        std::io::copy(&mut input.take(bytes.saturating_add(1)), &mut output).map_err(|e| bad(cmd, e.to_string()))?;
        output.sync_all().map_err(|e| bad(cmd, e.to_string()))?;
        if digest_file(&staged.join("artifact")).map_err(|e| bad(cmd, e))? != (hash.to_string(), bytes) {
            return Err(bad(cmd, "render changed while copying; no cache entry published"));
        }
        let m = json!({"format":FORMAT,"version":1,"key":k,"engine":ENGINE,"sha256":hash,"bytes":bytes,"plan":manifest});
        let mut f = std::fs::File::create_new(staged.join("manifest.json")).map_err(|e| bad(cmd, e.to_string()))?;
        f.write_all(&serde_json::to_vec(&m).map_err(|e| bad(cmd, e.to_string()))?).map_err(|e| bad(cmd, e.to_string()))?;
        f.sync_all().map_err(|e| bad(cmd, e.to_string()))?;
        std::fs::rename(&staged, &destination).map_err(|e| bad(cmd, e.to_string()))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = std::fs::remove_file(staged.join("artifact"));
        let _ = std::fs::remove_file(staged.join("manifest.json"));
        let _ = std::fs::remove_dir(&staged);
    }
    result?;
    Ok(json!({"key":k,"hit":false,"artifact":lookup(s,k).map_err(|e|bad(cmd,e))?}))
}

pub(crate) fn copy_artifact(artifact: &Value, path: &Path) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    let source = artifact["path"].as_str().ok_or_else(|| Error::new(ErrorKind::InvalidData, "cache artifact path is missing"))?;
    let expected = artifact["bytes"].as_u64().filter(|n| *n > 0).ok_or_else(|| Error::new(ErrorKind::InvalidData, "cache artifact size is missing"))?;
    let hash = artifact["sha256"].as_str().ok_or_else(|| Error::new(ErrorKind::InvalidData, "cache artifact digest is missing"))?;
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    filmcraft_format::atomic_write_new_with(path, |output| {
        let mut input = filmcraft_media::reader::open_regular_file(Path::new(source))?.take(expected.saturating_add(1));
        let mut actual = Sha256::new();
        let mut total = 0u64;
        let mut buffer = [0u8; 65536];
        loop {
            let n = input.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            total = total.checked_add(n as u64).ok_or_else(|| Error::new(ErrorKind::InvalidData, "cache copy is too large"))?;
            if total > expected {
                return Err(Error::new(ErrorKind::InvalidData, "cache artifact changed size"));
            }
            let bytes = buffer.get(..n).ok_or_else(|| Error::new(ErrorKind::InvalidData, "invalid read size"))?;
            actual.update(bytes);
            output.write_all(bytes)?;
        }
        if total != expected || format!("{:x}", actual.finalize()) != hash {
            return Err(Error::new(ErrorKind::InvalidData, "cache copy failed integrity verification"));
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn session(name: &str) -> (Session, PathBuf) {
        let dir = crate::media_test_util::tmp_dir(name);
        let s = Session { prefs_path: Some(dir.join("prefs.json")), ..Default::default() };
        (s, dir)
    }
    #[test]
    fn content_settings_and_external_parameters_invalidate_and_corruption_misses() {
        let (mut s, dir) = session("render-cache");
        let source = dir.join("composition.txt");
        std::fs::write(&source, b"scene one").unwrap();
        let p = json!({"dependencies":[source],"external":{"fps":30,"width":3840},"settings":{"audio":true}});
        let first = s.execute("render.cachePlan", p.clone()).unwrap();
        assert_eq!(first["hit"], false);
        assert_eq!(first["sourceDigests"], json!({"hashed":1,"reused":0}));
        assert!(s.execute("render.cachePlan", json!({"dependencies":[source],"rehash":"yes"})).is_err());
        let artifact = dir.join("render.bin");
        std::fs::write(&artifact, b"complete artifact").unwrap();
        let saved = s.execute("render.cacheCommit", json!({"key":first["key"],"path":artifact,"manifest":first})).unwrap();
        assert_eq!(saved["hit"], false);
        assert_eq!(s.execute("render.cachePlan", p.clone()).unwrap()["hit"], true);
        let mut changed = p.clone();
        changed["external"]["fps"] = json!(60);
        assert_ne!(s.execute("render.cachePlan", changed).unwrap()["key"], first["key"]);
        std::fs::write(&source, b"scene two").unwrap();
        assert_ne!(s.execute("render.cachePlan", p).unwrap()["key"], first["key"]);
        let cache_path = PathBuf::from(saved["artifact"]["path"].as_str().unwrap());
        std::fs::write(cache_path, b"truncated").unwrap();
        assert_eq!(s.execute("render.cacheInspect", json!({"key":first["key"]})).unwrap()["hit"], false);
    }
    #[test]
    fn chapter_video_change_invalidates_only_the_affected_range() {
        let (mut s, _) = session("chapter-cache");
        s.execute("file.openDemoProject", json!({})).unwrap();
        let q = s.active_sequence().unwrap();
        let last = q.video_tracks[0].items.last().unwrap().id;
        let first = q.video_tracks[0].items.first().unwrap();
        let finish = first.end().0.min(Tick::from_seconds_f64(1.0).0);
        let p = json!({"start":0,"end":finish,"settings":{"width":3840}});
        let before = s.execute("render.cachePlan", p.clone()).unwrap();
        s.edit_sequence("Later shot change", |q, _, _| {
            let (_, c) = q.find_item_mut(last).unwrap();
            c.gain_db = -9.0;
            Ok(())
        })
        .unwrap();
        assert_eq!(s.execute("render.cachePlan", p.clone()).unwrap()["key"], before["key"]);
        s.edit_sequence("Sequence colour", |q, _, _| {
            q.settings.color.wide_gamut = true;
            Ok(())
        })
        .unwrap();
        assert_ne!(s.execute("render.cachePlan", p).unwrap()["key"], before["key"]);
    }
    /// A long edit's items serialize to more than the 8 MiB an identity may be (a real 17 MB
    /// project came to 99 % of it): the plan was refused, and with it every cached delivery.
    #[test]
    fn a_large_edit_still_has_a_small_identity() {
        let (mut s, _) = session("cache-large-edit");
        s.execute("file.openDemoProject", json!({})).unwrap();
        let p = json!({"settings":{"width":3840}});
        let small = s.execute("render.cachePlan", p.clone()).unwrap();
        s.edit_sequence("A very long clip name", |q, _, _| {
            q.video_tracks[0].items[0].name = "x".repeat(9 << 20);
            Ok(())
        })
        .unwrap();
        let large = s.execute("render.cachePlan", p).unwrap();
        assert_ne!(large["key"], small["key"], "the changed clip is still part of the key");
        assert!(serde_json::to_vec(&large["identity"]).unwrap().len() < 64 << 10);
        assert!(large["identity"]["items"].as_array().unwrap().iter().all(|v| v.as_str().is_some_and(|d| d.len() == 64)));
    }
    #[test]
    fn concurrent_publication_and_missing_sources_are_actionable() {
        let (mut s, dir) = session("cache-writer");
        let source = dir.join("source");
        std::fs::write(&source, b"source").unwrap();
        let plan = s.execute("render.cachePlan", json!({"dependencies":[source]})).unwrap();
        let cache = super::dir(&s).unwrap();
        std::fs::create_dir_all(&cache).unwrap();
        let lock = std::fs::File::create(cache.join(format!("{}.lock", plan["key"].as_str().unwrap()))).unwrap();
        lock.try_lock().unwrap();
        assert!(s.execute("render.cacheCommit", json!({"key":plan["key"],"manifest":plan,"path":source})).is_err());
        drop(lock);
        assert!(s.execute("render.cachePlan", json!({"dependencies":["/missing/animation-source"]})).is_err());
    }
    /// Every plan read every source in full (14 s for a 101-source edit), and the commit read
    /// them all again.
    #[cfg(unix)]
    #[test]
    fn an_unchanged_source_is_hashed_once_and_a_changed_one_again() {
        let dir = crate::media_test_util::tmp_dir("cache-source-digests");
        let (source, folder) = (dir.join("source.bin"), dir.join("cache"));
        std::fs::write(&source, b"scene one").unwrap();
        let digest = |settle, trust| {
            let mut memo = SourceDigests::open(Some(&folder), settle, trust);
            let (hash, _) = memo.digest(&source).unwrap();
            memo.save();
            (hash, memo.hashed, memo.reused)
        };
        let none = std::time::Duration::ZERO;
        // a file changed moments ago is not remembered: its times might not show the next change
        let first = digest(SETTLE, true);
        assert_eq!(digest(SETTLE, true), (first.0.clone(), 1, 0));
        // one that has settled is read once
        assert_eq!(digest(none, true), (first.0.clone(), 1, 0));
        assert_eq!(digest(none, true), (first.0.clone(), 0, 1));
        assert_eq!(digest(none, false), (first.0.clone(), 1, 0), "`rehash` reads it whatever is remembered");
        // the same size and modified time over other bytes: its change time still gives it away
        let modified = std::fs::metadata(&source).unwrap().modified().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(&source, b"scene two").unwrap();
        std::fs::File::options().write(true).open(&source).unwrap().set_modified(modified).unwrap();
        assert_eq!(std::fs::metadata(&source).unwrap().modified().unwrap(), modified);
        let changed = digest(none, true);
        assert_ne!(changed.0, first.0);
        assert_eq!((changed.1, changed.2), (1, 0));
        // a damaged memo is hashing, not an error
        std::fs::write(folder.join(MEMO), b"{\"format\":\"autocut.sourceDigests\",\"version\":1,\"entries\":{\"x\":7}}").unwrap();
        assert_eq!(digest(none, true), (changed.0.clone(), 1, 0));
        assert_eq!(digest(none, true), (changed.0, 0, 1));
    }
    /// A writer that opened the lock file before its holder removed it must not go on with a
    /// file nobody else can reach: two writers would publish the same key at once.
    #[cfg(unix)]
    #[test]
    fn a_lock_file_removed_by_its_holder_is_not_a_lock() {
        let dir = crate::media_test_util::tmp_dir("cache-lock-identity");
        let path = dir.join("key.lock");
        let holder = lock_entry(&path, "test").unwrap();
        let late = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        assert!(lock_entry(&path, "test").is_err(), "held");
        unlock_entry(holder, &path);
        assert!(!path.exists());
        late.try_lock().unwrap();
        assert!(!is_at(&late, &path), "the removed file is locked, the path is free");
        let next = lock_entry(&path, "test").unwrap();
        assert!(is_at(&next, &path));
        unlock_entry(next, &path);
    }
    /// Every committed key left its `<key>.lock` file in the cache folder for good.
    #[cfg(unix)]
    #[test]
    fn a_commit_leaves_no_lock_file_behind() {
        let (mut s, dir) = session("cache-lock-files");
        let source = dir.join("source");
        std::fs::write(&source, b"source").unwrap();
        let p = json!({"dependencies":[source]});
        let plan = s.execute("render.cachePlan", p.clone()).unwrap();
        let artifact = dir.join("render.bin");
        std::fs::write(&artifact, b"complete artifact").unwrap();
        let commit = json!({"key":plan["key"],"path":artifact,"manifest":plan});
        assert_eq!(s.execute("render.cacheCommit", commit.clone()).unwrap()["hit"], false);
        assert_eq!(s.execute("render.cacheCommit", commit).unwrap()["hit"], true, "a second writer finds the entry");
        let locks: Vec<String> = std::fs::read_dir(super::dir(&s).unwrap())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|name| name.ends_with(".lock"))
            .collect();
        assert!(locks.is_empty(), "{locks:?}");
        assert_eq!(s.execute("render.cachePlan", p).unwrap()["hit"], true);
    }
    #[test]
    fn embedded_lut_content_invalidates_without_changing_a_clip_reference() {
        let (mut s, _) = session("cache-project-lut");
        s.execute("file.openDemoProject", json!({})).unwrap();
        let p = json!({"start":0,"end":Tick::from_seconds_f64(1.0).0});
        s.edit("Add library LUT", |project, _| {
            project.luts.push(filmcraft_project::ProjectLut {
                id: "grade".into(),
                name: "Grade".into(),
                source_path: None,
                format: "cube".into(),
                text: "LUT_1D_SIZE 2\n0 0 0\n1 1 1\n".into(),
            });
            Ok(())
        })
        .unwrap();
        let before = s.execute("render.cachePlan", p.clone()).unwrap();
        let items = s.project.items.clone();
        s.edit("Replace LUT text", |project, _| {
            project.luts[0].text = "LUT_1D_SIZE 2\n0 0 0\n0.9 1 1\n".into();
            Ok(())
        })
        .unwrap();
        assert_eq!(items, s.project.items);
        assert_ne!(before["key"], s.execute("render.cachePlan", p).unwrap()["key"]);
    }
    #[test]
    fn transition_handles_and_audio_outside_the_chapter_remain_dependencies() {
        let (mut s, _) = session("cache-handles");
        s.execute("file.openDemoProject", json!({})).unwrap();
        let q = s.active_sequence().unwrap();
        let first = q.video_tracks[0].items[0].clone();
        let next = q.video_tracks[0].items[1].id;
        let audio = q.audio_tracks.iter().flat_map(|t| &t.items).last().unwrap().id;
        s.edit_sequence("Crossfade handle", |q, ctx, _| {
            let transition = filmcraft_project::Transition {
                id: filmcraft_project::TransitionId(ctx.alloc()),
                effect: filmcraft_project::vtransition::find_transition("cross_dissolve", filmcraft_project::EffectKind::VideoTransition).unwrap().instance(),
                start: Tick::ZERO,
                duration: first.duration,
                from: Some(first.id),
                to: Some(next),
                align: Default::default(),
                reverse: false,
                sound: None,
            };
            q.video_tracks[0].transitions.push(transition);
            Ok(())
        })
        .unwrap();
        let p = json!({"start":0,"end":first.end().0.min(Tick::from_seconds_f64(1.0).0)});
        let initial = s.execute("render.cachePlan", p.clone()).unwrap();
        s.edit_sequence("Incoming handle", |q, _, _| {
            q.find_item_mut(next).unwrap().1.gain_db = -3.0;
            Ok(())
        })
        .unwrap();
        let changed = s.execute("render.cachePlan", p.clone()).unwrap();
        assert_ne!(initial["key"], changed["key"]);
        s.edit_sequence("Audio history", |q, _, _| {
            q.find_item_mut(audio).unwrap().1.gain_db = -4.0;
            Ok(())
        })
        .unwrap();
        assert_ne!(changed["key"], s.execute("render.cachePlan", p).unwrap()["key"]);
    }
}
