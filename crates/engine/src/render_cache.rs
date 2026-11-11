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
            r#"{"sequence":id?,"start":ticks?,"end":ticks?,"settings":{…},"external":JSON?,"dependencies":[path]?}"#,
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
            selected.push(value);
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
    let mut hashes = BTreeMap::new();
    for path in sources {
        hashes.insert(path.clone(), digest_file(Path::new(&path)).map_err(|e| bad(cmd, e))?);
    }
    // These resources render outside ProjectItem: a library LUT can change pixels
    // without changing the clip's `lib:<id>` reference. Keep them conservatively.
    let resources = (!selected.is_empty()).then(|| {
        json!({"settings":s.project.settings,"luts":s.project.luts,
        "sourceGraphics":s.project.source_graphics,"motionTracks":s.project.motion_tracks})
    });
    let identity = json!({"format":FORMAT,"version":1,"engine":ENGINE,"range":range,"items":selected,"resources":resources,"sources":hashes,"settings":p.get("settings"),"external":p.get("external")});
    if serde_json::to_vec(&identity).map_err(|e| bad(cmd, e.to_string()))?.len() as u64 > MAX_JSON / 2 {
        return Err(bad(cmd, "cache identity exceeds 8 MiB; split into smaller chapters"));
    }
    let key = digest(&identity)?;
    let hit = lookup(s, &key).map_err(|e| bad(cmd, e))?;
    Ok(json!({"key":key,"identity":identity,"hit":hit.is_some(),"artifact":hit,"cacheAvailable":dir(s).is_some()}))
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
        for (path, expected) in sources {
            let (hash, bytes) = digest_file(Path::new(path)).map_err(|e| bad(cmd, e))?;
            if expected != &json!([hash, bytes]) {
                return Err(bad(cmd, "source changed during render; cache publication refused"));
            }
        }
    }
    let path = Path::new(p["path"].as_str().ok_or_else(|| bad(cmd, "provide a completed render path"))?);
    let (hash, bytes) = digest_file(path).map_err(|e| bad(cmd, e))?;
    if bytes == 0 {
        return Err(bad(cmd, "empty render cannot be cached"));
    }
    let cache = dir(s).ok_or_else(|| bad(cmd, "configure a native cache directory first"))?;
    std::fs::create_dir_all(&cache).map_err(|e| bad(cmd, e.to_string()))?;
    let lock =
        std::fs::OpenOptions::new().create(true).truncate(false).write(true).open(cache.join(format!("{k}.lock"))).map_err(|e| bad(cmd, e.to_string()))?;
    lock.try_lock().map_err(|e| bad(cmd, format!("another writer owns this cache entry: {e}")))?;
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
        if digest_file(&staged.join("artifact")).map_err(|e| bad(cmd, e))? != (hash.clone(), bytes) {
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
