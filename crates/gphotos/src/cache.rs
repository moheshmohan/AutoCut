//! The download cache: picked originals land under
//! `<data dir>/google-photos/cache/<media item id>/<filename>`, so picking the same item again
//! (or `photos add <id>`) imports it without a second download.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_json::{Value, json};

use crate::api::{MediaKind, PickedItem};
use crate::{GPhotosError, GooglePhotos};

/// The largest original we accept (Google's own upload cap is 10 GB for video).
const MAX_DOWNLOAD: u64 = 20 << 30;
/// A long video over a slow line still finishes; a stalled one does not hang forever.
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(4 * 3600);
/// Caps on directory walks, so a cache someone filled with junk stays cheap to read.
const MAX_CACHE_ENTRIES: usize = 10_000;
const MAX_NAME_CHARS: usize = 120;

pub fn cache_dir() -> PathBuf {
    crate::data_dir().join("cache")
}

/// Where a picked item's original lives locally.
pub fn cached_path(item: &PickedItem) -> Option<PathBuf> {
    let id = safe_id(&item.id)?;
    Some(cache_dir().join(id).join(safe_filename(&item.filename, item.kind)))
}

/// A cached original by media item id (from an earlier pick), if there is one.
pub fn find(id: &str) -> Option<PathBuf> {
    find_in(&cache_dir(), id)
}

fn find_in(root: &Path, id: &str) -> Option<PathBuf> {
    let dir = root.join(safe_id(id)?);
    std::fs::read_dir(dir).ok()?.flatten().take(64).map(|e| e.path()).find(|p| p.is_file() && is_complete(p))
}

/// Everything in the cache: `{id, filename, bytes, path}`.
pub fn list() -> Vec<Value> {
    list_in(&cache_dir())
}

fn list_in(root: &Path) -> Vec<Value> {
    let Ok(dirs) = std::fs::read_dir(root) else { return Vec::new() };
    let mut out: Vec<Value> = dirs
        .flatten()
        .take(MAX_CACHE_ENTRIES)
        .filter_map(|d| {
            let id = d.file_name().to_string_lossy().into_owned();
            let path = find_in(root, &id)?;
            let bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            let filename = path.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
            Some(json!({"id": id, "filename": filename, "bytes": bytes, "path": path.to_string_lossy(), "created": sidecar_created(&path)}))
        })
        .collect();
    out.sort_by(|a, b| a.get("filename").and_then(Value::as_str).cmp(&b.get("filename").and_then(Value::as_str)));
    out
}

/// A half-written download is never an original, and neither is a metadata sidecar.
fn is_complete(p: &Path) -> bool {
    p.file_name().is_some_and(|n| {
        let n = n.to_string_lossy();
        !n.ends_with(".part") && !n.starts_with('.') && !n.ends_with(".json")
    })
}

/// The metadata sidecar of a cached original: `<file>.json`, the Google Takeout convention.
pub fn sidecar_path(media: &Path) -> PathBuf {
    let mut s = media.as_os_str().to_os_string();
    s.push(".json");
    PathBuf::from(s)
}

/// Write what Google knows about `item` next to its cached original. Google's video downloads
/// are re-encoded without the camera's metadata, so this is the only record of when it was
/// recorded (`media.captureDate` reads it). Best effort: a failure only loses the date.
pub fn write_sidecar(item: &PickedItem, media: &Path) -> bool {
    let v = json!({
        "title": item.filename,
        "created": item.created,
        "googlePhotosId": item.id,
        "mimeType": item.mime_type,
    });
    serde_json::to_vec_pretty(&v).is_ok_and(|bytes| std::fs::write(sidecar_path(media), bytes).is_ok())
}

/// The `created` time recorded in a cached original's sidecar.
pub fn sidecar_created(media: &Path) -> Option<String> {
    let path = sidecar_path(media);
    if std::fs::metadata(&path).ok()?.len() > 1 << 20 {
        return None;
    }
    let v: Value = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    v.get("created").and_then(Value::as_str).map(str::to_string)
}

/// Media item ids are URL-safe base64; anything else is replaced so an id can never name a
/// path outside the cache.
pub fn safe_id(id: &str) -> Option<String> {
    let s: String = id.chars().take(256).map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    (!s.is_empty() && s.chars().any(|c| c != '_')).then_some(s)
}

/// A filename from Google, made safe to create: no separators, control characters or leading
/// dots, a bounded length (the extension kept), never empty, never a `.part`.
pub fn safe_filename(name: &str, kind: MediaKind) -> String {
    let cleaned: String = name.chars().map(|c| if c.is_alphanumeric() || matches!(c, '.' | '-' | '_' | ' ' | '(' | ')' | '+') { c } else { '_' }).collect();
    let cleaned = cleaned.trim().trim_start_matches('.').to_string();
    let mut out = if cleaned.chars().count() > MAX_NAME_CHARS {
        let (stem, ext) = match cleaned.rsplit_once('.') {
            Some((s, e)) if e.chars().count() <= 10 => (s.to_string(), format!(".{e}")),
            _ => (cleaned.clone(), String::new()),
        };
        let keep = MAX_NAME_CHARS.saturating_sub(ext.chars().count());
        format!("{}{ext}", stem.chars().take(keep).collect::<String>())
    } else {
        cleaned
    };
    if out.is_empty() || out.chars().all(|c| c == '_' || c == '.') {
        out = match kind {
            MediaKind::Video => "video.mp4".into(),
            _ => "photo.jpg".into(),
        };
    }
    if out.ends_with(".part") {
        out.push('_');
    }
    out
}

/// The bearer token rides along with every download, so it only ever goes to Google's own
/// media hosts over https.
pub fn is_google_media_url(url: &str) -> bool {
    let Some(rest) = url.strip_prefix("https://") else { return false };
    let authority = rest.split(['/', '?', '#']).next().unwrap_or_default();
    if authority.contains('@') {
        return false;
    }
    let host = authority.split(':').next().unwrap_or_default().to_ascii_lowercase();
    ["googleusercontent.com", "google.com", "googleapis.com"].iter().any(|d| host.strip_suffix(d).is_some_and(|sub| sub.ends_with('.')))
}

impl GooglePhotos {
    /// A picked item's local original, downloading it once: photos as originals (`=d`, Exif
    /// minus location), videos as Google's high-quality file (`=dv`). The download streams to
    /// a `.part` file that is renamed into place only when complete.
    pub fn local_copy(&self, item: &PickedItem) -> Result<PathBuf, GPhotosError> {
        let Some(dest) = cached_path(item) else {
            return Err(GPhotosError::Download("the picked item has no usable id".into()));
        };
        if dest.is_file() {
            return Ok(dest);
        }
        if let Some(found) = find(&item.id) {
            return Ok(found);
        }
        let name = item.display_name();
        if item.kind == MediaKind::Video {
            match item.video_status.as_deref() {
                Some("FAILED") => return Err(GPhotosError::Download(format!("`{name}`: Google Photos failed to process this video"))),
                Some("PROCESSING") => return Err(GPhotosError::Download(format!("`{name}` is still processing in Google Photos; pick it again later"))),
                _ => {}
            }
        }
        if !is_google_media_url(&item.base_url) {
            return Err(GPhotosError::Download(format!("`{name}` has no Google download link")));
        }
        let url = format!("{}={}", item.base_url, if item.kind == MediaKind::Video { "dv" } else { "d" });
        let auth = format!("Bearer {}", self.access_token()?);
        let resp = self
            .agent()
            .get(&url)
            .header("Authorization", &auth)
            .config()
            .timeout_global(Some(DOWNLOAD_TIMEOUT))
            .build()
            .call()
            .map_err(|e| GPhotosError::Download(format!("downloading `{name}` failed: {e}")))?;
        let status = resp.status().as_u16();
        if status != 200 {
            let hint = if status == 403 || status == 401 { " (download links last 60 minutes; pick it again)" } else { "" };
            return Err(GPhotosError::Download(format!("downloading `{name}`: Google answered {status}{hint}")));
        }
        let dir = dest.parent().ok_or_else(|| GPhotosError::Download("bad cache path".into()))?;
        std::fs::create_dir_all(dir)?;
        let part = dir.join(format!(".{}.part", dest.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()));
        let written = (|| -> Result<u64, GPhotosError> {
            let mut file = std::fs::File::create(&part)?;
            let n = std::io::copy(&mut resp.into_body().into_reader().take(MAX_DOWNLOAD + 1), &mut file)?;
            file.sync_all()?;
            Ok(n)
        })();
        // a failed or oversized download leaves nothing behind
        let fail = |msg: String| {
            let _ = std::fs::remove_file(&part);
            GPhotosError::Download(msg)
        };
        match written {
            Err(e) => Err(fail(format!("saving `{name}`: {e}"))),
            Ok(0) => Err(fail(format!("`{name}` downloaded empty"))),
            Ok(n) if n > MAX_DOWNLOAD => Err(fail(format!("`{name}` is larger than {} GiB", MAX_DOWNLOAD >> 30))),
            Ok(_) => {
                std::fs::rename(&part, &dest).map_err(|e| fail(format!("saving `{name}`: {e}")))?;
                Ok(dest)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sidecars_record_the_date_and_are_never_the_original() {
        let root = std::env::temp_dir().join(format!("filmcraft-gp-sidecar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(root.join("idS")).unwrap();
        let media = root.join("idS/clip.mov");
        std::fs::write(&media, b"video").unwrap();
        let item = PickedItem {
            id: "idS".into(),
            filename: "clip.mov".into(),
            mime_type: "video/quicktime".into(),
            kind: MediaKind::Video,
            created: "2022-10-14T10:01:08Z".into(),
            base_url: String::new(),
            video_status: None,
        };
        assert!(write_sidecar(&item, &media));
        assert_eq!(sidecar_created(&media).as_deref(), Some("2022-10-14T10:01:08Z"));
        // the sidecar sorts before nothing and is never picked as the original
        for _ in 0..3 {
            assert_eq!(find_in(&root, "idS"), Some(media.clone()));
        }
        let listed = list_in(&root);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0]["created"], "2022-10-14T10:01:08Z");
        // a damaged sidecar is no date, not an error
        std::fs::write(sidecar_path(&media), b"{nope").unwrap();
        assert_eq!(sidecar_created(&media), None);
    }

    use super::*;

    fn item(id: &str, filename: &str, kind: MediaKind) -> PickedItem {
        PickedItem {
            id: id.into(),
            filename: filename.into(),
            mime_type: String::new(),
            kind,
            created: String::new(),
            base_url: String::new(),
            video_status: None,
        }
    }

    #[test]
    fn cache_paths_are_keyed_and_readable() {
        let p = cached_path(&item("AF1Qip-AbC_123", "sunset.jpg", MediaKind::Photo)).unwrap();
        assert!(p.ends_with("google-photos/cache/AF1Qip-AbC_123/sunset.jpg"), "{p:?}");
        assert!(cached_path(&item("", "x.jpg", MediaKind::Photo)).is_none());
        assert!(cached_path(&item("../..", "x.jpg", MediaKind::Photo)).is_none(), "nothing but separators is no id");
    }

    #[test]
    fn hostile_names_stay_inside_the_cache() {
        assert_eq!(safe_id("../../etc"), Some("______etc".into()));
        assert_eq!(safe_id("a/b\\c"), Some("a_b_c".into()));
        assert_eq!(safe_id("/"), None);
        assert_eq!(safe_id(&"x".repeat(1000)).map(|s| s.len()), Some(256));
        assert_eq!(safe_filename("../../.ssh/authorized_keys", MediaKind::Photo), "_.._.ssh_authorized_keys");
        assert_eq!(safe_filename(".hidden.jpg", MediaKind::Photo), "hidden.jpg");
        assert_eq!(safe_filename("a\0b\nc:d.mov", MediaKind::Video), "a_b_c_d.mov");
        assert_eq!(safe_filename("", MediaKind::Video), "video.mp4");
        assert_eq!(safe_filename("///", MediaKind::Photo), "photo.jpg");
        assert_eq!(safe_filename("clip.part", MediaKind::Video), "clip.part_");
        assert_eq!(safe_filename("東京タワー.HEIC", MediaKind::Photo), "東京タワー.HEIC", "non-ASCII letters survive");
        let long = safe_filename(&format!("{}.mp4", "y".repeat(500)), MediaKind::Video);
        assert!(long.ends_with(".mp4") && long.chars().count() == MAX_NAME_CHARS, "{long}");
        assert_eq!(safe_filename(&"z".repeat(500), MediaKind::Photo).chars().count(), MAX_NAME_CHARS);
    }

    #[test]
    fn the_token_only_goes_to_google_media_hosts() {
        assert!(is_google_media_url("https://lh3.googleusercontent.com/ppa/AbC"));
        assert!(is_google_media_url("https://video-downloads.googleusercontent.com/x?y=1"));
        assert!(is_google_media_url("https://LH3.GoogleUserContent.com:443/x"));
        for bad in [
            "http://lh3.googleusercontent.com/x",
            "https://evil.com/lh3.googleusercontent.com",
            "https://googleusercontent.com.evil.com/x",
            "https://evilgoogleusercontent.com/x",
            "https://lh3.googleusercontent.com@evil.com/x",
            "https://googleusercontent.com/x",
            "",
            "lh3.googleusercontent.com",
        ] {
            assert!(!is_google_media_url(bad), "{bad}");
        }
    }

    #[test]
    fn finds_and_lists_complete_downloads_only() {
        let root = std::env::temp_dir().join(format!("gphotos-cache-{}", std::process::id()));
        std::fs::create_dir_all(root.join("idA")).unwrap();
        std::fs::create_dir_all(root.join("idB")).unwrap();
        std::fs::write(root.join("idA/beach.mov"), b"mov").unwrap();
        std::fs::write(root.join("idB/.cat.jpg.part"), b"half").unwrap();
        assert_eq!(find_in(&root, "idA"), Some(root.join("idA/beach.mov")));
        assert_eq!(find_in(&root, "idB"), None, "a partial download is not cached");
        assert_eq!(find_in(&root, "../idA"), None);
        assert_eq!(find_in(&root, "missing"), None);
        let listed = list_in(&root);
        assert_eq!(listed.len(), 1);
        assert_eq!(listed.first().and_then(|v| v.get("bytes")).and_then(Value::as_u64), Some(3));
        assert!(list_in(&root.join("nope")).is_empty());
        std::fs::remove_dir_all(&root).ok();
    }
}
