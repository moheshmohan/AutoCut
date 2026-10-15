//! The download cache: media originals land under `<data dir>/google-photos/cache/`, keyed by
//! media item id, so an import is instant the second time and nothing is fetched twice.

use std::io::Read;
use std::path::PathBuf;

use serde_json::Value;

use crate::GooglePhotos;

/// Where a media item's original lives locally (`<id>_<filename>` keeps it debuggable).
pub fn cached_path(media_item: &Value) -> Option<PathBuf> {
    let id = media_item["id"].as_str()?;
    let filename = media_item["filename"].as_str()?;
    Some(crate::data_dir().join("cache").join(format!("{id}_{filename}")))
}

impl GooglePhotos {
    /// A media item's local original, downloading it once. Photos come down whole (`=d`);
    /// videos stream (`=dv`). `BaseUrl` is per-session, so it is always fetched fresh.
    pub fn local_copy(&self, media_item: &Value) -> Result<PathBuf, crate::GPhotosError> {
        let Some(dest) = cached_path(media_item) else {
            return Err(crate::GPhotosError::Download("the item has no id/filename".into()));
        };
        if dest.is_file() {
            return Ok(dest);
        }
        let id = media_item["id"].as_str().unwrap_or_default().to_string();
        let fresh = self.item(&id)?;
        let base = fresh["baseUrl"].as_str().ok_or_else(|| crate::GPhotosError::Download(format!("`{id}` has no baseUrl (video not finished processing?)")))?;
        let is_video = fresh["mediaMetadata"]["video"].is_object();
        let url = format!("{}={}", base, if is_video { "dv" } else { "d" });
        let token = self.access_token()?;
        let resp = self
            .agent()
            .get(&url)
            .header("Authorization", format!("Bearer {token}"))
            .call()
            .map_err(|e| crate::GPhotosError::Download(format!("downloading `{}` failed: {e}", fresh["filename"].as_str().unwrap_or(&id))))?;
        if resp.status() != 200 {
            return Err(crate::GPhotosError::Download(format!("downloading `{id}` answered {}", resp.status())));
        }
        std::fs::create_dir_all(dest.parent().unwrap_or_else(|| std::path::Path::new(".")))?;
        let part = dest.with_extension("part");
        let reader = resp.into_body().into_reader();
        let mut bytes = Vec::new();
        reader.take(8 << 30).read_to_end(&mut bytes).map_err(crate::GPhotosError::from)?;
        std::fs::write(&part, &bytes).map_err(crate::GPhotosError::from)?;
        std::fs::rename(&part, &dest).map_err(crate::GPhotosError::from)?;
        Ok(dest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn cache_paths_are_keyed_and_readable() {
        let m = json!({"id": "AbC123", "filename": "sunset.jpg"});
        let p = cached_path(&m).unwrap();
        assert!(p.to_string_lossy().contains("google-photos/cache/AbC123_sunset.jpg"), "{p:?}");
        assert!(cached_path(&json!({"id": "x"})).is_none());
    }
}
