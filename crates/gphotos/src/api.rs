//! The Google Photos Library API client: albums, media search, item lookup.
//!
//! Google's `mediaItems.search` filters by content category, date range and album — there is
//! no free-text semantic search server-side, so a local filename filter rides on top (an
//! agent composes the filters; filenames often carry the words people remember).

use std::io::Read;
use std::time::Duration;

use serde_json::{Value, json};

use crate::GooglePhotos;

const API: &str = "https://photoslibrary.googleapis.com/v1";

/// The content categories Google recognizes for `search`.
pub const CATEGORIES: &[&str] = &[
    "NONE",
    "PEOPLE",
    "SELFIES",
    "ANIMALS",
    "FOOD",
    "SPORT",
    "LANDSCAPES",
    "CITYSCAPES",
    "RECEIPTS",
    "SCREENSHOTS",
    "UTILITY",
    "ARTS",
    "CRAFTS",
    "FASHION",
    "HOME",
    "GARDENS",
    "FLOWERS",
    "HOLIDAYS",
    "NIGHT",
    "PERFORMANCES",
    "WHITEBOARDS",
    "INFographics",
    "PRODUCTS",
    "WEDDINGS",
    "BIRTHDAYS",
    "DOCUMENTS",
    "TRAVEL",
];

/// Search parameters; all optional.
#[derive(Debug, Default, Clone)]
pub struct Search {
    /// Local, case-insensitive substring match against filenames (words people remember).
    pub query: Option<String>,
    /// A Google content category (see [`CATEGORIES`]).
    pub category: Option<String>,
    /// Inclusive, `YYYY-MM-DD`.
    pub after: Option<String>,
    /// Exclusive, `YYYY-MM-DD`.
    pub before: Option<String>,
    /// Search within one album id (from [`GooglePhotos::albums`]).
    pub album: Option<String>,
    /// Stop after this many results (default 50, capped at 200).
    pub limit: usize,
}

impl GooglePhotos {
    /// The user's albums: id, title, item count.
    pub fn albums(&self) -> Result<Vec<Value>, crate::GPhotosError> {
        let mut out = Vec::new();
        let mut token: Option<String> = None;
        for _ in 0..5 {
            let mut url = format!("{API}/albums?pageSize=50");
            if let Some(t) = &token {
                url.push_str(&format!("&pageToken={}", crate::auth::urlencode(t)));
            }
            let v = self.get_json(&url)?;
            for a in v["albums"].as_array().into_iter().flatten() {
                out.push(json!({
                    "id": a["id"], "title": a["title"].as_str().unwrap_or("Untitled"),
                    "items": a["mediaItemsCount"].as_str().unwrap_or("?"),
                }));
            }
            token = v["nextPageToken"].as_str().map(str::to_string);
            if token.is_none() {
                break;
            }
        }
        Ok(out)
    }

    /// One media item by id (the shape [`Self::search`] returns per item).
    pub fn item(&self, id: &str) -> Result<Value, crate::GPhotosError> {
        self.get_json(&format!("{API}/mediaItems/{id}"))
    }

    /// Search the library. Each result: `{id, filename, mimeType, kind, created, album}` —
    /// `baseUrl` is left out deliberately; it is per-session and [`crate::cache`] fetches it
    /// fresh when a download is actually wanted.
    pub fn search(&self, s: &Search) -> Result<Vec<Value>, crate::GPhotosError> {
        let limit = if s.limit == 0 { 50 } else { s.limit.min(200) };
        let mut filters = json!({});
        if let Some(c) = valid_category(&s.category)? {
            filters["contentFilter"] = json!({"includedContentCategories": [c]});
        }
        if s.after.is_some() || s.before.is_some() {
            filters["dateFilter"] = json!({"ranges": [json!({
                "startDate": date(&s.after)?, "endDate": date_or(&s.before, "9999-12-31"),
            })]});
        }
        let mut body = json!({"pageSize": 100.min(limit.max(1) as u32 + 25)});
        if let Some(a) = &s.album {
            body["albumId"] = json!(a);
        } else if filters.as_object().is_some_and(|f| !f.is_empty()) {
            body["filters"] = filters;
        }
        let query_words: Vec<String> = s.query.as_deref().map(|q| q.split_whitespace().map(str::to_ascii_lowercase).collect()).unwrap_or_default();

        let mut out: Vec<Value> = Vec::new();
        let mut token: Option<String> = None;
        for _ in 0..4 {
            if let Some(t) = &token {
                body["pageToken"] = json!(t);
            } else if let Some(o) = body.as_object_mut() {
                o.remove("pageToken");
            }
            let v = self.post_json(&format!("{API}/mediaItems:search"), &body)?;
            for m in v["mediaItems"].as_array().into_iter().flatten() {
                let filename = m["filename"].as_str().unwrap_or_default();
                if !query_words.is_empty() && !matches_filename(filename, &query_words) {
                    continue;
                }
                let metadata = &m["mediaMetadata"];
                out.push(json!({
                    "id": m["id"], "filename": filename, "mimeType": m["mimeType"],
                    "kind": if metadata["photo"].is_object() { "photo" } else { "video" },
                    "created": metadata["creationTime"],
                }));
                if out.len() >= limit {
                    return Ok(out);
                }
            }
            token = v["nextPageToken"].as_str().map(str::to_string);
            if token.is_none() {
                break;
            }
        }
        Ok(out)
    }

    fn get_json(&self, url: &str) -> Result<Value, crate::GPhotosError> {
        let token = self.access_token()?;
        let resp = self.agent().get(url).header("Authorization", format!("Bearer {token}")).call().map_err(|e| crate::GPhotosError::Api(e.to_string()))?;
        Self::json_of(resp)
    }

    fn post_json(&self, url: &str, body: &Value) -> Result<Value, crate::GPhotosError> {
        let token = self.access_token()?;
        let resp =
            self.agent().post(url).header("Authorization", format!("Bearer {token}")).send_json(body).map_err(|e| crate::GPhotosError::Api(e.to_string()))?;
        Self::json_of(resp)
    }

    fn json_of(resp: ureq::http::Response<ureq::Body>) -> Result<Value, crate::GPhotosError> {
        let status = resp.status();
        let mut text = String::new();
        resp.into_body().into_reader().take(1 << 22).read_to_string(&mut text).map_err(|e| crate::GPhotosError::Api(e.to_string()))?;
        if status != 200 {
            return Err(crate::GPhotosError::Api(format!("Google answered {status}: {}", truncate(&text, 300))));
        }
        serde_json::from_str(&text).map_err(|e| crate::GPhotosError::Api(format!("Google sent invalid JSON: {e}")))
    }
}

/// Case-insensitive: every query word appears in the filename.
pub fn matches_filename(filename: &str, lowercase_words: &[String]) -> bool {
    let name = filename.to_ascii_lowercase();
    lowercase_words.iter().all(|w| name.contains(w))
}

fn valid_category(c: &Option<String>) -> Result<Option<String>, crate::GPhotosError> {
    let Some(c) = c else { return Ok(None) };
    let up = c.trim().to_ascii_uppercase();
    if CATEGORIES.contains(&up.as_str()) {
        Ok(Some(up))
    } else {
        Err(crate::GPhotosError::Api(format!("unknown category `{c}`; one of {}", CATEGORIES.join(", "))))
    }
}

fn date(d: &Option<String>) -> Result<Value, crate::GPhotosError> {
    let Some(d) = d else { return Ok(Value::Null) };
    let parts: Vec<&str> = d.split('-').collect();
    if parts.len() != 3 || parts.iter().any(|p| p.parse::<u32>().is_err()) {
        return Err(crate::GPhotosError::Api(format!("`{d}` is not YYYY-MM-DD")));
    }
    Ok(json!({"year": parts[0], "month": parts[1], "day": parts[2]}))
}

fn date_or(d: &Option<String>, fallback: &str) -> Value {
    d.as_deref().and_then(|d| date(&Some(d.to_string())).ok()).filter(|v| !v.is_null()).unwrap_or_else(|| json!(fallback))
}

/// The shared HTTP agent (pure-Rust TLS), as the other crates configure ureq.
pub fn http_agent() -> ureq::Agent {
    let provider = std::sync::Arc::new(rustls_rustcrypto::provider());
    ureq::Agent::config_builder()
        .tls_config(
            ureq::tls::TlsConfig::builder()
                .provider(ureq::tls::TlsProvider::Rustls)
                .root_certs(ureq::tls::RootCerts::PlatformVerifier)
                .unversioned_rustls_crypto_provider(provider)
                .build(),
        )
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_global(Some(Duration::from_secs(300)))
        .build()
        .into()
}

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filename_matching_is_all_words_case_insensitive() {
        let words = |q: &str| q.split_whitespace().map(str::to_ascii_lowercase).collect::<Vec<_>>();
        assert!(matches_filename("Beach_Sunset_2024.jpg", &words("beach sunset")));
        assert!(matches_filename("beach.mov", &words("BEACH")));
        assert!(!matches_filename("beach.jpg", &words("beach sunset")));
    }

    #[test]
    fn category_and_date_validation() {
        assert_eq!(valid_category(&Some("landscapes".into())).unwrap().as_deref(), Some("LANDSCAPES"));
        assert!(valid_category(&Some("dragons".into())).is_err());
        let d = date(&Some("2025-03-01".into())).unwrap();
        assert_eq!((&d["year"], &d["month"], &d["day"]), (&json!("2025"), &json!("03"), &json!("01")));
        assert!(date(&Some("03/01/2025".into())).is_err());
        assert_eq!(date_or(&None, "9999-12-31"), json!("9999-12-31"));
    }
}
