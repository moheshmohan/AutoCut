//! When media was recorded, read from the file: for sorting clips chronologically.
//!
//! | command | does |
//! |---|---|
//! | `media.captureDate` | the capture date of media items (all, or `items`): a metadata sidecar (`<file>.json` from the Google Photos picker or Google Takeout), else QuickTime `com.apple.quicktime.creationdate` / `©day` (with the camera's time zone), else the `mvhd` creation time — as RFC 3339 UTC plus Unix seconds |
//!
//! Sidecars are read for any file; embedded metadata only for MP4 / MOV / M4V / 3GP (other
//! containers report `null`). Google's video downloads are re-encoded without the camera's
//! metadata, which is why the picker writes the sidecar. Nothing is stored in the project: the
//! files are the record, so a re-linked file answers for itself.

use serde_json::{Value, json};

use filmcraft_project::{ItemId, ItemKind, MediaRef};

use crate::commands::{CommandSpec, always, bad};
use crate::{Result, Session};

/// Seconds from 1904-01-01 (the MP4 epoch) to 1970-01-01.
#[cfg(not(target_arch = "wasm32"))]
const MP4_EPOCH_OFFSET: i64 = 2_082_844_800;
/// Dates before 1990 are cameras with an unset clock, not recordings.
#[cfg(not(target_arch = "wasm32"))]
const EARLIEST: i64 = 631_152_000;

pub fn commands() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: "media.captureDate",
        label: "Media Capture Date",
        menu: &[],
        shortcut: None,
        params: r#"{"items":[id]?}"#,
        enabled: always,
        run: capture_dates,
        journal: false,
    }]
}

/// Days since 1970-01-01 for a civil date (proleptic Gregorian).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// The civil date of a day count since 1970-01-01.
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// RFC 3339 UTC of Unix seconds.
pub fn rfc3339(unix: i64) -> String {
    let days = unix.div_euclid(86_400);
    let secs = unix.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z", secs / 3600, secs % 3600 / 60, secs % 60)
}

/// Unix seconds of an ISO 8601 date-time as cameras write it: `2024-05-17T13:30:00+0400`,
/// `…+04:00`, `…Z`, or without a zone (taken as UTC); `None` for anything else.
pub fn parse_iso(s: &str) -> Option<i64> {
    let s = s.trim();
    let num = |a: usize, b: usize| -> Option<i64> { s.get(a..b).filter(|t| t.bytes().all(|c| c.is_ascii_digit())).and_then(|t| t.parse().ok()) };
    let (y, mo, d) = (num(0, 4)?, num(5, 7)?, num(8, 10)?);
    if s.get(4..5) != Some("-") || s.get(7..8) != Some("-") || !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let (h, mi, se) = if s.len() >= 19 && matches!(s.get(10..11), Some("T") | Some(" ")) { (num(11, 13)?, num(14, 16)?, num(17, 19)?) } else { (0, 0, 0) };
    if h > 23 || mi > 59 || se > 60 {
        return None;
    }
    // zone: after the seconds (skipping fractional seconds)
    let mut rest = s.get(19.min(s.len())..).unwrap_or("");
    if let Some(r) = rest.strip_prefix('.') {
        rest = r.trim_start_matches(|c: char| c.is_ascii_digit());
    }
    let offset = match rest.chars().next() {
        None | Some('Z') | Some('z') => 0,
        Some(sign @ ('+' | '-')) => {
            let digits: String = rest.chars().skip(1).filter(char::is_ascii_digit).take(4).collect();
            if digits.len() != 4 {
                return None;
            }
            let hh: i64 = digits.get(0..2)?.parse().ok()?;
            let mm: i64 = digits.get(2..4)?.parse().ok()?;
            let o = hh * 3600 + mm * 60;
            if sign == '-' { -o } else { o }
        }
        Some(_) => return None,
    };
    Some(days_from_civil(y, mo, d) * 86_400 + h * 3600 + mi * 60 + se - offset)
}

/// A file as the MP4 demuxer's byte source (portable: seek + read under a lock).
#[cfg(not(target_arch = "wasm32"))]
struct FileSrc {
    f: std::sync::Mutex<std::fs::File>,
    len: u64,
}

#[cfg(not(target_arch = "wasm32"))]
impl filmcraft_isobmff::ByteSource for FileSrc {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        use std::io::{Read, Seek, SeekFrom};
        let mut f = self.f.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        f.seek(SeekFrom::Start(offset))?;
        f.read_exact(buf)
    }
}

/// The date in a metadata sidecar next to `path`: `<file>.json` (what the Google Photos picker
/// writes, and Google Takeout's naming) or `<file>.supplemental-metadata.json` (newer Takeout).
/// Ours carries `created` (RFC 3339); Takeout carries `photoTakenTime.timestamp` (Unix seconds).
#[cfg(not(target_arch = "wasm32"))]
pub fn sidecar_date(path: &str) -> Option<i64> {
    for side in [format!("{path}.json"), format!("{path}.supplemental-metadata.json")] {
        let Ok(meta) = std::fs::metadata(&side) else { continue };
        if meta.len() > 1 << 20 {
            continue;
        }
        let Some(v) = std::fs::read(&side).ok().and_then(|b| serde_json::from_slice::<Value>(&b).ok()) else { continue };
        let created = v.get("created").and_then(Value::as_str).and_then(parse_iso);
        let taken = v.pointer("/photoTakenTime/timestamp").and_then(|t| t.as_str().and_then(|s| s.trim().parse::<i64>().ok()).or_else(|| t.as_i64()));
        if let Some(t) = created.or(taken).filter(|t| *t >= EARLIEST) {
            return Some(t);
        }
    }
    None
}

/// `(unix seconds, where it came from)`: a metadata sidecar, else (MP4 family) the file's own
/// metadata.
#[cfg(not(target_arch = "wasm32"))]
pub fn file_capture_date(path: &str) -> Option<(i64, &'static str)> {
    if let Some(t) = sidecar_date(path) {
        return Some((t, "sidecar"));
    }
    let ext = std::path::Path::new(path).extension()?.to_str()?.to_ascii_lowercase();
    if !matches!(ext.as_str(), "mp4" | "mov" | "m4v" | "3gp" | "qt") {
        return None;
    }
    let f = std::fs::File::open(path).ok()?;
    let len = f.metadata().ok()?.len();
    let file = filmcraft_isobmff::open(FileSrc { f: std::sync::Mutex::new(f), len }).ok()?;
    if let Some(t) = file.metadata.get("com.apple.quicktime.creationdate").and_then(parse_iso).filter(|t| *t >= EARLIEST) {
        return Some((t, "quicktime"));
    }
    if let Some(t) = file.metadata.get("©day").and_then(parse_iso).filter(|t| *t >= EARLIEST) {
        return Some((t, "©day"));
    }
    let t = i64::try_from(file.creation_time?).ok()?.checked_sub(MP4_EPOCH_OFFSET)?;
    (t >= EARLIEST).then_some((t, "mvhd"))
}

#[cfg(target_arch = "wasm32")]
pub fn file_capture_date(_path: &str) -> Option<(i64, &'static str)> {
    None
}

fn capture_dates(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "media.captureDate";
    let ids: Vec<ItemId> = match p.get("items") {
        Some(v) => v
            .as_array()
            .ok_or_else(|| bad(cmd, "`items` must be a list of item ids"))?
            .iter()
            .map(|x| x.as_u64().map(ItemId).ok_or_else(|| bad(cmd, "`items` must be a list of item ids")))
            .collect::<Result<_>>()?,
        None => {
            let mut v: Vec<ItemId> = s.project.items.iter().filter(|(_, i)| matches!(i.kind, ItemKind::Media(_))).map(|(id, _)| *id).collect();
            v.sort();
            v
        }
    };
    let mut out = Vec::with_capacity(ids.len());
    for id in ids {
        let it = s.project.item(id).ok_or_else(|| bad(cmd, format!("no item {}", id.0)))?;
        let path = match &it.kind {
            ItemKind::Media(m) => match &m.media {
                MediaRef::File { path } => Some(path.clone()),
                MediaRef::Generator(_) => None,
            },
            _ => None,
        };
        let found = path.as_deref().and_then(file_capture_date);
        out.push(json!({
            "item": id.0,
            "name": it.name,
            "created": found.map(|(t, _)| rfc3339(t)),
            "unix": found.map(|(t, _)| t),
            "source": found.map(|(_, src)| src),
        }));
    }
    Ok(json!({"items": out}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dates_round_trip() {
        for (y, m, d) in [(1970, 1, 1), (2000, 2, 29), (2020, 3, 20), (2024, 12, 31), (1904, 1, 1)] {
            assert_eq!(civil_from_days(days_from_civil(y, m, d)), (y, m, d));
        }
        assert_eq!(days_from_civil(1970, 1, 1), 0);
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_715_938_200), "2024-05-17T09:30:00Z");
        assert_eq!(-days_from_civil(1904, 1, 1) * 86_400, MP4_EPOCH_OFFSET);
    }

    #[test]
    fn camera_dates_parse_with_zones() {
        let utc = parse_iso("2024-05-17T09:30:00Z").unwrap();
        assert_eq!(parse_iso("2024-05-17T13:30:00+0400"), Some(utc));
        assert_eq!(parse_iso("2024-05-17T13:30:00+04:00"), Some(utc));
        assert_eq!(parse_iso("2024-05-17T05:30:00-04:00"), Some(utc));
        assert_eq!(parse_iso("2024-05-17T09:30:00.123Z"), Some(utc));
        assert_eq!(parse_iso("2024-05-17 09:30:00"), Some(utc));
        assert_eq!(parse_iso("2024-05-17"), Some(utc - 9 * 3600 - 30 * 60));
        for bad in ["", "2024", "2024-13-01T00:00:00Z", "2024-05-17T25:00:00Z", "yesterday", "2024-05-17T09:30:00+4", "2024/05/17"] {
            assert_eq!(parse_iso(bad), None, "{bad}");
        }
    }

    #[test]
    fn sidecars_are_read_first_ours_and_takeouts() {
        let d = std::env::temp_dir().join(format!("filmcraft-sidecar-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&d);
        let a = d.join("a.mov").to_string_lossy().into_owned();
        std::fs::write(&a, b"not really a movie").unwrap();
        std::fs::write(format!("{a}.json"), br#"{"created": "2022-10-14T10:01:08Z"}"#).unwrap();
        assert_eq!(file_capture_date(&a), Some((parse_iso("2022-10-14T10:01:08Z").unwrap(), "sidecar")));
        let b = d.join("b.jpg").to_string_lossy().into_owned();
        std::fs::write(format!("{b}.supplemental-metadata.json"), br#"{"photoTakenTime": {"timestamp": "1665741668"}}"#).unwrap();
        assert_eq!(sidecar_date(&b), Some(1_665_741_668));
        // damaged or too-early dates are ignored
        let c = d.join("c.mp4").to_string_lossy().into_owned();
        std::fs::write(format!("{c}.json"), b"{broken").unwrap();
        assert_eq!(sidecar_date(&c), None);
        std::fs::write(format!("{c}.json"), br#"{"created": "1970-01-01T00:00:00Z"}"#).unwrap();
        assert_eq!(sidecar_date(&c), None);
    }

    #[test]
    fn command_reports_items_without_dates() {
        let mut s = Session::default();
        s.execute("file.openDemoProject", json!({})).unwrap();
        let r = s.execute("media.captureDate", json!({})).unwrap();
        let items = r["items"].as_array().unwrap();
        assert!(!items.is_empty());
        assert!(items.iter().all(|i| i["created"].is_null() || i["created"].is_string()));
        assert!(s.execute("media.captureDate", json!({"items": [999_999]})).is_err());
        assert!(s.execute("media.captureDate", json!({"items": "x"})).is_err());
    }
}
