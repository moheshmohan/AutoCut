//! Insta360 recordings on disk: where Insta360 Studio puts the cloud files it downloads, and the
//! camera's own card.
//!
//! An Insta360 camera names its files `VID_<yyyymmdd>_<hhmmss>_<ll>_<nnn>.insv` (video), `IMG_…
//! .insp` (photos) and `LRV_…_01_<nnn>.lrv` (the low-resolution copy it records beside a video,
//! used here as its proxy): the date and time it started (camera time), the lens (`00` the file
//! of both lenses — X3 and later record both in one file — `10` the second lens of older two-file
//! cameras) and the camera's file counter. Studio's cloud download puts each one in a folder
//! `<stem>-Original/` under `~/Downloads/Insta360 Download` by default.
//!
//! A recording is [`Recording::complete`] when its file ends with the camera's trailer (see
//! `filmcraft_isobmff::insta360`): a download still being written does not have it yet.

use std::path::{Path, PathBuf};

use chrono::NaiveDateTime;
use serde::Serialize;

/// Most files looked at in one scan (a home folder is not a camera card).
pub const MAX_FILES: usize = 200_000;
/// How deep a scan goes below each folder.
pub const MAX_DEPTH: usize = 4;

/// What a file is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum Kind {
    Video,
    Photo,
    Proxy,
}

/// The parts of an Insta360 file name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Name {
    pub kind: Kind,
    /// `VID_20260913_114003_00_093` — the stem of the main file (a proxy's is its video's).
    pub stem: String,
    pub shot: NaiveDateTime,
    /// `00` both lenses (or the first), `10` the second lens of a two-file camera, `01` a proxy.
    pub lens: String,
    pub number: String,
}

/// Parses an Insta360 file name (`None` for any other file).
pub fn parse_name(file: &str) -> Option<Name> {
    let (base, ext) = file.rsplit_once('.')?;
    let ext = ext.to_ascii_lowercase();
    let mut parts = base.split('_');
    let prefix = parts.next()?;
    let date = parts.next()?;
    let time = parts.next()?;
    let lens = parts.next()?;
    let number = parts.next()?;
    // (Studio may add a suffix to a second copy: `…_093(1)`; anything after is not ours)
    if parts.next().is_some() || date.len() != 8 || time.len() != 6 || lens.len() != 2 || number.is_empty() || number.len() > 6 {
        return None;
    }
    if !(date.bytes().chain(time.bytes()).chain(lens.bytes()).chain(number.bytes()).all(|b| b.is_ascii_digit())) {
        return None;
    }
    let kind = match (prefix, ext.as_str()) {
        ("VID", "insv" | "mp4") => Kind::Video,
        ("IMG", "insp" | "jpg" | "dng") => Kind::Photo,
        ("LRV", "lrv" | "insv" | "mp4") => Kind::Proxy,
        _ => return None,
    };
    let shot = NaiveDateTime::parse_from_str(&format!("{date}{time}"), "%Y%m%d%H%M%S").ok()?;
    // a proxy belongs to the video of its time and counter, whose lens part is 00
    let stem = match kind {
        Kind::Proxy => format!("VID_{date}_{time}_00_{number}"),
        _ => format!("{prefix}_{date}_{time}_{lens}_{number}"),
    };
    Some(Name { kind, stem, shot, lens: lens.to_string(), number: number.to_string() })
}

/// One recording: its main file and what goes with it.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Recording {
    /// `VID_20260913_114003_00_093`.
    pub stem: String,
    pub kind: Kind,
    /// When it started, camera time (`2026-09-13T11:40:03`).
    #[serde(serialize_with = "iso")]
    pub shot: NaiveDateTime,
    pub path: PathBuf,
    pub bytes: u64,
    /// The second lens's file of a two-file camera (`…_10_…`), if any.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub second_lens: Option<PathBuf>,
    /// The low-resolution copy (`LRV_…`), if beside it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub proxy: Option<PathBuf>,
    /// The file — and its proxy, when there is one — ends with the camera's trailer: finished (a
    /// download in progress has none yet; Studio writes the proxy after the recording).
    pub complete: bool,
    /// What the trailer says, when complete.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub camera: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub lens_size: Option<(u32, u32)>,
}

fn iso<S: serde::Serializer>(t: &NaiveDateTime, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_str(&t.format("%Y-%m-%dT%H:%M:%S").to_string())
}

/// The folders looked in by default: Studio's download folder, and every mounted card's
/// `DCIM/Camera01` (the camera connected over USB, or its card in a reader).
pub fn default_folders() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(home) = std::env::var_os("HOME").map(PathBuf::from) {
        v.push(home.join("Downloads").join("Insta360 Download"));
    }
    if let Ok(vols) = std::fs::read_dir("/Volumes") {
        for e in vols.flatten().take(64) {
            let dcim = e.path().join("DCIM");
            if let Ok(cams) = std::fs::read_dir(&dcim) {
                for c in cams.flatten().take(64) {
                    if c.file_name().to_string_lossy().starts_with("Camera") {
                        v.push(c.path());
                    }
                }
            }
        }
    }
    v
}

/// Every Insta360 recording under `folders` (to [`MAX_DEPTH`] levels, at most [`MAX_FILES`]
/// files looked at), newest first; proxies and second-lens files joined to their recording.
pub fn scan(folders: &[PathBuf]) -> Vec<Recording> {
    let mut files: Vec<(PathBuf, Name, u64)> = Vec::new();
    let mut seen = 0usize;
    for f in folders {
        walk(f, 0, &mut seen, &mut files);
    }
    let mut recs: Vec<Recording> = Vec::new();
    // main files first, so that their companions find them
    files.sort_by_key(|(_, n, _)| (n.kind == Kind::Proxy, n.lens == "10"));
    for (path, name, bytes) in files {
        let at = recs.iter().position(|r| r.stem == name.stem || (name.lens == "10" && r.stem == name.stem.replacen("_10_", "_00_", 1)));
        match (name.kind, name.lens.as_str(), at) {
            (Kind::Proxy, _, Some(i)) => {
                if let Some(r) = recs.get_mut(i) {
                    r.proxy.get_or_insert(path);
                }
            }
            (Kind::Video, "10", Some(i)) => {
                if let Some(r) = recs.get_mut(i) {
                    r.second_lens.get_or_insert(path);
                }
            }
            // the same recording found twice (downloaded again, on the card too): the first kept
            (_, _, Some(_)) => {}
            (Kind::Proxy, _, None) => {}
            (kind, _, None) => recs.push(Recording {
                stem: name.stem,
                kind,
                shot: name.shot,
                path,
                bytes,
                second_lens: None,
                proxy: None,
                complete: false,
                camera: None,
                lens_size: None,
            }),
        }
    }
    let finished = |p: &Path| std::fs::File::open(p).is_ok_and(|f| filmcraft_isobmff::insta360::read_trailer(&f).is_some());
    for r in &mut recs {
        if let Ok(f) = std::fs::File::open(&r.path) {
            let main = filmcraft_isobmff::insta360::read_trailer(&f).is_some();
            r.complete = main && r.proxy.as_deref().is_none_or(finished);
            if let Some(i) = main.then(|| filmcraft_isobmff::insta360::read_info(&f)).flatten() {
                r.camera = (!i.camera.is_empty()).then_some(i.camera);
                r.lens_size = i.lens_size;
            }
        }
    }
    recs.sort_by(|a, b| b.shot.cmp(&a.shot).then_with(|| a.stem.cmp(&b.stem)));
    recs
}

fn walk(dir: &Path, depth: usize, seen: &mut usize, out: &mut Vec<(PathBuf, Name, u64)>) {
    if depth > MAX_DEPTH || *seen >= MAX_FILES {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else { return };
    for e in rd.flatten() {
        *seen += 1;
        if *seen >= MAX_FILES {
            return;
        }
        let Ok(ft) = e.file_type() else { continue };
        let path = e.path();
        if ft.is_dir() {
            // not into hidden folders or app bundles
            let n = e.file_name();
            let n = n.to_string_lossy();
            if !n.starts_with('.') && !n.ends_with(".app") {
                walk(&path, depth + 1, seen, out);
            }
        } else if ft.is_file()
            && let Some(name) = parse_name(&e.file_name().to_string_lossy())
        {
            let bytes = e.metadata().map(|m| m.len()).unwrap_or(0);
            out.push((path, name, bytes));
        }
    }
}

/// The recordings matching `wanted` — full stems (`VID_20260913_114003_00_093`), Studio's short
/// names (`VID_093`), or file names — with the names nothing matched.
pub fn select<'a>(recs: &'a [Recording], wanted: &[String]) -> (Vec<&'a Recording>, Vec<String>) {
    let mut found: Vec<&Recording> = Vec::new();
    let mut missing = Vec::new();
    for w in wanted {
        let w = w.trim();
        let base = w.rsplit_once('.').map_or(w, |(b, _)| b);
        let hits: Vec<&Recording> = recs.iter().filter(|r| r.stem.eq_ignore_ascii_case(base) || short_name(&r.stem).eq_ignore_ascii_case(base)).collect();
        if hits.is_empty() {
            missing.push(w.to_string());
        }
        for h in hits {
            if !found.iter().any(|f| f.stem == h.stem) {
                found.push(h);
            }
        }
    }
    (found, missing)
}

/// Studio's short name of a recording: `VID_093` for `VID_20260913_114003_00_093`.
pub fn short_name(stem: &str) -> String {
    let mut p = stem.split('_');
    match (p.next(), p.next_back()) {
        (Some(a), Some(n)) => format!("{a}_{n}"),
        _ => stem.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_parse() {
        let n = parse_name("VID_20260913_114003_00_093.insv").unwrap();
        assert_eq!((n.kind, n.stem.as_str(), n.lens.as_str(), n.number.as_str()), (Kind::Video, "VID_20260913_114003_00_093", "00", "093"));
        assert_eq!(n.shot.format("%Y-%m-%d %H:%M:%S").to_string(), "2026-09-13 11:40:03");
        let p = parse_name("LRV_20260913_114003_01_093.lrv").unwrap();
        assert_eq!((p.kind, p.stem.as_str()), (Kind::Proxy, "VID_20260913_114003_00_093"));
        assert_eq!(parse_name("IMG_20250101_000000_00_001.insp").unwrap().kind, Kind::Photo);
        for bad in [
            "VID_2026091_114003_00_093.insv",
            "VID_20261399_114003_00_093.insv",
            "clip.mp4",
            "VID_20260913_114003_00_093(1).insv",
            "VID_20260913_114003_00_093_x.insv",
            "VID.insv",
            "",
            ".",
            "VID_20260913_114003_0a_093.insv",
        ] {
            assert!(parse_name(bad).is_none(), "{bad}");
        }
        assert_eq!(short_name("VID_20260913_114003_00_093"), "VID_093");
    }

    #[test]
    fn scanning_groups_recordings_and_their_proxies() {
        let dir = std::env::temp_dir().join(format!("insta360-scan-{}", std::process::id()));
        let sub = dir.join("VID_20260913_114003_00_093-Original");
        std::fs::create_dir_all(&sub).unwrap();
        std::fs::write(sub.join("VID_20260913_114003_00_093.insv"), b"not finished").unwrap();
        std::fs::write(sub.join("LRV_20260913_114003_01_093.lrv"), b"proxy").unwrap();
        std::fs::write(dir.join("VID_20250101_100000_00_001.insv"), b"x").unwrap();
        std::fs::write(dir.join("VID_20250101_100000_10_001.insv"), b"x").unwrap();
        std::fs::write(dir.join("notes.txt"), b"x").unwrap();
        let recs = scan(std::slice::from_ref(&dir));
        assert_eq!(recs.len(), 2, "{recs:?}");
        assert_eq!(recs[0].stem, "VID_20260913_114003_00_093");
        assert!(recs[0].proxy.is_some() && !recs[0].complete);
        assert!(recs[1].second_lens.is_some());
        let (found, missing) = select(&recs, &["VID_093".into(), "VID_20250101_100000_00_001.insv".into(), "VID_777".into()]);
        assert_eq!(found.len(), 2);
        assert_eq!(missing, vec!["VID_777".to_string()]);
        let _ = std::fs::remove_dir_all(&dir);
        // a folder that is not there is no recordings
        assert!(scan(&[dir.join("gone")]).is_empty());
    }
}
