//! What is checked on this machine before a byte goes to YouTube: the video file (a type YouTube
//! takes, within its size limit), the thumbnail (JPEG or PNG, within its limit), the chapters a
//! showcase export leaves beside itself, and whether the video looks like a Short.

use std::io::Read as _;
use std::path::{Path, PathBuf};

use crate::YtError;

/// YouTube's limit on a video file.
pub const MAX_VIDEO_BYTES: u64 = 256 * 1024 * 1024 * 1024;
/// YouTube's limit on a custom thumbnail (`thumbnails.set`).
pub const MAX_THUMBNAIL_BYTES: u64 = 50 * 1024 * 1024;
/// The chapters file read at most (a showcase's is a few KB).
const MAX_CHAPTERS_BYTES: u64 = 16 * 1024;
/// Shorts: vertical or square, at most three minutes (YouTube classifies such uploads itself).
pub const SHORTS_MAX_SECONDS: f64 = 180.0;
/// Videos longer than this need a verified channel (`status.longUploadsStatus: allowed`).
pub const LONG_UPLOAD_SECONDS: f64 = 15.0 * 60.0;

/// The video types YouTube takes, by extension, and the MIME type sent as `X-Upload-Content-Type`
/// (YouTube takes any `video/*`).
const VIDEO_TYPES: &[(&str, &str)] = &[
    ("mp4", "video/mp4"),
    ("m4v", "video/x-m4v"),
    ("mov", "video/quicktime"),
    ("webm", "video/webm"),
    ("mkv", "video/x-matroska"),
    ("avi", "video/x-msvideo"),
    ("wmv", "video/x-ms-wmv"),
    ("flv", "video/x-flv"),
    ("3gp", "video/3gpp"),
    ("mpg", "video/mpeg"),
    ("mpeg", "video/mpeg"),
    ("mts", "video/mp2t"),
    ("m2ts", "video/mp2t"),
    ("ts", "video/mp2t"),
    ("mxf", "application/mxf"),
];

/// A video about to be uploaded.
#[derive(Debug, Clone, PartialEq)]
pub struct LocalVideo {
    pub bytes: u64,
    pub mime: &'static str,
    /// Its shape, when its header could be read (MP4, MOV, M4V).
    pub shape: Option<Shape>,
}

/// A video's displayed size and length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub width: u32,
    pub height: u32,
    pub seconds: f64,
}

impl Shape {
    /// Vertical or square, at most three minutes: YouTube makes it a Short.
    pub fn is_short(&self) -> bool {
        self.height >= self.width && self.width > 0 && self.seconds > 0.0 && self.seconds <= SHORTS_MAX_SECONDS
    }
}

/// Check `path` before anything is sent: a regular, non-empty file of a type YouTube takes,
/// within its size limit; its shape when its header can be read.
pub fn check_video(path: &Path) -> Result<LocalVideo, YtError> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    let Some((_, mime)) = VIDEO_TYPES.iter().find(|(e, _)| *e == ext) else {
        let what = if ext.is_empty() { "a file without an extension".to_string() } else { format!("a .{ext} file") };
        return Err(YtError::Invalid(format!("YouTube does not take {what}: upload a video (MP4, MOV, WebM, MKV, AVI, …)")));
    };
    let meta = std::fs::metadata(path).map_err(|e| YtError::Invalid(format!("{}: {e}", path.display())))?;
    if !meta.is_file() {
        return Err(YtError::Invalid(format!("{} is not a file", path.display())));
    }
    let bytes = meta.len();
    if bytes == 0 {
        return Err(YtError::Invalid(format!("{} is empty", path.display())));
    }
    if bytes > MAX_VIDEO_BYTES {
        return Err(YtError::Invalid(format!("{} is {bytes} bytes; YouTube takes videos up to 256 GB", path.display())));
    }
    let shape = if matches!(ext.as_str(), "mp4" | "m4v" | "mov") { shape_of(path) } else { None };
    Ok(LocalVideo { bytes, mime, shape })
}

/// An MP4/MOV's displayed size (its first video track, turned as its matrix says) and length,
/// from its header alone; `None` when it cannot be read.
pub fn shape_of(path: &Path) -> Option<Shape> {
    let file = std::fs::File::open(path).ok()?;
    let mp4 = filmcraft_isobmff::open(file).ok()?;
    let t = mp4.tracks.get(mp4.track_of_kind(filmcraft_isobmff::TrackKind::Video)?)?;
    let (mut w, mut h) = (t.width, t.height);
    if w == 0 || h == 0 {
        let v = t.video()?;
        (w, h) = (u32::from(v.width), u32::from(v.height));
    }
    if matches!(t.display_rotation(), Some(1 | 3)) {
        std::mem::swap(&mut w, &mut h);
    }
    let seconds = if mp4.timescale > 0 && mp4.duration > 0 {
        mp4.duration as f64 / f64::from(mp4.timescale)
    } else if t.timescale > 0 {
        t.duration as f64 / f64::from(t.timescale)
    } else {
        0.0
    };
    (w > 0 && h > 0 && seconds.is_finite()).then_some(Shape { width: w, height: h, seconds })
}

/// What to tell the user about Shorts: whether YouTube will make it one, and why (not).
pub fn shorts_note(shape: Option<&Shape>) -> (bool, String) {
    match shape {
        Some(s) if s.is_short() => {
            (true, format!("{}×{}, {:.0} s: YouTube will make this a Short (vertical or square, 3 minutes or less)", s.width, s.height, s.seconds))
        }
        Some(s) if s.height >= s.width => {
            (false, format!("{}×{} but {:.0} s: longer than 3 minutes, so a regular video, not a Short", s.width, s.height, s.seconds))
        }
        Some(s) => (false, format!("{}×{}: horizontal, a regular video (a Short is vertical or square and 3 minutes or less)", s.width, s.height)),
        None => (false, "its size and length could not be read here; YouTube makes vertical or square videos of 3 minutes or less Shorts".into()),
    }
}

/// A thumbnail about to be set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Thumbnail {
    pub path: PathBuf,
    pub mime: &'static str,
    pub bytes: u64,
}

/// Check a thumbnail: a JPEG or PNG (by its first bytes), within YouTube's limit.
pub fn check_thumbnail(path: &Path) -> Result<Thumbnail, YtError> {
    let meta = std::fs::metadata(path).map_err(|e| YtError::Invalid(format!("thumbnail {}: {e}", path.display())))?;
    if !meta.is_file() || meta.len() == 0 {
        return Err(YtError::Invalid(format!("thumbnail {} is not a file with an image in it", path.display())));
    }
    if meta.len() > MAX_THUMBNAIL_BYTES {
        return Err(YtError::Invalid(format!("thumbnail {} is {} bytes; YouTube takes at most 50 MB (1280×720 is what it shows)", path.display(), meta.len())));
    }
    let mut head = [0u8; 8];
    let n = std::fs::File::open(path).and_then(|f| f.take(8).read(&mut head)).map_err(|e| YtError::Invalid(format!("thumbnail {}: {e}", path.display())))?;
    let head = head.get(..n).unwrap_or_default();
    let mime = if head.starts_with(&[0xFF, 0xD8, 0xFF]) {
        "image/jpeg"
    } else if head.starts_with(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]) {
        "image/png"
    } else {
        return Err(YtError::Invalid(format!("thumbnail {} is not a JPEG or PNG image", path.display())));
    };
    Ok(Thumbnail { path: path.to_path_buf(), mime, bytes: meta.len() })
}

/// `<file>.chapters.txt` beside a video (what `showcase --export` writes).
pub fn chapters_beside(video: &Path) -> PathBuf {
    video.with_extension("chapters.txt")
}

/// `<file>.thumbnail.jpg` beside a video (what `showcase --export` writes).
pub fn thumbnail_beside(video: &Path) -> PathBuf {
    video.with_extension("thumbnail.jpg")
}

/// The largest QA report read.
const MAX_QA_BYTES: u64 = 32 << 20;

/// Why a video must not go out: the QA report beside it (`<file>.qa.json`, what a showcase's QA
/// writes) lists privacy errors — faces of people its picture shows unblurred. `None` without a
/// report, or with one that cannot be read, or without such errors.
pub fn privacy_blocker(video: &Path) -> Option<String> {
    let qa = video.with_extension("qa.json");
    let f = std::fs::File::open(&qa).ok()?;
    let mut text = String::new();
    f.take(MAX_QA_BYTES).read_to_string(&mut text).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let errors: Vec<&serde_json::Value> =
        v["findings"].as_array()?.iter().filter(|f| f["check"].as_str() == Some("privacy") && f["severity"].as_str() == Some("error")).collect();
    let first = errors.first()?;
    let at = first["seconds"].as_f64().filter(|s| s.is_finite() && *s >= 0.0).unwrap_or(0.0);
    Some(format!(
        "{} lists {} unblurred face(s) of people (the first at {}:{:04.1}): blur them and check again, or upload anyway with --allow-privacy-findings",
        qa.display(),
        errors.len(),
        (at / 60.0).floor() as u64,
        at % 60.0
    ))
}

/// YouTube chapters from a file: lines of `0:00 Title` (the first at 0:00, at least three, as
/// YouTube wants them). `Ok(None)` when there is no file; `Err` says why a file is not usable.
pub fn read_chapters(path: &Path) -> Result<Option<String>, String> {
    let Ok(f) = std::fs::File::open(path) else { return Ok(None) };
    let mut b = Vec::new();
    f.take(MAX_CHAPTERS_BYTES + 1).read_to_end(&mut b).map_err(|e| format!("{}: {e}", path.display()))?;
    if b.len() as u64 > MAX_CHAPTERS_BYTES {
        return Err(format!("{} is larger than chapters ever are", path.display()));
    }
    let text = String::from_utf8_lossy(&b);
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
    let stamp = |l: &str| {
        let t = l.split_whitespace().next().unwrap_or_default();
        !t.is_empty() && t.split(':').count() >= 2 && t.split(':').all(|p| !p.is_empty() && p.bytes().all(|c| c.is_ascii_digit()))
    };
    if lines.len() < 3 || !lines.iter().all(|l| stamp(l)) {
        return Err(format!("{} is not YouTube chapters (lines of `0:00 Title`, at least three)", path.display()));
    }
    if !lines.first().is_some_and(|l| l.starts_with("0:00") || l.starts_with("00:00")) {
        return Err(format!("{}: YouTube chapters start at 0:00", path.display()));
    }
    if lines.iter().any(|l| l.contains(['<', '>'])) {
        return Err(format!("{} contains < or >, which YouTube descriptions do not take", path.display()));
    }
    Ok(Some(lines.join("\n")))
}

/// `description` with `chapters` appended after a blank line, unless that passes YouTube's limit
/// (then the description alone, and why).
pub fn with_chapters(description: Option<&str>, chapters: &str) -> Result<String, String> {
    let joined = match description.map(str::trim).filter(|d| !d.is_empty()) {
        Some(d) => format!("{d}\n\n{chapters}"),
        None => chapters.to_string(),
    };
    if joined.len() > filmcraft_engine::youtube::MAX_DESCRIPTION_BYTES {
        return Err(format!(
            "the chapters would make the description {} bytes, more than YouTube's {}; uploaded without them",
            joined.len(),
            filmcraft_engine::youtube::MAX_DESCRIPTION_BYTES
        ));
    }
    Ok(joined)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-yt-media-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn video_files_are_checked() {
        let d = dir("video");
        let ok = d.join("show.MP4");
        std::fs::write(&ok, b"not really an mp4").unwrap();
        let v = check_video(&ok).unwrap();
        assert_eq!((v.bytes, v.mime, v.shape), (17, "video/mp4", None), "an unreadable header is no shape, not an error");
        for (name, body, want) in [("a.wav", &b"x"[..], "does not take a .wav"), ("noext", b"x", "without an extension"), ("e.mov", b"", "is empty")] {
            let p = d.join(name);
            std::fs::write(&p, body).unwrap();
            assert!(check_video(&p).unwrap_err().to_string().contains(want), "{name}");
        }
        assert!(check_video(&d.join("missing.mp4")).is_err());
        std::fs::create_dir_all(d.join("dir.mp4")).unwrap();
        assert!(check_video(&d.join("dir.mp4")).unwrap_err().to_string().contains("not a file"));
        let _ = std::fs::remove_dir_all(&d);
    }

    /// A showcase whose QA found a visitor's face unblurred is not uploaded (Every Robot I've Met
    /// went out with two, 2026-10-21); warnings and other checks' errors do not stop it.
    #[test]
    fn a_qa_report_with_unblurred_faces_stops_the_upload() {
        let d = dir("qa");
        let v = d.join("Every Robot (Malayalam, 4K).mp4");
        std::fs::write(&v, b"x").unwrap();
        assert_eq!(privacy_blocker(&v), None, "no report");
        let qa = d.join("Every Robot (Malayalam, 4K).qa.json");
        let write = |findings: serde_json::Value| std::fs::write(&qa, serde_json::json!({"findings": findings}).to_string()).unwrap();
        write(serde_json::json!([{"check": "privacy", "severity": "warning", "seconds": 3.0}, {"check": "black", "severity": "error", "seconds": 4.0}]));
        assert_eq!(privacy_blocker(&v), None);
        write(serde_json::json!([{"check": "privacy", "severity": "error", "seconds": 427.8}, {"check": "privacy", "severity": "error", "seconds": "x"}]));
        let why = privacy_blocker(&v).unwrap();
        assert!(why.contains("2 unblurred face(s)") && why.contains("7:07.8") && why.contains("--allow-privacy-findings"), "{why}");
        // a damaged report is no report
        std::fs::write(&qa, b"{not json").unwrap();
        assert_eq!(privacy_blocker(&v), None);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn shorts_are_vertical_or_square_and_short() {
        let s = |w, h, seconds| Shape { width: w, height: h, seconds };
        assert!(s(1080, 1920, 59.0).is_short() && s(1080, 1080, 180.0).is_short());
        assert!(!s(1920, 1080, 30.0).is_short() && !s(1080, 1920, 181.0).is_short() && !s(0, 0, 10.0).is_short());
        assert!(shorts_note(Some(&s(1080, 1920, 30.0))).0);
        assert!(shorts_note(Some(&s(1080, 1920, 300.0))).1.contains("longer than 3 minutes"));
        assert!(shorts_note(Some(&s(1920, 1080, 30.0))).1.contains("horizontal"));
        assert!(!shorts_note(None).0);
    }

    #[test]
    fn thumbnails_are_jpeg_or_png() {
        let d = dir("thumb");
        let jpg = d.join("t.jpg");
        std::fs::write(&jpg, [0xFF, 0xD8, 0xFF, 0xE0, 0, 0]).unwrap();
        assert_eq!(check_thumbnail(&jpg).unwrap().mime, "image/jpeg");
        let png = d.join("t.png");
        std::fs::write(&png, [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0]).unwrap();
        assert_eq!(check_thumbnail(&png).unwrap().mime, "image/png");
        for (name, body) in [("g.gif", &b"GIF89a"[..]), ("short.jpg", &[0xFF][..]), ("empty.jpg", &[][..])] {
            let p = d.join(name);
            std::fs::write(&p, body).unwrap();
            assert!(check_thumbnail(&p).is_err(), "{name}");
        }
        assert!(check_thumbnail(&d.join("none.jpg")).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn chapters_beside_a_showcase_export() {
        let d = dir("chapters");
        let video = d.join("show.mp4");
        assert_eq!(chapters_beside(&video), d.join("show.chapters.txt"));
        assert_eq!(thumbnail_beside(&video), d.join("show.thumbnail.jpg"));
        assert_eq!(read_chapters(&chapters_beside(&video)).unwrap(), None);
        std::fs::write(chapters_beside(&video), "0:00 Intro\n0:12 #1 Ameca\n\n1:02:03 Outro\n").unwrap();
        assert_eq!(read_chapters(&chapters_beside(&video)).unwrap().unwrap(), "0:00 Intro\n0:12 #1 Ameca\n1:02:03 Outro");
        for bad in ["0:00 Intro\n0:12 Two\n", "0:05 A\n0:12 B\n0:20 C\n", "0:00 A\nlater B\n0:20 C\n", "0:00 A\n0:10 <b>\n0:20 C\n", "0:00 A\n:10 B\n0:20 C"] {
            std::fs::write(chapters_beside(&video), bad).unwrap();
            assert!(read_chapters(&chapters_beside(&video)).is_err(), "{bad:?}");
        }
        std::fs::write(chapters_beside(&video), "0:00 x\n".repeat(5000)).unwrap();
        assert!(read_chapters(&chapters_beside(&video)).unwrap_err().contains("larger"));
        assert_eq!(with_chapters(Some(" Robots "), "0:00 A").unwrap(), "Robots\n\n0:00 A");
        assert_eq!(with_chapters(None, "0:00 A").unwrap(), "0:00 A");
        assert!(with_chapters(Some(&"a".repeat(4990)), "0:00 A\n0:10 B\n0:20 C").is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
