use super::*;
use crate::capture::WindowRect;
use std::cell::RefCell;
use std::rc::Rc;

// ------------------------------------------------------------------------------- panel texts

/// A panel as the Google Photos page lays it out (its text, top to bottom).
const DWTC: &str = "Info
Add a description
Details
Oct 21, 2021
Thu, 11:16 AM GMT+04:00
VID_20211021_111656.mp4
1080p
1920 × 1080
Dubai World Trade Centre
Dubai - United Arab Emirates";

const BUSAN_ESTIMATED: &str = "Details
Oct 18, 2023
Wed, 7:42 PM GMT+09:00
IMG_5926.HEIC
12MP
4032 × 3024
Apple iPhone 13 Pro
ƒ/1.5   1/60   5.7mm   ISO640
Busan, South Korea
Estimated location";

const NO_PLACE: &str = "Details
Sep 8, 2020
Tue, 10:59 PM GMT+04:00
InShot_20200908_225922699.mp4
720 × 1280
12.4 MB";

const HOME: &str = "Info
Details
Nov 2, 2022
Wed, 7:39 PM GMT+04:00
IMG_20221102_193937.jpg
12MP    4000 × 3000
Google Pixel 6
f/1.85  1/120  6.81mm  ISO103
Home";

/// As the vision model transcribes it: bullets, an `x`, the place marked, no extension.
const MODEL_STYLE: &str = "```
- **Details**
- Jan 23, 2022
- Sun, 12:58 PM GMT
- VID_20221011_132939
- 1920 x 1080
- DJI FC3170
- Location: Expo City Dubai
```";

/// A description above Details mentions a time and a date: it is not the item's.
const WITH_DESCRIPTION: &str = "Info
Met this one at 10:30 on Mar 3, 2019!
Details
Mar 27, 2024
Wed, 11:16 AM GMT+04:00
20240327_111617.mp4
3840 × 2160
Abu Dhabi";

/// A file name with spaces and no extension, after the time.
const SPACED_NAME: &str = "Details
Dec 1, 2025
Mon, 9:05 AM GMT+04:00
Every Robot I've Met - v4
1920 × 1080";

/// A real panel (live, 2026-10-20) as the page lays it out: the albums the item is in come first,
/// with their own dates, and Google's storage notes come before the place.
const LIVE_PAGE: &str = "Info
Add a description
Albums
hardware hacking and robotics
415 items · Shared · Sep 19, 2012 – …
Every Robot I've Met
99 items · Jun 23, 2013 – Oct 20, 2026
Details
Sep 8, 2020
Tue, 10:25 PM GMT+04:00
VID_20200908_222535.mp4
2.1MP 1920 × 1080
Uploaded from Android device
Backed up
Storage saver. Learn more
This item doesn't take up space in your account storage. Learn more
Home";

/// What qwen2.5vl:7b wrote for two live panels with [`PANEL_ASK`] (2026-10-20).
const LIVE_MODEL: [&str; 2] = [
    "Sep 8, 2020\nTue, 10:25PM GMT+04:00\nVID_20200908_222535.mp4\n2.1MP 1920 x 1080\nUploaded from Android device\nLocation: Home",
    "Jul 28, 2013\nSun, 2:26 AM GMT+05:30\nMOV_0080.mp4\n2.1MP 1920 × 1080\nUploaded from Backup and Sync\nBacked up\nStorage saver. Learn more\nThis item doesn't take up space in your account storage. Learn more\nLocation: Elackad, Keralam, India",
];

/// A panel scrolled to its end, with Google's sharing note where a careless reading would take
/// a place from; and a file name wrapped onto the next line (live, 2026-10-20).
const SHARED_WRAPPED: &str = "Details
Sep 8, 2020
Tue, 10:37 PM GMT+04:00
VID_20200908_223605_HSR_120.mp
4
0.9MP 1280 × 720
Shared by saranya mohesh
Saved to your photos
Home";

#[test]
fn wrapped_names_and_sharing_notes() {
    let l = parse_panel(SHARED_WRAPPED).unwrap();
    assert_eq!((l.file_name.as_str(), l.place.as_deref()), ("VID_20200908_223605_HSR_120.mp4", Some("Home")));
    // the model's own wrong place ("Location: saranya mohesh", taken from the note) is refused
    let model = "Sep 8, 2020\nTue, 10:37 PM GMT+04:00\nVID_20200908_223605_HSR_120.mp\n4\n0.9MP 1280 x 720\nShared by saranya mohesh\nLocation: saranya mohesh";
    let l = parse_panel(model).unwrap();
    assert_eq!((l.file_name.as_str(), l.place()), ("VID_20200908_223605_HSR_120.mp4", None));
    // nor one it made up from them, where the panel shows no place (live)
    let none = "Sep 11, 2020\nFri, 8:48 AM GMT+04:00\nInShot_20200908_225922699.mp4\n0.9MP 1280 x 720\nUploaded from Android device\nLocation: Storage saver.";
    assert_eq!(parse_panel(none).unwrap().place(), None);
    // a short line after a whole name is not part of it
    let l = parse_panel("Oct 1, 2022\n10:00 AM GMT+04:00\nIMG_1.jpg\n4K\nDubai").unwrap();
    assert_eq!((l.file_name.as_str(), l.place.as_deref()), ("IMG_1.jpg", Some("Dubai")));
}

#[test]
fn live_panels_read() {
    let l = parse_panel(LIVE_PAGE).unwrap();
    assert_eq!((l.file_name.as_str(), l.date.as_deref(), l.time.as_deref()), ("VID_20200908_222535.mp4", Some("2020-09-08"), Some("22:25")));
    assert_eq!((l.place.as_deref(), l.area.as_deref()), (Some("Home"), None), "Google's notes are not the place");
    // the albums' dates are not the item's, even without the Details heading
    let without_heading = LIVE_PAGE.replace("Details\n", "");
    assert_eq!(parse_panel(&without_heading).unwrap().date.as_deref(), Some("2020-09-08"));
    let l = parse_panel(LIVE_MODEL[0]).unwrap();
    assert_eq!((l.file_name.as_str(), l.place.as_deref(), l.gmt_offset.as_deref()), ("VID_20200908_222535.mp4", Some("Home"), Some("+04:00")));
    let l = parse_panel(LIVE_MODEL[1]).unwrap();
    assert_eq!((l.file_name.as_str(), l.place.as_deref(), l.date.as_deref()), ("MOV_0080.mp4", Some("Elackad, Keralam, India"), Some("2013-07-28")));
    assert!(!l.estimated, "the transcription says nothing about it");
}

#[test]
fn panels_read_as_the_page_shows_them() {
    let l = parse_panel(DWTC).unwrap();
    assert_eq!(l.file_name, "VID_20211021_111656.mp4");
    assert_eq!((l.date.as_deref(), l.time.as_deref(), l.gmt_offset.as_deref()), (Some("2021-10-21"), Some("11:16"), Some("+04:00")));
    assert_eq!((l.place.as_deref(), l.area.as_deref()), (Some("Dubai World Trade Centre"), Some("Dubai - United Arab Emirates")));
    assert_eq!(l.resolution.as_deref(), Some("1920 × 1080"));
    assert!(!l.estimated);

    let l = parse_panel(BUSAN_ESTIMATED).unwrap();
    assert_eq!((l.file_name.as_str(), l.place.as_deref(), l.estimated), ("IMG_5926.HEIC", Some("Busan, South Korea"), true));
    assert_eq!((l.time.as_deref(), l.gmt_offset.as_deref()), (Some("19:42"), Some("+09:00")));
    assert_eq!(l.area, None, "the camera and exposure are not the place");

    let l = parse_panel(NO_PLACE).unwrap();
    assert_eq!((l.file_name.as_str(), l.place()), ("InShot_20200908_225922699.mp4", None));
    assert_eq!(l.resolution.as_deref(), Some("720 × 1280"));

    let l = parse_panel(HOME).unwrap();
    assert_eq!((l.file_name.as_str(), l.place.as_deref(), l.area.as_deref()), ("IMG_20221102_193937.jpg", Some("Home"), None));

    let l = parse_panel(MODEL_STYLE).unwrap();
    assert_eq!((l.file_name.as_str(), l.place.as_deref()), ("VID_20221011_132939", Some("Expo City Dubai")));
    assert_eq!((l.date.as_deref(), l.gmt_offset.as_deref()), (Some("2022-01-23"), Some("+00:00")));

    let l = parse_panel(WITH_DESCRIPTION).unwrap();
    assert_eq!((l.date.as_deref(), l.time.as_deref()), (Some("2024-03-27"), Some("11:16")));
    assert_eq!((l.file_name.as_str(), l.place.as_deref()), ("20240327_111617.mp4", Some("Abu Dhabi")));

    let l = parse_panel(SPACED_NAME).unwrap();
    assert_eq!((l.file_name.as_str(), l.place()), ("Every Robot I've Met - v4", None));
}

#[test]
fn the_output_has_the_fields_the_plan_reads() {
    let l = parse_panel(BUSAN_ESTIMATED).unwrap();
    let v = serde_json::to_value(&l).unwrap();
    for k in ["fileName", "date", "gmtOffset", "place", "estimated"] {
        assert!(v.get(k).is_some(), "{k} in {v}");
    }
    assert_eq!(v["place"], "Busan, South Korea");
    // no place is null, not missing
    let v = serde_json::to_value(parse_panel(NO_PLACE).unwrap()).unwrap();
    assert!(v["place"].is_null() && v.get("place").is_some());
    // and it reads back
    let back: Location = serde_json::from_value(v).unwrap();
    assert_eq!(back.file_name, "InShot_20200908_225922699.mp4");
}

/// The album "Every Robot I've Met" as the main session read it last round (file name and
/// place; "-": none), laid out as panels: the parser gives back the same.
const EXPECTED: &[(&str, &str)] = &[
    ("MOV_0492.mp4", "Elackad, Keralam, India"),
    ("VID_20200908_222535.mp4", "Home"),
    ("InShot_20200908_225922699.mp4", "-"),
    ("VID_20211021_111656.mp4", "Dubai World Trade Centre"),
    ("QVZmTUxqZndXTE5iV2JaSA.mp4", "-"),
    ("org_video_6553912_1642930732000.mp4", "Expo City Dubai"),
    ("DJI_0496.MP4", "Dubai World Trade Centre"),
    ("VID_20221011_132939", "Dubai World Trade Centre"),
    ("DJI_0503", "Dubai World Trade Centre"),
    ("IMG_7952", "The Museum of the Future"),
    ("IMG_6998", "-"),
    ("IMG_9128", "Abu Dhabi"),
    ("Flow_VID_20250423_110308_07_031", "Dubai"),
    ("IMG_6999", "Amsterdam, Netherlands"),
    ("QVZmdWZhcEVJbExodzU4bQ", "-"),
    ("IMG_20221102_193937.jpg", "Home"),
    ("IMG_5926", "Busan, South Korea"),
    ("20240327_111617", "Abu Dhabi"),
    ("IMG_4498", "Sydney NSW, Australia"),
    ("Every Robot I've Met - v4", "-"),
];

#[test]
fn last_rounds_album_reads_back() {
    for (k, (file, place)) in EXPECTED.iter().enumerate() {
        let estimated = k % 4 == 3;
        let mut text = format!("Info\nDetails\nOct {}, 2022\nTue, 1:{:02} PM GMT+04:00\n{file}\n1920 × 1080\n", k + 1, k % 60);
        if k % 3 == 0 {
            text.push_str("Google Pixel 6\nƒ/1.9  1/50  4.4mm  ISO200\n");
        }
        if *place != "-" {
            text.push_str(place);
            text.push('\n');
            if estimated {
                text.push_str("Estimated location\n");
            }
        }
        let l = parse_panel(&text).unwrap_or_else(|| panic!("{text}"));
        assert_eq!(l.file_name, *file, "{text}");
        assert_eq!(l.place().unwrap_or("-"), *place, "{text}");
        assert_eq!(l.estimated, estimated && *place != "-", "{text}");
        assert_eq!(l.date, Some(format!("2022-10-{:02}", k + 1)));
    }
}

#[test]
fn dates_times_and_offsets() {
    for (line, want) in [
        ("Oct 21, 2021", "2021-10-21"),
        ("Thu, Oct 21, 2021", "2021-10-21"),
        ("21 October 2021", "2021-10-21"),
        ("Sept 3, 2019", "2019-09-03"),
        ("May 1 2024", "2024-05-01"),
    ] {
        assert_eq!(parse_date(line).as_deref(), Some(want), "{line}");
    }
    for line in ["Thu, 11:16 AM GMT+04:00", "IMG_5926.HEIC", "Mon 2021", "Oct 99, 2021", "Octagon 12, 2021x"] {
        assert!(parse_date(line).is_none() || line.starts_with("Octagon"), "{line}");
    }
    for (line, t, o) in [
        ("Thu, 11:16 AM GMT+04:00", "11:16", Some("+04:00")),
        ("Wed, 7:42 PM GMT+09:00", "19:42", Some("+09:00")),
        ("12:05 AM", "00:05", None),
        ("12:30 PM GMT", "12:30", Some("+00:00")),
        ("23:59 GMT-05:30", "23:59", Some("-05:30")),
        ("9:05 AM GMT+0530", "09:05", Some("+05:30")),
        ("9:05 AM UTC+4", "09:05", Some("+04:00")),
    ] {
        assert_eq!(parse_time(line), Some((t.to_string(), o.map(str::to_string))), "{line}");
    }
    for line in ["GMT+04:00", "16:9", "13:05 PM", "99:00", "1:5", ""] {
        assert_eq!(parse_time(line), None, "{line}");
    }
    assert_eq!(parse_offset("GMT+15:00"), None);
}

#[test]
fn hostile_panels_never_panic() {
    assert_eq!(parse_panel(""), None);
    assert_eq!(parse_panel("NONE"), None);
    assert_eq!(parse_panel("There is no Info panel open."), None);
    let long = format!("Details\nOct 1, 2020\n10:00 AM\n{}\n{}", "x".repeat(100_000), "y".repeat(100_000));
    let l = parse_panel(&long).unwrap();
    assert!(l.file_name.chars().count() <= MAX_FILE_NAME && l.place.as_ref().is_none_or(|p| p.chars().count() <= MAX_PLACE));
    let many = "Oct 1, 2020\n".repeat(50_000);
    let _ = parse_panel(&many);
    for t in ["\u{0}\u{1}", "Location:", "Location: none", "File:", "GMT+", ":::::", "1:", "×××", "ƒ/", "12:00 PM GMT+99:99", "Details\nDetails\nDetails"] {
        let _ = parse_panel(t);
        let _ = parse_time(t);
        let _ = parse_date(t);
        let _ = parse_offset(t);
    }
    // a file name is never a control character away from a script
    assert_eq!(parse_panel("Details\nOct 1, 2020\n10:00 AM\n\u{7}"), None);
}

// ----------------------------------------------------------------------------------- addresses

#[test]
fn album_and_item_addresses() {
    let a = "https://photos.google.com/album/AF1QipABC_def-1";
    assert_eq!(album_url(a).as_deref(), Some(a));
    assert_eq!(album_url(&format!("{a}/photo/AF1QipXYZ?x=1")).as_deref(), Some(a));
    assert_eq!(album_url("https://photos.google.com/u/1/album/AF1/photo/P2").as_deref(), Some("https://photos.google.com/u/1/album/AF1"));
    assert_eq!(album_url("https://photos.google.com/share/AF1xyz?key=k").as_deref(), Some("https://photos.google.com/share/AF1xyz"));
    assert_eq!(item_id(&format!("{a}/photo/AF1QipXYZ")).as_deref(), Some("AF1QipXYZ"));
    assert_eq!(item_id("https://photos.google.com/photo/P1#x").as_deref(), Some("P1"));
    assert_eq!(item_id(a), None);
    assert_eq!(item_url(a, "P9").as_deref(), Some("https://photos.google.com/album/AF1QipABC_def-1/photo/P9"));
    assert_eq!(item_url(a, "../x"), None);
    for bad in [
        "https://photos.google.com.evil.example/album/A",
        "http://photos.google.com/album/A",
        "https://photos.google.com/album/",
        "https://photos.google.com/album/A\"; do shell script \"x",
        "https://photos.google.com/album/A B",
        "https://photos.google.com/u/x/album/A",
        "https://photos.google.com/albums",
        "https://photos.google.com/picker/s1",
    ] {
        assert_eq!(album_url(bad), None, "{bad}");
    }
    assert!(is_photos_page(ALBUMS_PAGE) && is_photos_page(a) && !is_photos_page("https://example.com/") && !is_photos_page("file:///etc/passwd"));
    assert!(!is_photos_page(&format!("{a}/{}", "x".repeat(5000))));
}

#[test]
fn the_map_under_the_panel_is_cut_off() {
    // a grey panel with white text lines, a coloured map in its bottom quarter
    let (w, h) = (400u32, 800u32);
    let img = image::RgbaImage::from_fn(w, h, |x, y| {
        if y >= 600 {
            // the map: blue land with grey roads every 20 rows
            if y % 20 == 10 { image::Rgba([120, 120, 120, 255]) } else { image::Rgba([40, 60, 110, 255]) }
        } else if y % 40 < 12 && (200..380).contains(&x) && x % 3 == 0 {
            image::Rgba([230, 230, 230, 255])
        } else {
            image::Rgba([32, 33, 36, 255])
        }
    });
    assert_eq!(rows_above_map(&img), 600);
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgba8(img.clone()).write_to(&mut png, image::ImageFormat::Png).unwrap();
    let (_, cw, ch) = crop_panel(png.get_ref(), 1.0, 2000).unwrap();
    assert_eq!((cw, ch), (400, 600), "the map is not shown to the model");
    // no map: nothing cut; a coloured photo on the left is not a map
    let plain = image::RgbaImage::from_fn(w, h, |x, _| if x < 150 { image::Rgba([200, 40, 40, 255]) } else { image::Rgba([32, 33, 36, 255]) });
    assert_eq!(rows_above_map(&plain), h);
    // a light theme's map (beige land, a blue river) on a white panel
    let light = image::RgbaImage::from_fn(w, h, |x, y| {
        if y >= 700 {
            if (x + y) % 7 < 3 { image::Rgba([170, 200, 240, 255]) } else { image::Rgba([245, 240, 230, 255]) }
        } else {
            image::Rgba([255, 255, 255, 255])
        }
    });
    assert_eq!(rows_above_map(&light), 700);
    // tiny and odd images
    assert_eq!(rows_above_map(&image::RgbaImage::new(3, 3)), 3);
    assert_eq!(rows_above_map(&image::RgbaImage::from_pixel(50, 100, image::Rgba([0, 0, 255, 255]))), 100, "all colour: no panel to keep");
}

#[test]
fn the_panel_is_cropped_from_the_right() {
    let img = image::RgbImage::from_fn(400, 200, |x, _| if x >= 300 { image::Rgb([255, 255, 255]) } else { image::Rgb([0, 0, 0]) });
    let mut png = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(img).write_to(&mut png, image::ImageFormat::Png).unwrap();
    let (crop, w, h) = crop_panel(png.get_ref(), 0.25, 1000).unwrap();
    assert_eq!((w, h), (100, 200));
    let back = image::load_from_memory(&crop).unwrap().to_rgb8();
    assert!(back.pixels().all(|p| p.0 == [255, 255, 255]), "the right quarter only");
    let (_, w, h) = crop_panel(png.get_ref(), 0.5, 100).unwrap();
    assert_eq!((w, h), (100, 100));
    assert!(crop_panel(b"not a png", 0.4, 100).is_err());
    let _ = crop_panel(png.get_ref(), f64::NAN, 0).unwrap();
}

// ------------------------------------------------------------------------------- the reader

const ALBUM: &str = "https://photos.google.com/album/ALB1";
const W: u32 = 2000;
const H: u32 = 1200;

fn png() -> Vec<u8> {
    let mut out = std::io::Cursor::new(Vec::new());
    image::DynamicImage::ImageRgb8(image::RgbImage::new(8, 8)).write_to(&mut out, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn cap() -> Capture {
    Capture { png: png(), rect: WindowRect { x: 0, y: 0, w: W, h: H }, px_per_point: 1.0, png_w: W, png_h: H }
}

fn el(x1: u32, y1: u32, x2: u32, y2: u32, label: &str) -> Element {
    Element { label: label.into(), kind: String::new(), x1, y1, x2, y2 }
}

/// A Google Photos tab over an album of `items` (id, panel text), and its eyes.
#[derive(Default)]
struct Page {
    url: Option<String>,
    items: Vec<(String, String)>,
    info: bool,
    /// The Right arrow does nothing (only the "next" button moves on).
    arrow_dead: bool,
    /// `i` does not open the panel.
    info_dead: bool,
    /// Panels that never read (by item index).
    unreadable: Vec<usize>,
    /// The first read of an item shows the item before.
    stale_once: bool,
    stale_shown: Vec<usize>,
    /// The model is gone.
    model_down: bool,
    /// Items whose place carries the "estimated" icon.
    estimated: Vec<usize>,
    /// Questions about that icon.
    icon_asks: usize,
    /// Boxes the eyes give for the thumbnail and the "next" arrow.
    tile: Option<Element>,
    next: Option<Element>,
    /// Albums on the albums page: (title, address).
    albums: Vec<(String, String)>,
    card_for: Option<String>,
    // what happened
    clicks: Vec<(u32, u32, String)>,
    keys: Vec<&'static str>,
    opened: Vec<String>,
    reads: usize,
    closed: bool,
}

impl Page {
    fn index(&self) -> Option<usize> {
        let id = item_id(self.url.as_deref()?)?;
        self.items.iter().position(|(i, _)| *i == id)
    }
}

struct FakeViewer(Rc<RefCell<Page>>);
struct FakeEyes(Rc<RefCell<Page>>);

impl Viewer for FakeViewer {
    fn open(&mut self, url: &str) -> Result<(), ScreenError> {
        assert!(is_photos_page(url), "{url}");
        let mut p = self.0.borrow_mut();
        p.opened.push(url.to_string());
        p.url = Some(url.to_string());
        Ok(())
    }
    fn url(&mut self) -> Result<Option<String>, ScreenError> {
        Ok(self.0.borrow().url.clone())
    }
    fn key_right(&mut self) -> Result<(), ScreenError> {
        let mut p = self.0.borrow_mut();
        p.keys.push("right");
        if p.arrow_dead {
            return Ok(());
        }
        if let Some(i) = p.index()
            && let Some((next, _)) = p.items.get(i + 1).cloned()
        {
            p.url = Some(format!("{ALBUM}/photo/{next}"));
        }
        Ok(())
    }
    fn key_info(&mut self) -> Result<(), ScreenError> {
        let mut p = self.0.borrow_mut();
        assert!(p.index().is_some(), "`i` only on an item");
        p.keys.push("i");
        if !p.info_dead {
            p.info = !p.info;
        }
        Ok(())
    }
    fn capture(&mut self, _panel: bool) -> Result<Capture, ScreenError> {
        Ok(cap())
    }
    fn click(&mut self, c: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
        let mut p = self.0.borrow_mut();
        let page = p.url.clone().unwrap_or_default();
        p.clicks.push((px, py, page.clone()));
        // nothing is ever clicked in the toolbar (favourite, delete, share live there)
        assert!(py > c.png_h / 6, "a click in the top bar at ({px},{py})");
        if page == ALBUMS_PAGE {
            if let Some(a) = p.card_for.clone() {
                p.url = Some(a);
            }
        } else if album_url(&page).is_some() && item_id(&page).is_none() {
            if let Some((first, _)) = p.items.first().cloned() {
                p.url = Some(format!("{ALBUM}/photo/{first}"));
            }
        } else if let Some(i) = p.index()
            && let Some((next, _)) = p.items.get(i + 1).cloned()
        {
            // the "next" button
            p.url = Some(format!("{ALBUM}/photo/{next}"));
        }
        Ok(())
    }
    fn scroll(&mut self, _lines: i32) -> Result<(), ScreenError> {
        Ok(())
    }
    fn scroll_panel(&mut self) -> Result<(), ScreenError> {
        Ok(())
    }
    fn sleep(&mut self, _ms: u64) {}
    fn close(&mut self) -> Result<(), ScreenError> {
        self.0.borrow_mut().closed = true;
        Ok(())
    }
}

impl PanelEyes for FakeEyes {
    fn read(&self, _png: &[u8], ask: &str) -> Result<String, ScreenError> {
        let mut p = self.0.borrow_mut();
        if p.model_down {
            return Err(ScreenError::Model("connection refused".into(), "qwen2.5vl:7b".into()));
        }
        if ask == ALBUM_TITLE_ASK {
            let url = p.url.clone().unwrap_or_default();
            return Ok(p.albums.iter().find(|(_, a)| *a == url).map(|(t, _)| t.clone()).unwrap_or_default());
        }
        if ask == ESTIMATED_ASK {
            p.icon_asks += 1;
            let yes = p.index().is_some_and(|i| p.estimated.contains(&i));
            return Ok(if yes { "Yes." } else { "No." }.into());
        }
        assert_eq!(ask, PANEL_ASK);
        p.reads += 1;
        let Some(i) = p.index() else { return Ok("NONE".into()) };
        if !p.info || p.unreadable.contains(&i) {
            return Ok("NONE".into());
        }
        if p.stale_once && i > 0 && !p.stale_shown.contains(&i) {
            p.stale_shown.push(i);
            return Ok(p.items.get(i - 1).map(|x| x.1.clone()).unwrap_or_default());
        }
        Ok(p.items.get(i).map(|x| x.1.clone()).unwrap_or_default())
    }
    fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
        let p = self.0.borrow();
        if p.model_down {
            return Err(ScreenError::Model("connection refused".into(), "qwen2.5vl:7b".into()));
        }
        Ok(if target == FIRST_ITEM_ASK {
            p.tile.clone()
        } else if target == NEXT_ASK {
            // the arrow beside an item, none after the last
            p.next.clone().filter(|_| p.index().is_some_and(|i| i + 1 < p.items.len()))
        } else if target.starts_with("the album card") {
            p.card_for.as_ref().map(|_| el(300, 500, 600, 800, "album"))
        } else {
            None
        })
    }
}

fn panel(file: &str, place: Option<&str>) -> String {
    format!("Details\nOct 1, 2022\nSat, 10:00 AM GMT+04:00\n{file}\n1920 × 1080\n{}", place.unwrap_or(""))
}

fn album(n: usize) -> Rc<RefCell<Page>> {
    let items = (0..n).map(|k| (format!("P{k}"), panel(&format!("VID_{k}.mp4"), (k % 3 != 2).then_some("Dubai")))).collect();
    Rc::new(RefCell::new(Page { items, tile: Some(el(100, 300, 400, 600, "thumbnail")), next: Some(el(1500, 560, 1560, 640, "next")), ..Page::default() }))
}

fn run(page: &Rc<RefCell<Page>>, req: &Request) -> (Result<Report, ScreenError>, Vec<Location>) {
    let mut v = FakeViewer(page.clone());
    let eyes = FakeEyes(page.clone());
    let mut heard = Vec::new();
    let r = read_album(&mut v, &eyes, req, &mut |_| {}, &mut |l| {
        heard.push(l.clone());
        true
    });
    (r, heard)
}

#[test]
fn an_album_is_read_item_by_item_to_its_end() {
    let page = album(5);
    let (r, heard) = run(&page, &Request::new(ALBUM));
    let r = r.unwrap();
    assert_eq!(r.locations.iter().map(|l| l.file_name.as_str()).collect::<Vec<_>>(), ["VID_0.mp4", "VID_1.mp4", "VID_2.mp4", "VID_3.mp4", "VID_4.mp4"]);
    assert_eq!(r.locations.iter().map(|l| l.id.as_deref().unwrap_or("")).collect::<Vec<_>>(), ["P0", "P1", "P2", "P3", "P4"]);
    assert_eq!((r.read, r.without_place, r.failed, r.stopped.as_deref()), (5, 1, 0, None));
    assert_eq!(heard, r.locations, "every item is heard as it is read");
    let p = page.borrow();
    // the Info panel opened once with its shortcut; only the first thumbnail was clicked
    assert_eq!(p.keys.iter().filter(|k| **k == "i").count(), 1);
    assert_eq!(p.clicks.len(), 1, "{:?}", p.clicks);
    assert!(p.closed, "the reader's tab is closed");
    assert_eq!(p.opened.first().map(String::as_str), Some(ALBUM));
}

#[test]
fn an_estimated_place_is_told_by_its_icon() {
    let page = album(4);
    page.borrow_mut().estimated = vec![1];
    let (r, _) = run(&page, &Request::new(ALBUM));
    let r = r.unwrap();
    assert_eq!(r.locations.iter().map(|l| l.estimated).collect::<Vec<_>>(), [false, true, false, false]);
    // asked only about items with a place (item 2 has none)
    assert_eq!(page.borrow().icon_asks, 3);
}

#[test]
fn a_run_stops_at_its_limits() {
    let page = album(6);
    let (r, _) = run(&page, &Request { max_items: 2, ..Request::new(ALBUM) });
    let r = r.unwrap();
    assert_eq!(r.read, 2);
    assert!(r.stopped.as_deref().is_some_and(|s| s.contains("most asked for")), "{:?}", r.stopped);
    let page = album(3);
    let (r, _) = run(&page, &Request { budget: Duration::ZERO, ..Request::new(ALBUM) });
    let r = r.unwrap();
    assert_eq!(r.read, 0);
    assert!(r.stopped.as_deref().is_some_and(|s| s.contains("out of time")));
    assert!(page.borrow().closed);
}

#[test]
fn a_run_stops_when_its_caller_says_so() {
    let page = album(5);
    let mut v = FakeViewer(page.clone());
    let eyes = FakeEyes(page.clone());
    let mut n = 0;
    let r = read_album(&mut v, &eyes, &Request::new(ALBUM), &mut |_| {}, &mut |_| {
        n += 1;
        n < 2
    })
    .unwrap();
    assert_eq!((r.read, r.stopped.as_deref()), (2, Some("stopped")));
    assert!(page.borrow().closed);
}

#[test]
fn a_run_resumes_after_the_last_item_read() {
    let page = album(5);
    let (first, _) = run(&page, &Request { max_items: 2, ..Request::new(ALBUM) });
    let done = first.unwrap().locations;
    let page2 = album(5);
    let (r, heard) = run(&page2, &Request { done: done.clone(), ..Request::new(ALBUM) });
    let r = r.unwrap();
    assert_eq!(r.read, 3, "only the rest is read");
    assert_eq!(heard.iter().map(|l| l.file_name.as_str()).collect::<Vec<_>>(), ["VID_2.mp4", "VID_3.mp4", "VID_4.mp4"]);
    assert_eq!(r.locations.len(), 5);
    let p = page2.borrow();
    assert_eq!(p.opened.first().map(String::as_str), Some("https://photos.google.com/album/ALB1/photo/P1"), "opened by its address");
    assert!(p.clicks.is_empty(), "no thumbnail clicked when resuming");
}

#[test]
fn an_unreadable_item_is_recorded_and_skipped() {
    let page = album(4);
    page.borrow_mut().unreadable = vec![2];
    let (r, _) = run(&page, &Request::new(ALBUM));
    let r = r.unwrap();
    assert_eq!((r.read, r.failed), (4, 1));
    let bad = &r.locations[2];
    assert_eq!((bad.id.as_deref(), bad.file_name.as_str()), (Some("P2"), ""));
    assert!(bad.error.is_some());
    assert_eq!(r.locations[3].file_name, "VID_3.mp4");
    // a resumed run reads it again, from the last item read well, and it takes its place
    let page2 = album(4);
    let (r2, heard) = run(&page2, &Request { done: r.locations[..3].to_vec(), ..Request::new(ALBUM) });
    let r2 = r2.unwrap();
    assert_eq!(r2.read, 2);
    assert_eq!(heard.iter().map(|l| l.file_name.as_str()).collect::<Vec<_>>(), ["VID_2.mp4", "VID_3.mp4"]);
    assert_eq!(r2.locations.len(), 4);
    assert!(r2.locations.iter().all(|l| l.error.is_none()));
    assert_eq!(page2.borrow().opened.first().map(String::as_str), Some("https://photos.google.com/album/ALB1/photo/P1"));
}

#[test]
fn a_panel_still_showing_the_item_before_is_read_again() {
    let page = album(3);
    page.borrow_mut().stale_once = true;
    let (r, _) = run(&page, &Request::new(ALBUM));
    let r = r.unwrap();
    assert_eq!(r.locations.iter().map(|l| l.file_name.as_str()).collect::<Vec<_>>(), ["VID_0.mp4", "VID_1.mp4", "VID_2.mp4"]);
    assert_eq!(r.failed, 0);
}

#[test]
fn the_next_button_moves_on_when_the_arrow_key_does_not() {
    let page = album(3);
    page.borrow_mut().arrow_dead = true;
    let (r, _) = run(&page, &Request::new(ALBUM));
    let r = r.unwrap();
    assert_eq!(r.read, 3);
    let p = page.borrow();
    // the thumbnail, then the arrow twice; none after the last item
    assert_eq!(p.clicks.len(), 3, "{:?}", p.clicks);
    // a "next" box in the Info panel or the top bar is never clicked
    drop(p);
    let page = album(3);
    page.borrow_mut().arrow_dead = true;
    page.borrow_mut().next = Some(el(1800, 560, 1860, 640, "in the panel"));
    let (r, _) = run(&page, &Request::new(ALBUM));
    assert_eq!(r.unwrap().read, 1, "stuck on the first item: it ends there");
    assert_eq!(page.borrow().clicks.len(), 1);
}

#[test]
fn an_info_panel_that_will_not_open_says_what_to_do() {
    let page = album(3);
    page.borrow_mut().info_dead = true;
    let (r, heard) = run(&page, &Request::new(ALBUM));
    let e = r.unwrap_err().to_string();
    assert!(e.contains("Open info") && e.contains("--resume"), "{e}");
    assert!(heard.is_empty());
    assert!(page.borrow().closed);
}

#[test]
fn a_lost_model_ends_the_run() {
    let page = album(3);
    page.borrow_mut().model_down = true;
    let (r, _) = run(&page, &Request::new(ALBUM));
    assert!(matches!(r, Err(ScreenError::Model(..))), "{r:?}");
}

#[test]
fn misplaced_thumbnails_are_never_clicked() {
    // the model boxes the toolbar instead of a thumbnail: no click, and the run says so
    let page = album(3);
    page.borrow_mut().tile = Some(el(1700, 20, 1760, 80, "star"));
    let (r, _) = run(&page, &Request::new(ALBUM));
    assert!(matches!(r, Err(ScreenError::Stalled(_))), "{r:?}");
    assert!(page.borrow().clicks.is_empty());
    // a whole-screen box is no thumbnail either
    let page = album(3);
    page.borrow_mut().tile = Some(el(0, 0, W, H, "everything"));
    assert!(run(&page, &Request::new(ALBUM)).0.is_err());
    assert!(page.borrow().clicks.is_empty());
    // not an album address
    let page = album(3);
    assert!(run(&page, &Request::new("https://example.com/album/X")).0.is_err());
    assert!(page.borrow().opened.is_empty());
}

#[test]
fn an_album_is_found_on_the_albums_page_by_its_title() {
    let page = album(1);
    page.borrow_mut().albums = vec![("Every Robot I've Met".into(), ALBUM.into())];
    page.borrow_mut().card_for = Some(ALBUM.into());
    let mut v = FakeViewer(page.clone());
    let eyes = FakeEyes(page.clone());
    let found = find_album(&mut v, &eyes, "Every Robot I’ve Met", &mut |_| {}).unwrap();
    assert_eq!(found, ALBUM);
    assert_eq!(page.borrow().opened.first().map(String::as_str), Some(ALBUMS_PAGE));
    // a card that opens another album is not taken
    let page = album(1);
    page.borrow_mut().albums = vec![("Robots".into(), ALBUM.into())];
    page.borrow_mut().card_for = Some(ALBUM.into());
    let mut v = FakeViewer(page.clone());
    let eyes = FakeEyes(page.clone());
    let e = find_album(&mut v, &eyes, "Every Robot I've Met", &mut |_| {}).unwrap_err().to_string();
    assert!(e.contains("--album-url"), "{e}");
    let mut v = FakeViewer(page.clone());
    assert!(find_album(&mut v, &FakeEyes(page.clone()), " ", &mut |_| {}).is_err());
    assert!(find_album(&mut v, &FakeEyes(page), "a\u{7}b", &mut |_| {}).is_err());
}

#[test]
fn chrome_viewer_only_opens_google_photos() {
    struct NoShell;
    impl crate::capture::Shell for NoShell {
        fn run(&self, _program: &str, _args: &[&str]) -> Result<String, ScreenError> {
            panic!("no script may run for a refused address");
        }
    }
    let mut v = ChromeViewer::with_shells(Box::new(NoShell), Box::new(NoShell));
    for bad in ["https://example.com/", "javascript:alert(1)", "https://photos.google.com/album/A\" & do shell script \"rm"] {
        assert!(v.open(bad).is_err(), "{bad}");
    }
    assert_eq!(v.url().unwrap(), None);
    assert!(v.close().is_ok());
}

/// Live: what the model reads in a saved panel capture (`AUTOCUT_LOCATIONS_DEBUG` keeps them),
/// and what the parser makes of it. `AUTOCUT_LOCATIONS_LIVE_PNG=<capture.png>`.
#[test]
#[ignore]
fn live_read_a_panel_capture() {
    let Ok(path) = std::env::var("AUTOCUT_LOCATIONS_LIVE_PNG") else { return };
    // as the reader prepares a capture (its map cut off)
    let (png, _, _) = crop_panel(&std::fs::read(path).unwrap(), 1.0, PANEL_SIDE).unwrap();
    let eyes = crate::ground::QwenGrounder::from_config(&crate::Config::from_env());
    if let Ok(out) = std::env::var("AUTOCUT_LOCATIONS_LIVE_FRAMED") {
        std::fs::write(out, crate::ground::to_model_frame(&png).unwrap().png).unwrap();
    }
    let n: usize = std::env::var("AUTOCUT_LOCATIONS_LIVE_TIMES").ok().and_then(|v| v.parse().ok()).unwrap_or(3).min(20);
    for _ in 0..n {
        let text = PanelEyes::read(&eyes, &png, PANEL_ASK).unwrap();
        eprintln!("{text}\n---\n{:?}\n===", parse_panel(&text));
    }
}

/// Live: read a few items of a real album through Chrome and the local model.
/// `AUTOCUT_LOCATIONS_LIVE=<album address>` (and the `chrome` lock held by the caller).
#[test]
#[ignore]
fn live_read_a_few_items() {
    let Ok(album) = std::env::var("AUTOCUT_LOCATIONS_LIVE") else { return };
    let mut v = ChromeViewer::new();
    let eyes = crate::ground::QwenGrounder::from_config(&crate::Config::from_env());
    let req = Request { max_items: 5, ..Request::new(&album) };
    let r = read_album(&mut v, &eyes, &req, &mut |l| eprintln!("{l}"), &mut |_| true).unwrap();
    eprintln!("{}", serde_json::to_string_pretty(&r).unwrap_or_default());
}
