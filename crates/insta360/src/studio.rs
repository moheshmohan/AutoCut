//! Downloads from the Insta360 cloud through Insta360 Studio, the way a person would: the cloud
//! has no public API, and Studio (signed in by the user) is the way to it. Studio's media list is
//! drawn by the app itself (no accessibility tree), so this reads the screen with the local
//! vision model (`filmcraft_screenagent`: Qwen2.5-VL through Ollama) and clicks like a person,
//! inside Studio's window only.
//!
//! For each wanted recording:
//!
//! 1. Studio in front, **Cloud Media** chosen in its sidebar, the list in **List View** (full
//!    file names and dates; the toggle left of the sort menu, chosen once and put back after).
//! 2. The rows read; the wanted name found among them, scrolling the list (at most
//!    [`MAX_SCROLLS`] screens) when it is not on this one — towards it whichever way Studio
//!    sorts the list (newest first, as it does by default, or oldest first: read off the rows).
//! 3. Right-click its row → **Download** in the menu (the menu item is checked to say Download:
//!    the same menu ends with "Remove from Insta360 Cloud", which is never clicked).
//! 4. The Download dialog's file name read: the wanted one → **OK** (Studio's own download
//!    folder, which this does not change); anything else → **Cancel**.
//!    Studio remembers the folder last chosen in that dialog: when another tool pointed it
//!    elsewhere (seen 2026-11-19: a catalogue job's folder, since deleted), files never arrive in
//!    the folder watched here, and the wait's message says to choose it again with Browse.
//! 5. Wait for the file in the download folder: complete when it ends with the camera's trailer.
//!
//! The mouse and keyboard are one at a time across agents: the `chrome` lock (`filmcraft_locks`)
//! is held while Studio is driven. Every step is budgeted; nothing is clicked outside the window.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use filmcraft_screenagent::ScreenError;
use filmcraft_screenagent::act::Act;
use filmcraft_screenagent::capture::Capture;
use filmcraft_screenagent::ground::{Element, Ground};

/// The application driven.
pub const APP: &str = "Insta360 Studio";
/// Most list screens scrolled through looking for one name.
pub const MAX_SCROLLS: usize = 60;
/// Wheel lines a list screen scrolls.
const SCROLL_LINES: i32 = 6;

/// Reads text off a screenshot (the vision model's transcription).
pub trait ReadText {
    fn read_text(&self, png: &[u8], ask: &str) -> Result<String, ScreenError>;
}

impl ReadText for filmcraft_screenagent::ground::QwenGrounder {
    fn read_text(&self, png: &[u8], ask: &str) -> Result<String, ScreenError> {
        filmcraft_screenagent::ground::QwenGrounder::read_text(self, png, ask)
    }
}

/// What happened to one wanted recording.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Outcome {
    pub name: String,
    /// Studio was asked to download it.
    pub requested: bool,
    /// Where it was found finished, when it was waited for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub why: Option<String>,
}

/// The stem of a name as Studio lists it (`VID_20260913_114003_00_093.insv` → its stem).
pub fn stem_of(name: &str) -> String {
    let n = name.trim();
    n.rsplit_once('.').map_or(n, |(b, _)| b).to_string()
}

/// The camera file name in what the model read off a row (`VID_20260913_114003_00_093.insv
/// 09/13/2026 11:22` → its stem), upper case; the whole label's stem when there is none.
pub fn name_in(label: &str) -> String {
    let is_name = |w: &&str| {
        let u = w.to_ascii_uppercase();
        u.starts_with("VID_") || u.starts_with("IMG_") || u.starts_with("LRV_")
    };
    label.split_whitespace().find(is_name).map_or_else(|| stem_of(label), stem_of).to_ascii_uppercase()
}

/// Whether a name read off the screen is the wanted recording: the same stem, or — a wanted
/// short name like Studio's grid shows (`VID_093`) — the same short name.
pub fn same_recording(read: &str, wanted: &str) -> bool {
    let (r, w) = (name_in(read), stem_of(wanted).to_ascii_uppercase());
    r == w || (w.split('_').count() == 2 && crate::library::short_name(&r) == w)
}

/// The row of `wanted` among the elements the model listed (full stems are certain; a short name
/// takes the first row that has it).
pub fn row_for<'a>(rows: &'a [Element], wanted: &str) -> Option<&'a Element> {
    rows.iter().find(|r| same_recording(&r.label, wanted))
}

/// The camera names among the rows, top to bottom as they are on screen.
fn names_down(rows: &[Element]) -> Vec<String> {
    let mut named: Vec<(u32, String)> =
        rows.iter().map(|r| (r.center().1, name_in(&r.label))).filter(|(_, n)| n.starts_with("VID_") || n.starts_with("IMG_")).collect();
    named.sort_by_key(|(y, _)| *y);
    named.into_iter().map(|(_, n)| n).collect()
}

/// What a camera name sorts by: its date and time (the names begin with them, so they sort as
/// the recordings were made).
fn time_key(name: &str) -> &str {
    name.get(4..19).unwrap_or(name)
}

/// Whether the rows on screen are all before (`Less`) or all after (`Greater`) the wanted name
/// in name order — the wanted one lies beyond them — or neither (it should be on screen, or the
/// order is not by name). One row of four or more read out of line (a name half hidden under a
/// tooltip, misread) does not decide it.
pub fn direction(rows: &[Element], wanted: &str) -> std::cmp::Ordering {
    let w = stem_of(wanted).to_ascii_uppercase();
    let names = names_down(rows);
    if names.is_empty() {
        return std::cmp::Ordering::Equal;
    }
    let before = names.iter().filter(|n| time_key(n) < time_key(&w)).count();
    let after = names.iter().filter(|n| time_key(n) > time_key(&w)).count();
    let stray = usize::from(names.len() >= 4);
    if after == 0 || (after <= stray && before >= 3) {
        std::cmp::Ordering::Less
    } else if before == 0 || (before <= stray && after >= 3) {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

/// How Studio sorts the list, read off the rows on screen: `Some(true)` newest at the top (its
/// default), `Some(false)` oldest at the top, `None` when the rows do not tell (fewer than two
/// names, or as many steps one way as the other).
pub fn newest_first(rows: &[Element]) -> Option<bool> {
    let names = names_down(rows);
    let (mut down, mut up) = (0usize, 0usize);
    for pair in names.windows(2) {
        if let [a, b] = pair {
            match time_key(a).cmp(time_key(b)) {
                std::cmp::Ordering::Greater => down += 1,
                std::cmp::Ordering::Less => up += 1,
                std::cmp::Ordering::Equal => {}
            }
        }
    }
    match down.cmp(&up) {
        std::cmp::Ordering::Greater => Some(true),
        std::cmp::Ordering::Less => Some(false),
        std::cmp::Ordering::Equal => None,
    }
}

/// Wheel lines towards a wanted name that lies beyond the rows on screen: `beyond` as
/// [`direction`] says (`Less`: every row is before it), in a list sorted `newest_first` or not.
/// Positive scrolls down.
pub fn lines_towards(beyond: std::cmp::Ordering, newest_first: bool) -> i32 {
    match (beyond, newest_first) {
        // every row is older than the wanted one: it is below in an oldest-first list, above in a newest-first one
        (std::cmp::Ordering::Less, false) | (std::cmp::Ordering::Greater, true) => SCROLL_LINES,
        (std::cmp::Ordering::Less, true) | (std::cmp::Ordering::Greater, false) => -SCROLL_LINES,
        (std::cmp::Ordering::Equal, _) => 0,
    }
}

/// Whether a context-menu element is the Download item (and certainly not Remove).
pub fn is_download_item(e: &Element) -> bool {
    let l = e.label.to_ascii_lowercase();
    l.contains("download") && !l.contains("remove") && !l.contains("delete")
}

/// The driver: hands, eyes and a reader, plus where Studio puts downloads.
pub struct Driver<'a> {
    pub act: &'a mut dyn Act,
    pub ground: &'a dyn Ground,
    pub read: &'a dyn ReadText,
    pub download_dir: PathBuf,
    /// How long one download may take to finish.
    pub file_timeout: Duration,
    pub progress: &'a mut dyn FnMut(&str),
    pub cancelled: &'a dyn Fn() -> bool,
}

impl Driver<'_> {
    fn capture(&mut self) -> Result<Capture, ScreenError> {
        self.act.capture()
    }

    fn click(&mut self, cap: &Capture, e: &Element) -> Result<(), ScreenError> {
        let (x, y) = e.center();
        self.act.click_png(cap, x, y)
    }

    /// Cloud Media in front, in List View. Returns whether List View was switched on here.
    fn prepare(&mut self) -> Result<bool, ScreenError> {
        self.act.activate()?;
        self.act.sleep(600);
        let cap = self.capture()?;
        let cloud = self.ground.find(&cap.png, "the 'Cloud Media' item in the left sidebar under 'Media Library'")?;
        let Some(cloud) = cloud.filter(|e| e.label.to_ascii_lowercase().contains("cloud")) else {
            return Err(ScreenError::Stalled("Insta360 Studio does not show Cloud Media (is it open on its home screen, signed in?)".into()));
        };
        self.click(&cap, &cloud)?;
        // the pointer off the sidebar: left there, its tooltip covers two rows of the list
        self.act.scroll(0)?;
        self.act.sleep(1500);
        let cap = self.capture()?;
        if self.ground.find(&cap.png, "the 'File Name' column header of the file list")?.is_some() {
            return Ok(false);
        }
        // The view toggle is the leftmost of the small icons in the row at the top right of the
        // grid; two along is the trash. Only a clear row of three to five icons is trusted, its
        // leftmost clicked once, and the list checked for after.
        let icons = self
            .ground
            .inventory(&cap.png, "Locate each small icon button in the row at the top right of the file grid (view toggle, sort menu, delete, refresh).")?;
        let Some(toggle) = leftmost_of_row(&icons) else {
            return Err(ScreenError::Stalled("cannot tell Studio's view toggle from its other buttons: switch Cloud Media to List View (the leftmost icon above the files), then run this again".into()));
        };
        self.click(&cap, &toggle)?;
        self.act.sleep(1200);
        let cap = self.capture()?;
        if self.ground.find(&cap.png, "the 'File Name' column header of the file list")?.is_none() {
            return Err(ScreenError::Stalled(
                "Studio did not switch to List View: switch Cloud Media to List View (the leftmost icon above the files), then run this again".into(),
            ));
        }
        Ok(true)
    }

    fn rows(&self, cap: &Capture) -> Result<Vec<Element>, ScreenError> {
        self.ground.inventory(&cap.png, "Locate every row of the file list (each row shows a file name such as VID_20260913_114003_00_093.insv and a date); the label of each row is its exact file name.")
    }

    /// Asks Studio to download `wanted`; `Ok(())` once the dialog's OK was pressed.
    fn request(&mut self, wanted: &str) -> Result<(), String> {
        let mut last_dir = std::cmp::Ordering::Equal;
        // Studio lists the newest first unless the user sorted it otherwise; each screen's rows say
        let mut newest = true;
        for scrolls in 0..=MAX_SCROLLS {
            if (self.cancelled)() {
                return Err("cancelled".into());
            }
            let cap = self.capture().map_err(|e| e.to_string())?;
            let rows = self.rows(&cap).map_err(|e| e.to_string())?;
            if let Some(row) = row_for(&rows, wanted).cloned() {
                return self.download_row(&cap, &row, wanted);
            }
            let dir = direction(&rows, wanted);
            newest = newest_first(&rows).unwrap_or(newest);
            // on screen by order, yet not read: one more look at a smaller scroll, then give up
            let lines = match dir {
                std::cmp::Ordering::Equal if scrolls == 0 => SCROLL_LINES,
                std::cmp::Ordering::Equal => return Err(format!("{wanted} is not in Studio's Cloud Media list")),
                _ => lines_towards(dir, newest),
            };
            if dir != std::cmp::Ordering::Equal && last_dir != std::cmp::Ordering::Equal && dir != last_dir {
                // passed it both ways: it is not there
                return Err(format!("{wanted} is not in Studio's Cloud Media list"));
            }
            last_dir = dir;
            self.act.scroll(lines).map_err(|e| e.to_string())?;
            self.act.sleep(700);
        }
        Err(format!("{wanted} not found after {MAX_SCROLLS} screens of the list"))
    }

    fn download_row(&mut self, cap: &Capture, row: &Element, wanted: &str) -> Result<(), String> {
        let (x, y) = row.center();
        self.act.right_click_png(cap, x, y).map_err(|e| e.to_string())?;
        self.act.sleep(700);
        let menu = self.capture().map_err(|e| e.to_string())?;
        let item = self.ground.find(&menu.png, "the 'Download' item of the context menu").map_err(|e| e.to_string())?;
        let Some(item) = item.filter(is_download_item) else {
            let _ = self.act.key_escape();
            return Err("Studio's menu had no Download item".into());
        };
        self.click(&menu, &item).map_err(|e| e.to_string())?;
        self.act.sleep(1200);
        let dialog = self.capture().map_err(|e| e.to_string())?;
        let shown = self
            .read
            .read_text(&dialog.png, "The 'Download' dialog shows one file name ending in .insv (or .insp). Reply with that file name only.")
            .map_err(|e| e.to_string())?;
        let ok = same_recording(shown.trim(), wanted);
        if !ok && !shown.to_ascii_lowercase().contains(".ins") {
            // no dialog: Studio downloads straight to its default folder (its "Set as default
            // download path" was ticked): the file wait tells
            return Ok(());
        }
        let button = if ok { "the yellow 'OK' button of the Download dialog" } else { "the 'Cancel' button of the Download dialog" };
        let b = self.ground.find(&dialog.png, button).map_err(|e| e.to_string())?;
        let Some(b) = b else {
            let _ = self.act.key_escape();
            return Err("cannot find the Download dialog's buttons".into());
        };
        self.click(&dialog, &b).map_err(|e| e.to_string())?;
        self.act.sleep(800);
        if ok { Ok(()) } else { Err(format!("Studio offered {} instead of {wanted}; cancelled", shown.trim())) }
    }

    /// Requests every wanted recording, then waits for each file (when `wait`).
    pub fn download(&mut self, wanted: &[String], wait: bool) -> Result<Vec<Outcome>, ScreenError> {
        let switched = self.prepare()?;
        let mut out = Vec::new();
        for w in wanted {
            (self.progress)(&format!("asking Studio for {w}"));
            match self.request(w) {
                Ok(()) => out.push(Outcome { name: w.clone(), requested: true, path: None, why: None }),
                Err(why) => out.push(Outcome { name: w.clone(), requested: false, path: None, why: Some(why) }),
            }
        }
        if switched {
            // (not switched back: the toggle sits two icons from the trash)
            (self.progress)("Studio's Cloud Media is left in List View");
        }
        if wait {
            for o in out.iter_mut().filter(|o| o.requested) {
                (self.progress)(&format!("waiting for {} to download", o.name));
                match wait_for_file(&self.download_dir, &o.name, self.file_timeout, self.cancelled) {
                    Some(p) => o.path = Some(p),
                    None => {
                        o.why = Some(format!(
                            "not finished downloading in time (Studio's task centre shows it; if nothing arrived, the Download dialog's 'Download to' folder is not {}: choose it there with Browse, then run this again)",
                            self.download_dir.display()
                        ));
                    }
                }
            }
        }
        Ok(out)
    }
}

/// Waits for recording `name` to finish downloading under `dir`: its file ends with the camera's
/// trailer and has stopped growing.
pub fn wait_for_file(dir: &Path, name: &str, timeout: Duration, cancelled: &dyn Fn() -> bool) -> Option<PathBuf> {
    let stem = stem_of(name);
    let start = Instant::now();
    let mut last: Option<(PathBuf, u64)> = None;
    while start.elapsed() < timeout && !cancelled() {
        let recs = crate::library::scan(&[dir.to_path_buf()]);
        if let Some(r) = recs.iter().find(|r| r.stem.eq_ignore_ascii_case(&stem)) {
            if r.complete && last.as_ref().is_some_and(|(p, b)| *p == r.path && *b == r.bytes) {
                return Some(r.path.clone());
            }
            last = Some((r.path.clone(), r.bytes));
        }
        std::thread::sleep(Duration::from_secs(2));
    }
    None
}

/// The leftmost of a row of three to five small icons (their centres level within a quarter of an
/// icon), not one that says delete; `None` when the elements are not such a row.
pub fn leftmost_of_row(icons: &[Element]) -> Option<Element> {
    if !(3..=5).contains(&icons.len()) {
        return None;
    }
    let size = icons.iter().map(|e| e.y2.saturating_sub(e.y1).max(e.x2.saturating_sub(e.x1))).max()?.max(1);
    let ys: Vec<u32> = icons.iter().map(|e| e.center().1).collect();
    let (lo, hi) = (*ys.iter().min()?, *ys.iter().max()?);
    if hi - lo > size / 4 + 2 || icons.iter().any(|e| e.x2.saturating_sub(e.x1) > size.max(8) * 3) {
        return None;
    }
    let first = icons.iter().min_by_key(|e| e.x1)?;
    let l = first.label.to_ascii_lowercase();
    if l.contains("delete") || l.contains("trash") || l.contains("remove") {
        return None;
    }
    Some(first.clone())
}

/// Whether Studio can be driven now: macOS, Insta360 Studio running, the vision model served by
/// Ollama. `Err` says what to do.
pub fn ready() -> Result<(), String> {
    if !cfg!(target_os = "macos") {
        return Err("driving Insta360 Studio needs macOS".into());
    }
    let shell = filmcraft_screenagent::capture::OsShell;
    use filmcraft_screenagent::capture::Shell as _;
    let running = shell
        // (System Events lists Studio's process under its executable's name: ask by the app's)
        .run("osascript", &["-e", "application \"Insta360 Studio\" is running"])
        .map_err(|e| format!("cannot ask macOS which apps run: {e}"))?;
    if running.trim() != "true" {
        return Err("open Insta360 Studio (signed in to your Insta360 account) first".into());
    }
    let cfg = filmcraft_screenagent::Config::from_env();
    let g = filmcraft_screenagent::ground::QwenGrounder::from_config(&cfg);
    match g.model_available() {
        Ok(true) => Ok(()),
        Ok(false) => Err(format!("the vision model is not installed: `ollama pull {}`", cfg.model)),
        Err(e) => Err(e.to_string()),
    }
}

/// Downloads `names` through Studio (macOS, the real screen): see [`Driver`].
pub fn download(
    names: &[String],
    wait: bool,
    dir: &Path,
    timeout: Duration,
    progress: &mut dyn FnMut(&str),
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<Outcome>, ScreenError> {
    let cfg = filmcraft_screenagent::Config::from_env();
    let grounder = filmcraft_screenagent::ground::QwenGrounder::from_config(&cfg);
    let mut actor = filmcraft_screenagent::act::MacActor::new(APP, Box::new(filmcraft_screenagent::capture::OsShell));
    let mut d = Driver { act: &mut actor, ground: &grounder, read: &grounder, download_dir: dir.to_path_buf(), file_timeout: timeout, progress, cancelled };
    d.download(names, wait)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn el(label: &str) -> Element {
        Element { label: label.into(), kind: "row".into(), x1: 0, y1: 0, x2: 10, y2: 10 }
    }

    fn icon(label: &str, x: u32, y: u32) -> Element {
        Element { label: label.into(), kind: "icon".into(), x1: x, y1: y, x2: x + 20, y2: y + 20 }
    }

    #[test]
    fn the_view_toggle_is_the_leftmost_of_a_clear_row() {
        let row = [icon("sort", 1040, 266), icon("view toggle", 1004, 265), icon("delete", 1080, 266), icon("refresh", 1110, 265)];
        assert_eq!(leftmost_of_row(&row).unwrap().label, "view toggle");
        // a row whose leftmost says delete, a ragged row, too few: not trusted
        assert!(leftmost_of_row(&[icon("trash", 900, 265), icon("sort", 1040, 266), icon("x", 1080, 266)]).is_none());
        assert!(leftmost_of_row(&[icon("a", 1004, 265), icon("b", 1040, 300), icon("c", 1080, 266)]).is_none());
        assert!(leftmost_of_row(&row[..2]).is_none());
    }

    #[test]
    fn names_are_read_out_of_row_labels() {
        assert_eq!(name_in("VID_20260913_112239_00_091.insv 09/13/2026 11:22"), "VID_20260913_112239_00_091");
        assert_eq!(name_in("  vid_20260913_112239_00_091.INSV"), "VID_20260913_112239_00_091");
        assert_eq!(name_in("File Name"), "FILE NAME");
    }

    #[test]
    fn rows_and_directions() {
        let rows = vec![el("VID_20260913_112239_00_091.insv"), el("VID_20260913_113306_00_092.insv")];
        assert!(row_for(&rows, "VID_20260913_113306_00_092").is_some());
        assert!(row_for(&rows, "vid_20260913_113306_00_092.INSV").is_some());
        assert!(row_for(&rows, "VID_20260913_114003_00_093").is_none());
        assert!(row_for(&rows, "VID_092").is_some_and(|r| r.label.contains("113306")));
        assert!(row_for(&rows, "VID_093").is_none());
        assert_eq!(direction(&rows, "VID_20260919_111418_00_095"), std::cmp::Ordering::Less);
        assert_eq!(direction(&rows, "VID_20250101_000000_00_001"), std::cmp::Ordering::Greater);
        assert_eq!(direction(&rows, "VID_20260913_113000_00_099"), std::cmp::Ordering::Equal);
        assert_eq!(direction(&[], "VID_20260913_113000_00_099"), std::cmp::Ordering::Equal);
    }

    fn row(label: &str, y: u32) -> Element {
        Element { label: label.into(), kind: "row".into(), x1: 0, y1: y, x2: 900, y2: y + 40 }
    }

    #[test]
    fn the_sort_order_is_read_off_the_rows_and_one_stray_row_does_not_decide() {
        use std::cmp::Ordering::{Equal, Greater, Less};
        let newest = [
            row("VID_20261108_151520_00_104.insv", 0),
            row("VID_20261108_151037_00_103.insv", 50),
            row("VID_20261108_150942_00_102.insv", 100),
            row("VID_20261108_145123_00_099.insv", 150),
        ];
        let oldest: Vec<Element> = newest.iter().rev().enumerate().map(|(i, r)| row(&r.label, i as u32 * 50)).collect();
        assert_eq!(newest_first(&newest), Some(true));
        assert_eq!(newest_first(&oldest), Some(false));
        assert_eq!(newest_first(&newest[..1]), None);
        // the model lists rows in any order: their place on screen decides
        let shuffled = [newest[2].clone(), newest[0].clone(), newest[3].clone(), newest[1].clone()];
        assert_eq!(newest_first(&shuffled), Some(true));
        // an older recording lies below a newest-first list and above an oldest-first one
        assert_eq!(direction(&newest, "VID_20260716_120437_00_060"), Greater);
        assert!(lines_towards(Greater, true) > 0 && lines_towards(Greater, false) < 0);
        assert!(lines_towards(Less, true) < 0 && lines_towards(Less, false) > 0);
        assert_eq!(lines_towards(Equal, true), 0);
        // one row misread as an old name among four newer ones does not put the wanted one on screen
        let mut stray = newest.to_vec();
        stray.push(row("VID_20250108_150623_00_101.insv", 200));
        assert_eq!(direction(&stray, "VID_20260716_120437_00_060"), Greater);
        // but two do: then it should be on screen
        stray.push(row("VID_20250108_150700_00_100.insv", 250));
        assert_eq!(direction(&stray, "VID_20260716_120437_00_060"), Equal);
        assert_eq!(direction(&newest[..2], "VID_20261108_151100_00_000"), Equal);
        assert_eq!(direction(&oldest, "VID_20270101_000000_00_001"), Less);
    }

    /// A Cloud Media list of `names`, `SHOWN` rows on screen, scrolled by the wheel.
    struct List {
        names: Vec<String>,
        top: std::cell::Cell<usize>,
        asked: std::cell::RefCell<Vec<String>>,
        menu: std::cell::Cell<bool>,
    }
    const SHOWN: usize = 7;
    struct Hands<'a>(&'a List);
    impl Act for Hands<'_> {
        fn activate(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn click_png(&mut self, _: &Capture, _: u32, _: u32) -> Result<(), ScreenError> {
            self.0.menu.set(false);
            Ok(())
        }
        fn shift_click_png(&mut self, _: &Capture, _: u32, _: u32) -> Result<(), ScreenError> {
            Ok(())
        }
        fn right_click_png(&mut self, _: &Capture, _: u32, py: u32) -> Result<(), ScreenError> {
            let i = self.0.top.get() + (py / 100) as usize;
            self.0.asked.borrow_mut().push(self.0.names.get(i).cloned().unwrap_or_default());
            self.0.menu.set(true);
            Ok(())
        }
        fn type_text(&mut self, _: &str) -> Result<(), ScreenError> {
            Ok(())
        }
        fn key_return(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn key_escape(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn scroll(&mut self, lines: i32) -> Result<(), ScreenError> {
            let last = self.0.names.len().saturating_sub(SHOWN) as i64;
            self.0.top.set((self.0.top.get() as i64 + i64::from(lines)).clamp(0, last) as usize);
            Ok(())
        }
        fn sleep(&mut self, _: u64) {}
        fn capture(&mut self) -> Result<Capture, ScreenError> {
            Ok(Capture {
                png: Vec::new(),
                rect: filmcraft_screenagent::capture::WindowRect { x: 0, y: 0, w: 1000, h: 700 },
                px_per_point: 1.0,
                png_w: 1000,
                png_h: 700,
            })
        }
    }
    impl Ground for List {
        fn inventory(&self, _: &[u8], _: &str) -> Result<Vec<Element>, ScreenError> {
            Ok(self.names.iter().skip(self.top.get()).take(SHOWN).enumerate().map(|(i, n)| row(&format!("{n}.insv"), i as u32 * 100 + 30)).collect())
        }
        fn find(&self, _: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            Ok(Some(row(if target.contains("'Download' item") { "Download" } else { "OK" }, 10)))
        }
    }
    impl ReadText for List {
        fn read_text(&self, _: &[u8], _: &str) -> Result<String, ScreenError> {
            Ok(self.asked.borrow().last().map(|n| format!("{n}.insv")).unwrap_or_default())
        }
    }

    fn ask(names: Vec<String>, wanted: &str) -> (Result<(), String>, Vec<String>) {
        let list = List { names, top: std::cell::Cell::new(0), asked: std::cell::RefCell::new(Vec::new()), menu: std::cell::Cell::new(false) };
        let mut hands = Hands(&list);
        let mut d = Driver {
            act: &mut hands,
            ground: &list,
            read: &list,
            download_dir: PathBuf::new(),
            file_timeout: Duration::ZERO,
            progress: &mut |_| {},
            cancelled: &|| false,
        };
        let r = d.request(wanted);
        (r, list.asked.borrow().clone())
    }

    /// Studio lists the newest recording first: an older one is further down. The search scrolled
    /// up instead (it took the list for oldest first) and gave up after sixty screens at the top.
    #[test]
    fn an_older_recording_is_found_down_a_newest_first_list() {
        let names: Vec<String> = (0..90u32).rev().map(|i| format!("VID_2026{:02}{:02}_120000_00_{i:03}", 1 + i / 28, 1 + i % 28)).collect();
        let wanted = names[61].clone();
        let (r, asked) = ask(names.clone(), &wanted);
        assert_eq!(r, Ok(()));
        assert_eq!(asked, vec![wanted]);
        // oldest first (the user's own sort) still works, and a name that is not there is said so
        let mut oldest = names.clone();
        oldest.reverse();
        let wanted = oldest[61].clone();
        assert_eq!(ask(oldest, &wanted), (Ok(()), vec![wanted]));
        let (r, asked) = ask(names, "VID_20260214_120001_00_999");
        assert!(r.is_err_and(|e| e.contains("not in Studio's Cloud Media list")), "a missing name must be said");
        assert!(asked.is_empty());
    }

    #[test]
    fn only_the_download_item_is_clicked() {
        assert!(is_download_item(&el("Download")));
        assert!(!is_download_item(&el("Remove from Insta360 Cloud")));
        assert!(!is_download_item(&el("Remove download")));
        assert!(!is_download_item(&el("Share")));
    }
}
