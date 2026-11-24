//! Hands: activate the app, click, type, press keys, scroll — always inside the target
//! window, always through `osascript` subprocesses (the forms verified against a real
//! Google Photos picker), always `Result`-returning.
//!
//! Keys go through System Events (`keystroke`, `key code`). Pointer input does not: System
//! Events' `click at` acts on the accessibility tree, and Chrome's web content ignores it —
//! verified live, the picker never selected a tile. Clicks and wheel scrolls are real
//! CoreGraphics events posted from JXA (`osascript -l JavaScript`, its ObjC bridge), so
//! they move the pointer like a person would. Only integers (and the app's name, quoted) ever
//! reach those scripts.
//!
//! Keystrokes and posted clicks both go to whatever is in front, not to the app named — a
//! `tell process` block does not aim `keystroke` (seen live: a picker search typed into another
//! app's message box when the user switched apps mid-pick). So every input script first checks,
//! in the same script, that the target app is still the front app, and does nothing but fail
//! when it is not ([`NOT_IN_FRONT`]); the agent stops rather than take focus back.

use std::time::Duration;

use crate::ScreenError;
use crate::capture::{self, Capture, Shell};

/// Everything the agent loop can ask its hands to do. The macOS implementation is
/// [`MacActor`]; tests drive fakes.
pub trait Act {
    fn activate(&mut self) -> Result<(), ScreenError>;
    /// Click a PNG-pixel coordinate of the last [`Self::capture`] (clamped into the window).
    fn click_png(&mut self, cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError>;
    /// The same click with Shift held: in the picker it selects the whole range from the last
    /// clicked tile ("Hold Shift and click to select many items at once").
    fn shift_click_png(&mut self, cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError>;
    /// A right-click at a PNG-pixel coordinate of the last [`Self::capture`] (a context menu),
    /// clamped into the window.
    fn right_click_png(&mut self, _cap: &Capture, _px: u32, _py: u32) -> Result<(), ScreenError> {
        Err(ScreenError::Unsupported("right-clicking".into()))
    }
    fn type_text(&mut self, text: &str) -> Result<(), ScreenError>;
    fn key_return(&mut self) -> Result<(), ScreenError>;
    fn key_escape(&mut self) -> Result<(), ScreenError>;
    /// Scroll the window's content by `lines` wheel lines (negative scrolls up).
    fn scroll(&mut self, lines: i32) -> Result<(), ScreenError>;
    fn sleep(&mut self, ms: u64);
    fn capture(&mut self) -> Result<Capture, ScreenError>;
    /// Empty the focused text field: up to `chars` characters deleted on each side of the
    /// caret (Backspace, then Forward Delete) — keys that do nothing outside a text field,
    /// unlike Cmd+A, which in a photo grid could select everything.
    fn clear_text(&mut self, _chars: usize) -> Result<(), ScreenError> {
        Err(ScreenError::Unsupported("clearing a text field".into()))
    }
    /// The address of the front window's active tab, when the app is a browser that tells
    /// (`None`: unknown — and an unknown tab is never closed).
    fn front_url(&mut self) -> Result<Option<String>, ScreenError> {
        Ok(None)
    }
    /// Close the front window's active tab (Cmd+W).
    fn close_tab(&mut self) -> Result<(), ScreenError> {
        Err(ScreenError::Unsupported("closing a tab".into()))
    }
    /// The front window's active tab named by its window id, tab id and address, when the app
    /// is a browser whose tabs can be named so (Chrome); `None` otherwise.
    fn front_tab(&mut self) -> Result<Option<crate::tabs::Tab>, ScreenError> {
        Ok(None)
    }
    /// The address the tab with `tab`'s window and tab id shows now; `None` when it is gone.
    fn tab_url(&mut self, _tab: &crate::tabs::Tab) -> Result<Option<String>, ScreenError> {
        Err(ScreenError::Unsupported("naming tabs".into()))
    }
    /// Close that very tab — window id, tab id and address all still matching, and only a
    /// Google Photos picker; `Ok(false)`: no such tab any more.
    fn close_that_tab(&mut self, _tab: &crate::tabs::Tab) -> Result<bool, ScreenError> {
        Err(ScreenError::Unsupported("closing a tab by its id".into()))
    }
    /// Run the page script `js` in that picker tab and answer its value as text
    /// ([`crate::tabs::run_js`]); [`ScreenError::Unsupported`] when the browser runs none.
    fn run_js(&mut self, _tab: &crate::tabs::Tab, _js: &str) -> Result<String, ScreenError> {
        Err(ScreenError::Unsupported("page scripts".into()))
    }
}

/// The most wheel lines one scroll may post.
const MAX_SCROLL_LINES: i32 = 60;
/// The most characters [`Act::clear_text`] deletes each way.
const MAX_CLEAR: usize = 300;

/// Browsers whose AppleScript dictionary names the active tab `active tab` (Chromium's).
const CHROMIUM_BROWSERS: [&str; 8] =
    ["Google Chrome", "Google Chrome Beta", "Google Chrome Canary", "Chromium", "Brave Browser", "Microsoft Edge", "Vivaldi", "Arc"];

/// The AppleScript that reads the front tab's address in browser `app` (escaped already), or
/// `None` for an app that has no such dictionary.
fn front_url_script(app: &str) -> Option<String> {
    if CHROMIUM_BROWSERS.contains(&app) {
        Some(format!("tell application \"{app}\" to get URL of active tab of front window"))
    } else if app == "Safari" || app == "Safari Technology Preview" {
        Some(format!("tell application \"{app}\" to get URL of current tab of front window"))
    } else {
        None
    }
}

/// What an input script fails with when the target app is not the front app.
pub const NOT_IN_FRONT: &str = "is not in front";

/// The AppleScript line (inside `tell application "System Events"`) that fails unless `app`
/// (escaped already) is the front app.
fn front_check(app: &str) -> String {
    format!("if (name of first application process whose frontmost is true) is not \"{app}\" then error \"{app} {NOT_IN_FRONT}\"")
}

/// macOS hands: `osascript` + `System Events` scoped to one application process.
pub struct MacActor {
    app: String,
    shell: Box<dyn Shell>,
}

impl MacActor {
    pub fn new(app: impl Into<String>, shell: Box<dyn Shell>) -> Self {
        Self { app: app.into(), shell }
    }

    /// Post CoreGraphics events from JXA. Event types: 5 mouse moved, 1 left down, 2 left
    /// up; tap 0 = HID; scroll unit 1 = lines (`CGEventCreateScrollWheelEvent2`, macOS 13+).
    /// Every script here first checks that the target app is in front (see the module notes).
    fn jxa(&self, body: &str) -> Result<String, ScreenError> {
        let app = serde_json::to_string(&self.app).map_err(|e| ScreenError::Other(e.to_string()))?;
        let script = format!(
            "var front = Application(\"System Events\").processes.whose({{frontmost: true}}).name();\nif (front.length < 1 || front[0] !== {app}) throw new Error({app} + \" {NOT_IN_FRONT}\");\nObjC.import(\"CoreGraphics\");\n{body}\n\"ok\""
        );
        self.shell.run("osascript", &["-l", "JavaScript", "-e", &script]).map_err(|e| self.input_error(&e))
    }

    fn script(&self, body: &str) -> Result<String, ScreenError> {
        let app = capture::escape_applescript(&self.app)?;
        let full = format!("tell application \"System Events\"\n{}\ntell process \"{app}\"\n{body}\nend tell\nend tell", front_check(&app));
        self.shell.run("osascript", &["-e", &full]).map_err(|e| self.input_error(&e))
    }

    /// An input script's failure: [`ScreenError::NotInFront`] when the app had left the front.
    fn input_error(&self, e: &ScreenError) -> ScreenError {
        let text = e.to_string();
        if text.contains(NOT_IN_FRONT) { ScreenError::NotInFront(self.app.clone()) } else { ScreenError::Input(self.app.clone(), text) }
    }

    /// A wheel scroll of `lines` (negative: up) with the pointer at the screen point `(x, y)`,
    /// which must lie inside the target app's front window.
    pub fn scroll_at(&mut self, x: i32, y: i32, lines: i32) -> Result<(), ScreenError> {
        let lines = lines.clamp(-MAX_SCROLL_LINES, MAX_SCROLL_LINES);
        let r = capture::window_bounds(self.shell.as_ref(), &self.app)?;
        let (x, y) = r.clamp(x, y).ok_or_else(|| ScreenError::Other(format!("refusing to scroll outside the window at ({x},{y})")))?;
        let wheel = lines.saturating_neg();
        self.jxa(&format!(
            "var p = {{x: {x}, y: {y}}};\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 5, p, 0));\ndelay(0.08);\n$.CGEventPost(0, $.CGEventCreateScrollWheelEvent2(null, 1, 1, {wheel}, 0, 0));"
        ))
        .map(|_| ())
    }

    /// Press the key with macOS virtual key code `code` (124 the Right arrow, 34 the I key) in
    /// the target app, nothing held.
    pub fn key_code(&mut self, code: u16) -> Result<(), ScreenError> {
        self.script(&format!("key code {code}")).map(|_| ())
    }
}

impl Act for MacActor {
    fn activate(&mut self) -> Result<(), ScreenError> {
        let app = capture::escape_applescript(&self.app)?;
        let script = format!("tell application \"{app}\" to activate");
        self.shell.run("osascript", &["-e", &script]).map(|_| ()).map_err(|e| ScreenError::Input(self.app.clone(), e.to_string()))
    }

    fn click_png(&mut self, cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
        let Some((x, y)) = cap.click_point(px, py) else {
            return Err(ScreenError::Other(format!("refusing to click outside the window: png ({px},{py})")));
        };
        self.jxa(&format!(
            "var p = {{x: {x}, y: {y}}};\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 5, p, 0));\ndelay(0.08);\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 1, p, 0));\ndelay(0.05);\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 2, p, 0));"
        ))
        .map(|_| ())
    }

    fn right_click_png(&mut self, cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
        let Some((x, y)) = cap.click_point(px, py) else {
            return Err(ScreenError::Other(format!("refusing to click outside the window: png ({px},{py})")));
        };
        // right mouse down and up (event types 3 and 4) with the right button (1)
        self.jxa(&format!(
            "var p = {{x: {x}, y: {y}}};\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 5, p, 0));\ndelay(0.08);\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 3, p, 1));\ndelay(0.05);\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 4, p, 1));"
        ))
        .map(|_| ())
    }

    fn shift_click_png(&mut self, cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
        let Some((x, y)) = cap.click_point(px, py) else {
            return Err(ScreenError::Other(format!("refusing to click outside the window: png ({px},{py})")));
        };
        // Chrome reads the live modifier state, not a click event's flags (verified live: a
        // Shift-flagged CGEvent click selected one tile, not the range). So hold Shift
        // system-wide around a real click — and always let go of it.
        let app = capture::escape_applescript(&self.app)?;
        let shift = |down: bool| {
            // Shift is held only with the app in front; it is always let go of
            let s = if down {
                format!("tell application \"System Events\"\n{}\nkey down shift\nend tell", front_check(&app))
            } else {
                "tell application \"System Events\" to key up shift".to_string()
            };
            self.shell.run("osascript", &["-e", &s]).map(|_| ()).map_err(|e| self.input_error(&e))
        };
        shift(true)?;
        let clicked = self
            .jxa(&format!(
                "var p = {{x: {x}, y: {y}}};\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 5, p, 0));\ndelay(0.1);\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 1, p, 0));\ndelay(0.05);\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 2, p, 0));\ndelay(0.1);"
            ))
            .map(|_| ());
        let released = shift(false);
        clicked.and(released)
    }

    fn type_text(&mut self, text: &str) -> Result<(), ScreenError> {
        let text = capture::escape_applescript(text)?;
        self.script(&format!("keystroke \"{text}\"")).map(|_| ())
    }

    fn key_return(&mut self) -> Result<(), ScreenError> {
        self.script("key code 36").map(|_| ())
    }

    fn key_escape(&mut self) -> Result<(), ScreenError> {
        self.script("key code 53").map(|_| ())
    }

    /// A wheel scroll with the pointer over the window's centre. (Page Down was used before:
    /// it does nothing while the search box has focus, which it does right after a search.)
    fn scroll(&mut self, lines: i32) -> Result<(), ScreenError> {
        let lines = lines.clamp(-MAX_SCROLL_LINES, MAX_SCROLL_LINES);
        let r = capture::window_bounds(self.shell.as_ref(), &self.app)?;
        let half = |v: u32| i32::try_from(v / 2).unwrap_or(i32::MAX);
        let (cx, cy) = (r.x.saturating_add(half(r.w)), r.y.saturating_add(half(r.h)));
        // CoreGraphics wheels count up as positive
        let wheel = lines.saturating_neg();
        self.jxa(&format!(
            "var p = {{x: {cx}, y: {cy}}};\n$.CGEventPost(0, $.CGEventCreateMouseEvent(null, 5, p, 0));\ndelay(0.08);\n$.CGEventPost(0, $.CGEventCreateScrollWheelEvent2(null, 1, 1, {wheel}, 0, 0));"
        ))
        .map(|_| ())
    }

    fn sleep(&mut self, ms: u64) {
        std::thread::sleep(Duration::from_millis(ms.min(10_000)));
    }

    fn capture(&mut self) -> Result<Capture, ScreenError> {
        capture::capture_window(self.shell.as_ref(), &self.app)
    }

    fn clear_text(&mut self, chars: usize) -> Result<(), ScreenError> {
        let n = chars.min(MAX_CLEAR);
        if n == 0 {
            return Ok(());
        }
        // 51 = Delete (backspace), 117 = Forward Delete
        self.script(&format!("repeat {n} times\nkey code 51\nend repeat\nrepeat {n} times\nkey code 117\nend repeat")).map(|_| ())
    }

    fn front_url(&mut self) -> Result<Option<String>, ScreenError> {
        let app = capture::escape_applescript(&self.app)?;
        let Some(script) = front_url_script(&app) else { return Ok(None) };
        let out = self.shell.run("osascript", &["-e", &script])?;
        let url = out.trim();
        Ok((!url.is_empty() && url != "missing value").then(|| url.to_string()))
    }

    fn close_tab(&mut self) -> Result<(), ScreenError> {
        self.script("keystroke \"w\" using command down").map(|_| ())
    }

    fn front_tab(&mut self) -> Result<Option<crate::tabs::Tab>, ScreenError> {
        if self.app != crate::tabs::CHROME {
            return Ok(None);
        }
        crate::tabs::front_tab(self.shell.as_ref())
    }

    fn tab_url(&mut self, tab: &crate::tabs::Tab) -> Result<Option<String>, ScreenError> {
        crate::tabs::tab_url(self.shell.as_ref(), tab)
    }

    fn close_that_tab(&mut self, tab: &crate::tabs::Tab) -> Result<bool, ScreenError> {
        crate::tabs::close(self.shell.as_ref(), tab)
    }

    fn run_js(&mut self, tab: &crate::tabs::Tab, js: &str) -> Result<String, ScreenError> {
        if self.app != crate::tabs::CHROME {
            return Err(ScreenError::Unsupported(format!("page scripts in {}", self.app)));
        }
        crate::tabs::run_js(self.shell.as_ref(), tab, js)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::WindowRect;
    use std::sync::{Arc, Mutex};

    /// Records every script; optionally fails. The record is shared so assertions can read
    /// it after the shell moves into the actor.
    struct FakeShell {
        fail: bool,
        scripts: Arc<Mutex<Vec<String>>>,
    }
    impl FakeShell {
        fn new(fail: bool) -> (Arc<Mutex<Vec<String>>>, Self) {
            let scripts = Arc::new(Mutex::new(vec![]));
            (Arc::clone(&scripts), Self { fail, scripts })
        }
    }
    impl Shell for FakeShell {
        fn run(&self, program: &str, args: &[&str]) -> Result<String, ScreenError> {
            assert_eq!(program, "osascript");
            // AppleScript (`-e s`) or JXA (`-l JavaScript -e s`): the script is last
            assert!(matches!(args, ["-e", _] | ["-l", "JavaScript", "-e", _]), "{args:?}");
            let script = args.last().copied().unwrap_or_default();
            self.scripts.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(script.to_string());
            if self.fail {
                return Err(ScreenError::Other("not allowed assistive access".into()));
            }
            Ok(if script.contains("get {position, size} of front window") { "0, 33, 1728, 994\n".into() } else { "missing value\n".into() })
        }
    }

    fn recorded(shared: &Arc<Mutex<Vec<String>>>) -> String {
        shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner).join("\n")
    }

    fn cap_2x() -> Capture {
        Capture { png: vec![], rect: WindowRect { x: 0, y: 33, w: 1728, h: 994 }, px_per_point: 2.0, png_w: 3456, png_h: 1988 }
    }

    #[test]
    fn click_is_a_real_pointer_event_at_the_mapped_point() {
        let (shared, sh) = FakeShell::new(false);
        let mut actor = MacActor::new("Google Chrome", Box::new(sh));
        actor.click_png(&cap_2x(), 300, 500).unwrap();
        let s = recorded(&shared);
        assert!(s.contains("var p = {x: 150, y: 283};"), "{s}");
        // move, down, up — in that order
        let (mv, down, up) = (s.find("(null, 5, p, 0)"), s.find("(null, 1, p, 0)"), s.find("(null, 2, p, 0)"));
        assert!(mv < down && down < up && mv.is_some(), "{s}");
        assert!(!s.contains("click at"), "System Events clicks never reach web content: {s}");
    }

    #[test]
    fn shift_click_holds_shift_system_wide_around_a_real_click() {
        let (shared, sh) = FakeShell::new(false);
        let mut actor = MacActor::new("Google Chrome", Box::new(sh));
        actor.shift_click_png(&cap_2x(), 300, 500).unwrap();
        let s = recorded(&shared);
        let (down, click, up) = (s.find("key down shift"), s.find("CGEventCreateMouseEvent(null, 1, p, 0)"), s.find("key up shift"));
        assert!(down.is_some() && down < click && click < up, "{s}");
        assert!(s.contains("var p = {x: 150, y: 283};"), "{s}");
        assert!(actor.shift_click_png(&cap_2x(), 9999, 1).is_err(), "out of the window is refused");
    }

    #[test]
    fn shift_is_released_even_when_the_click_fails() {
        /// Fails only the JXA click.
        struct ClickFails(Arc<Mutex<Vec<String>>>);
        impl Shell for ClickFails {
            fn run(&self, _program: &str, args: &[&str]) -> Result<String, ScreenError> {
                let script = args.last().copied().unwrap_or_default().to_string();
                let jxa = args.first() == Some(&"-l");
                self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(script);
                if jxa { Err(ScreenError::Other("click refused".into())) } else { Ok(String::new()) }
            }
        }
        let shared = Arc::new(Mutex::new(vec![]));
        let mut actor = MacActor::new("Google Chrome", Box::new(ClickFails(Arc::clone(&shared))));
        assert!(actor.shift_click_png(&cap_2x(), 300, 500).is_err());
        let s = recorded(&shared);
        assert!(s.ends_with("tell application \"System Events\" to key up shift"), "Shift must not stay held: {s}");
    }

    #[test]
    fn click_outside_the_window_is_refused() {
        let (shared, sh) = FakeShell::new(false);
        let mut actor = MacActor::new("Google Chrome", Box::new(sh));
        assert!(actor.click_png(&cap_2x(), 9999, 10).is_err());
        assert!(actor.click_png(&cap_2x(), 10, 9999).is_err());
        assert!(recorded(&shared).is_empty(), "nothing may reach the OS");
    }

    #[test]
    fn every_input_script_first_checks_the_app_is_in_front() {
        let (shared, sh) = FakeShell::new(false);
        let mut actor = MacActor::new("Google Chrome", Box::new(sh));
        let cap = cap_2x();
        actor.click_png(&cap, 300, 500).unwrap();
        actor.right_click_png(&cap, 300, 500).unwrap();
        actor.shift_click_png(&cap, 300, 500).unwrap();
        actor.type_text("cake").unwrap();
        actor.key_return().unwrap();
        actor.key_escape().unwrap();
        actor.key_code(124).unwrap();
        actor.clear_text(3).unwrap();
        actor.scroll(2).unwrap();
        actor.scroll_at(10, 40, 2).unwrap();
        actor.close_tab().unwrap();
        let scripts = shared.lock().unwrap_or_else(std::sync::PoisonError::into_inner).clone();
        let input: Vec<&String> = scripts.iter().filter(|s| !s.contains("of front window") && !s.ends_with("key up shift")).collect();
        assert_eq!(input.len(), 12, "{scripts:#?}");
        for s in input {
            let guard = s.find(NOT_IN_FRONT).unwrap_or(usize::MAX);
            let acts = ["keystroke", "key code", "key down", "CGEventPost"].iter().filter_map(|a| s.find(a)).min().unwrap_or(0);
            assert!(guard < acts, "the check comes before any input: {s}");
            assert!(s.contains("frontmost"), "{s}");
        }
        let jxa = scripts.iter().find(|s| s.contains("CGEventPost")).unwrap();
        assert!(jxa.contains("front[0] !== \"Google Chrome\""), "the name is quoted as a JS string: {jxa}");
    }

    #[test]
    fn an_app_out_of_front_stops_the_run_instead_of_typing_elsewhere() {
        /// System Events answering as osascript does when the check fails.
        struct Elsewhere;
        impl Shell for Elsewhere {
            fn run(&self, _program: &str, _args: &[&str]) -> Result<String, ScreenError> {
                Err(ScreenError::Other("execution error: Google Chrome is not in front (-2700)".into()))
            }
        }
        let mut actor = MacActor::new("Google Chrome", Box::new(Elsewhere));
        for r in [actor.type_text("April 30 2023"), actor.key_return(), actor.click_png(&cap_2x(), 300, 500)] {
            let e = r.unwrap_err();
            assert!(matches!(e, ScreenError::NotInFront(ref app) if app == "Google Chrome"), "{e:?}");
            assert!(e.to_string().contains("no longer the front app"), "{e}");
        }
        // other failures stay what they were
        let (_, sh) = FakeShell::new(true);
        let mut actor = MacActor::new("Google Chrome", Box::new(sh));
        assert!(matches!(actor.type_text("x"), Err(ScreenError::Input(..))));
    }

    #[test]
    fn typing_escapes_and_keys_use_codes() {
        let (shared, sh) = FakeShell::new(false);
        let mut actor = MacActor::new("Safari", Box::new(sh));
        actor.type_text(r#"beach "sunset" \o/"#).unwrap();
        actor.key_return().unwrap();
        actor.key_escape().unwrap();
        let s = recorded(&shared);
        assert!(s.contains("keystroke \"beach \\\"sunset\\\" \\\\o/\""), "{s}");
        assert!(s.contains("key code 36"), "{s}");
        assert!(s.contains("key code 53"), "{s}");
    }

    #[test]
    fn scroll_wheels_over_the_window_centre_and_is_bounded() {
        let (shared, sh) = FakeShell::new(false);
        let mut actor = MacActor::new("Google Chrome", Box::new(sh));
        actor.scroll(18).unwrap();
        actor.scroll(-2).unwrap();
        actor.scroll(i32::MIN).unwrap();
        let s = recorded(&shared);
        assert!(s.contains("var p = {x: 864, y: 530};"), "{s}"); // centre of 0,33 1728x994
        assert!(s.contains("CGEventCreateScrollWheelEvent2(null, 1, 1, -18, 0, 0)"), "{s}"); // down
        assert!(s.contains("CGEventCreateScrollWheelEvent2(null, 1, 1, 2, 0, 0)"), "{s}"); // up
        assert!(s.contains(&format!("CGEventCreateScrollWheelEvent2(null, 1, 1, {MAX_SCROLL_LINES}, 0, 0)")), "{s}"); // clamped
    }

    #[test]
    fn tabs_close_with_cmd_w_and_fields_clear_without_select_all() {
        let (shared, sh) = FakeShell::new(false);
        let mut actor = MacActor::new("Google Chrome", Box::new(sh));
        actor.close_tab().unwrap();
        actor.clear_text(12).unwrap();
        actor.clear_text(usize::MAX).unwrap();
        actor.clear_text(0).unwrap();
        let s = recorded(&shared);
        assert!(s.contains("keystroke \"w\" using command down"), "{s}");
        assert!(s.contains("repeat 12 times\nkey code 51\nend repeat\nrepeat 12 times\nkey code 117"), "{s}");
        assert!(s.contains(&format!("repeat {MAX_CLEAR} times")), "bounded: {s}");
        assert!(!s.contains("keystroke \"a\" using command down"), "never Cmd+A: {s}");
        assert_eq!(s.matches("key code 51").count(), 2, "nothing to clear sends nothing");
    }

    #[test]
    fn the_front_tab_address_is_read_only_from_browsers_that_tell() {
        /// Answers every script with an address.
        struct Addr(Arc<Mutex<Vec<String>>>, &'static str);
        impl Shell for Addr {
            fn run(&self, _program: &str, args: &[&str]) -> Result<String, ScreenError> {
                self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner).push(args.last().copied().unwrap_or_default().into());
                Ok(self.1.into())
            }
        }
        let shared = Arc::new(Mutex::new(vec![]));
        let picker = "https://photos.google.com/picker/abc\n";
        let mut chrome = MacActor::new("Google Chrome", Box::new(Addr(Arc::clone(&shared), picker)));
        assert_eq!(chrome.front_url().unwrap().as_deref(), Some("https://photos.google.com/picker/abc"));
        let mut safari = MacActor::new("Safari", Box::new(Addr(Arc::clone(&shared), picker)));
        assert!(safari.front_url().unwrap().is_some());
        let s = recorded(&shared);
        assert!(s.contains("tell application \"Google Chrome\" to get URL of active tab of front window"), "{s}");
        assert!(s.contains("tell application \"Safari\" to get URL of current tab of front window"), "{s}");
        // a browser without the dictionary: unknown, and nothing is asked
        let shared = Arc::new(Mutex::new(vec![]));
        let mut firefox = MacActor::new("firefox", Box::new(Addr(Arc::clone(&shared), picker)));
        assert_eq!(firefox.front_url().unwrap(), None);
        assert!(recorded(&shared).is_empty());
        // an empty answer, or `missing value` (no window), is unknown too
        let (_, sh) = FakeShell::new(false);
        assert_eq!(MacActor::new("Google Chrome", Box::new(sh)).front_url().unwrap(), None);
    }

    #[test]
    fn assistive_access_failure_names_the_grant() {
        let (_, sh) = FakeShell::new(true);
        let mut actor = MacActor::new("Google Chrome", Box::new(sh));
        let err = actor.key_return().unwrap_err();
        assert!(err.to_string().contains("Accessibility"), "{err}");
        let err2 = actor.type_text("hi").unwrap_err();
        assert!(matches!(err2, ScreenError::Input(..)));
    }

    #[test]
    fn app_names_with_quotes_cannot_inject_applescript() {
        let (shared, sh) = FakeShell::new(false);
        let mut actor = MacActor::new("Chrome\" with extra", Box::new(sh));
        let _ = actor.key_return();
        let s = recorded(&shared);
        assert!(s.contains("tell process \"Chrome\\\" with extra\""), "{s}");
    }
}
