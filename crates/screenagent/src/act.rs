//! Hands: activate the app, click, type, press keys, scroll — always inside the target
//! window, always through `osascript` subprocesses (the forms verified against a real
//! Google Photos picker), always `Result`-returning.
//!
//! Keys go through System Events (`keystroke`, `key code`). Pointer input does not: System
//! Events' `click at` acts on the accessibility tree, and Chrome's web content ignores it —
//! verified live, the picker never selected a tile. Clicks and wheel scrolls are real
//! CoreGraphics events posted from JXA (`osascript -l JavaScript`, its ObjC bridge), so
//! they move the pointer like a person would. Only integers ever reach those scripts.

use std::time::Duration;

use crate::ScreenError;
use crate::capture::{self, Capture, Shell};

/// Everything the agent loop can ask its hands to do. The macOS implementation is
/// [`MacActor`]; tests drive fakes.
pub trait Act {
    fn activate(&mut self) -> Result<(), ScreenError>;
    /// Click a PNG-pixel coordinate of the last [`Self::capture`] (clamped into the window).
    fn click_png(&mut self, cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError>;
    fn type_text(&mut self, text: &str) -> Result<(), ScreenError>;
    fn key_return(&mut self) -> Result<(), ScreenError>;
    fn key_escape(&mut self) -> Result<(), ScreenError>;
    /// Scroll the window's content by `lines` wheel lines (negative scrolls up).
    fn scroll(&mut self, lines: i32) -> Result<(), ScreenError>;
    fn sleep(&mut self, ms: u64);
    fn capture(&mut self) -> Result<Capture, ScreenError>;
}

/// The most wheel lines one scroll may post.
const MAX_SCROLL_LINES: i32 = 60;

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
    fn jxa(&self, body: &str) -> Result<String, ScreenError> {
        let script = format!("ObjC.import(\"CoreGraphics\");\n{body}\n\"ok\"");
        self.shell.run("osascript", &["-l", "JavaScript", "-e", &script]).map_err(|e| ScreenError::Input(self.app.clone(), e.to_string()))
    }

    fn script(&self, body: &str) -> Result<String, ScreenError> {
        let app = capture::escape_applescript(&self.app)?;
        let full = format!("tell application \"System Events\"\ntell process \"{app}\"\n{body}\nend tell\nend tell");
        self.shell.run("osascript", &["-e", &full]).map_err(|e| ScreenError::Input(self.app.clone(), e.to_string()))
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
    fn click_outside_the_window_is_refused() {
        let (shared, sh) = FakeShell::new(false);
        let mut actor = MacActor::new("Google Chrome", Box::new(sh));
        assert!(actor.click_png(&cap_2x(), 9999, 10).is_err());
        assert!(actor.click_png(&cap_2x(), 10, 9999).is_err());
        assert!(recorded(&shared).is_empty(), "nothing may reach the OS");
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
