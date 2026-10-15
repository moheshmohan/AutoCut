//! The target window: where it is, and what it looks like.
//!
//! All OS contact is subprocesses (`osascript`, `screencapture`), the same stance as
//! `open_in_browser` in filmcraft-gphotos — no `unsafe` (the workspace forbids it), no
//! framework linkage, and a clean `Result` when the privacy prompt hasn't been granted.
//! macOS only for now; other platforms get [`ScreenError::Unsupported`].

use std::sync::atomic::{AtomicU64, Ordering};

use crate::ScreenError;

/// One subprocess invocation: the program, its arguments, and what came back. Fakes in
/// tests implement this instead of touching the real OS.
pub trait Shell: Send + Sync {
    fn run(&self, program: &str, args: &[&str]) -> Result<String, ScreenError>;
}

/// The real shell: runs the program with the given arguments, captures output.
pub struct OsShell;

impl Shell for OsShell {
    fn run(&self, program: &str, args: &[&str]) -> Result<String, ScreenError> {
        // AUTOCUT_SCREENAGENT_TRACE=1 logs every OS invocation for debugging a live run
        if std::env::var("AUTOCUT_SCREENAGENT_TRACE").ok().as_deref() == Some("1") {
            use std::io::Write as _;
            if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open("/tmp/screenagent-cmds.log") {
                let _ = writeln!(f, "=== {program}\n{}", args.join("\u{1f}"));
            }
        }
        let out = std::process::Command::new(program).args(args).output().map_err(|e| ScreenError::Other(format!("cannot run {program}: {e}")))?;
        if out.status.success() {
            Ok(String::from_utf8_lossy(&out.stdout).into_owned())
        } else {
            let msg = String::from_utf8_lossy(&out.stderr);
            let msg = msg.trim();
            Err(ScreenError::Other(if msg.is_empty() { format!("{program} exited with {}", out.status) } else { format!("{program}: {msg}") }))
        }
    }
}

/// A window rectangle in screen points (top-left origin, as AppleScript reports it).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WindowRect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl WindowRect {
    /// The largest rectangle safely inside this one (clicks are clamped into it, §0.4:
    /// every model-derived number is hostile until proven inside).
    pub fn clamp(&self, x: i32, y: i32) -> Option<(i32, i32)> {
        if self.w < 2 || self.h < 2 {
            return None;
        }
        let (mut x, mut y) = (x, y);
        let min_x = self.x.saturating_add(1);
        let max_x = self.x.saturating_add(self.w as i32).saturating_sub(1);
        let min_y = self.y.saturating_add(1);
        let max_y = self.y.saturating_add(self.h as i32).saturating_sub(1);
        // a clamped point is only meaningful if it started inside a sane neighbourhood;
        // anything wild is refused outright rather than dragged to the edge
        // i64 so the "is this wild?" test cannot itself overflow on i32::MIN (§0.4)
        let near = |v: i32, lo: i32, hi: i32| (i64::from(v)).abs() < 1 << 20 && i64::from(v) >= i64::from(lo) - 64 && i64::from(v) <= i64::from(hi) + 64;
        if !near(x, min_x, max_x) || !near(y, min_y, max_y) {
            return None;
        }
        x = x.clamp(min_x, max_x);
        y = y.clamp(min_y, max_y);
        Some((x, y))
    }
}

/// One screenshot of the target window: the PNG (downscaled for the model), the window's
/// screen rectangle, and how many PNG pixels make one screen point.
#[derive(Debug, Clone)]
pub struct Capture {
    pub png: Vec<u8>,
    pub rect: WindowRect,
    /// PNG pixels per screen point (Retina × any downscale); always finite and > 0.
    pub px_per_point: f32,
    pub png_w: u32,
    pub png_h: u32,
}

impl Capture {
    /// A point (in PNG pixel coordinates) → the screen point to click, clamped into the
    /// window. `None` when the coordinate is outside anything clickable.
    pub fn click_point(&self, px: u32, py: u32) -> Option<(i32, i32)> {
        if self.px_per_point <= 0.0 || px >= self.png_w || py >= self.png_h {
            return None;
        }
        let sx = self.rect.x as f32 + px as f32 / self.px_per_point;
        let sy = self.rect.y as f32 + py as f32 / self.px_per_point;
        let (sx, sy) = (sx.round(), sy.round());
        if !sx.is_finite() || !sy.is_finite() {
            return None;
        }
        self.rect.clamp(sx as i32, sy as i32)
    }
}

static CAPTURE_SEQ: AtomicU64 = AtomicU64::new(0);

/// AppleScript quoting: escape backslashes and quotes; refuse control characters (they
/// cannot be typed as literal keystrokes anyway and only smuggle syntax).
pub(crate) fn escape_applescript(s: &str) -> Result<String, ScreenError> {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '\\' | '"' => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c == '\u{7f}' => {
                return Err(ScreenError::Other(format!("text contains a control character (U+{:04X}) that cannot be typed", c as u32)));
            }
            c => out.push(c),
        }
    }
    Ok(out)
}

/// Parse `x, y, w, h` (AppleScript's `{position, size}` rendering) into a [`WindowRect`].
pub(crate) fn parse_bounds(text: &str) -> Result<WindowRect, ScreenError> {
    let parts: Vec<&str> = text.split(',').map(str::trim).collect();
    let nums: Result<Vec<i64>, _> = parts.iter().map(|p| p.parse::<i64>()).collect();
    let nums = nums.map_err(|e| ScreenError::Other(format!("window bounds `{text}` are not numbers: {e}")))?;
    if nums.len() != 4 {
        return Err(ScreenError::Other(format!("window bounds `{text}` do not have four numbers")));
    }
    let (x, y) = (
        i32::try_from(nums[0]).map_err(|e| ScreenError::Other(format!("window x out of range: {e}")))?,
        i32::try_from(nums[1]).map_err(|e| ScreenError::Other(format!("window y out of range: {e}")))?,
    );
    let (w, h) = (
        u32::try_from(nums[2]).map_err(|e| ScreenError::Other(format!("window width out of range: {e}")))?,
        u32::try_from(nums[3]).map_err(|e| ScreenError::Other(format!("window height out of range: {e}")))?,
    );
    if w == 0 || h == 0 {
        return Err(ScreenError::Other(format!("window has no area ({w}x{h})")));
    }
    Ok(WindowRect { x, y, w, h })
}

fn bounds_script(app: &str) -> String {
    format!("tell application \"System Events\" to tell process \"{app}\" to get {{position, size}} of front window")
}

fn frontmost_script() -> &'static str {
    "tell application \"System Events\" to get name of first application process whose frontmost is true"
}

/// The frontmost application's process name — how the agent learns which browser opened
/// the picker.
pub fn frontmost_app(shell: &dyn Shell) -> Result<String, ScreenError> {
    let name = shell.run("osascript", &["-e", frontmost_script()])?;
    let name = name.trim();
    if name.is_empty() {
        return Err(ScreenError::Other("no frontmost application".into()));
    }
    Ok(name.to_string())
}

/// The target app's front-window rectangle (screen points).
pub fn window_bounds(shell: &dyn Shell, app: &str) -> Result<WindowRect, ScreenError> {
    let app = escape_applescript(app)?;
    let out = shell.run("osascript", &["-e", &bounds_script(&app)]).map_err(|e| ScreenError::Input(app.clone(), e.to_string()))?;
    parse_bounds(&out)
}

/// The largest PNG side sent to the model. Qwen2.5-VL handles ~1.3–1.8k px happily; smaller
/// is faster and plenty for a thumbnail grid.
const MAX_SIDE: u32 = 1344;

/// Screenshot `app`'s front window, downscaled to at most [`MAX_SIDE`] on its long edge.
pub fn capture_window(shell: &dyn Shell, app: &str) -> Result<Capture, ScreenError> {
    #[cfg(not(target_os = "macos"))]
    {
        let _ = (shell, app);
        return Err(ScreenError::Unsupported(std::env::consts::OS.into()));
    }
    #[cfg(target_os = "macos")]
    {
        let rect = window_bounds(shell, app)?;
        let seq = CAPTURE_SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("filmcraft-screenagent-{}-{seq}.png", std::process::id()));
        let region = format!("-R{},{},{},{}", rect.x, rect.y, rect.w, rect.h);
        let path_str = path.to_string_lossy().into_owned();
        let ran = shell.run("screencapture", &["-x", &region, &path_str]);
        let png = match ran {
            Ok(_) => std::fs::read(&path),
            Err(e) => {
                let _ = std::fs::remove_file(&path);
                return Err(ScreenError::Capture(app.to_string(), e.to_string()));
            }
        };
        let _ = std::fs::remove_file(&path);
        let png = png.map_err(|e| ScreenError::Capture(app.to_string(), format!("no screenshot came back: {e}")))?;
        if png.is_empty() {
            return Err(ScreenError::Capture(app.to_string(), "the screenshot was empty".into()));
        }
        decode_and_scale(png, rect)
    }
}

/// Decode the captured PNG, downscale for the model, and derive the pixel-per-point scale
/// from the real image size (Retina-proof: whatever `screencapture` produced).
pub(crate) fn decode_and_scale(png: Vec<u8>, rect: WindowRect) -> Result<Capture, ScreenError> {
    let img = image::load_from_memory(&png).map_err(|e| ScreenError::Capture("the target".into(), format!("the screenshot is not a decodable image: {e}")))?;
    let (w0, h0) = (img.width(), img.height());
    if w0 == 0 || h0 == 0 {
        return Err(ScreenError::Capture("the target".into(), "the screenshot has no area".into()));
    }
    let long = u64::from(w0.max(h0));
    let (img, w, h) = if long > u64::from(MAX_SIDE) {
        let f = f64::from(MAX_SIDE) / long as f64;
        let nw = ((f64::from(w0) * f).round() as u32).max(1);
        let nh = ((f64::from(h0) * f).round() as u32).max(1);
        let scaled = img.resize_exact(nw, nh, image::imageops::FilterType::Triangle);
        (scaled, nw, nh)
    } else {
        (img, w0, h0)
    };
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).map_err(|e| ScreenError::Capture("the target".into(), format!("cannot re-encode the screenshot: {e}")))?;
    let px_per_point = f64::from(w) / f64::from(rect.w.max(1));
    let px_per_point = if px_per_point.is_finite() && px_per_point > 0.0 { px_per_point as f32 } else { 1.0 };
    Ok(Capture { png: out.into_inner(), rect, px_per_point, png_w: w, png_h: h })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fails only `screencapture`; other calls answer with fixed stdout.
    struct FakeShell {
        stdout: String,
        fail_capture: bool,
    }
    impl FakeShell {
        fn capture_denied() -> Self {
            Self { stdout: "0, 33, 1728, 994".into(), fail_capture: true }
        }
    }
    impl Shell for FakeShell {
        fn run(&self, program: &str, _args: &[&str]) -> Result<String, ScreenError> {
            if program == "screencapture" && self.fail_capture {
                return Err(ScreenError::Other("screencapture: could not create image".into()));
            }
            Ok(self.stdout.clone())
        }
    }

    #[test]
    fn bounds_parse_good_and_hostile() {
        let r = parse_bounds("0, 33, 1728, 994").unwrap();
        assert_eq!((r.x, r.y, r.w, r.h), (0, 33, 1728, 994));
        for bad in ["", "1,2,3", "1,2,3,4,5", "a,b,c,d", "0, 0, 0, 994", "1,2,-3,4", "999999999999999999999, 2, 3, 4"] {
            assert!(parse_bounds(bad).is_err(), "{bad} should not parse");
        }
    }

    #[test]
    fn clamp_keeps_points_inside_and_refuses_wild_ones() {
        let r = WindowRect { x: 0, y: 33, w: 1728, h: 994 };
        assert_eq!(r.clamp(100, 100), Some((100, 100)));
        assert_eq!(r.clamp(-5, 40), Some((1, 40))); // nudged inside
        assert_eq!(r.clamp(5000, 100), None); // wild → refused, not dragged
        assert_eq!(r.clamp(100, i32::MIN), None);
        assert_eq!(WindowRect { x: 0, y: 0, w: 1, h: 1 }.clamp(0, 0), None); // no interior
    }

    #[test]
    fn click_point_maps_png_to_screen() {
        // 2x retina capture of a window at (0,33): png 3456x1988 for 1728x994 points
        let cap = Capture { png: vec![], rect: WindowRect { x: 0, y: 33, w: 1728, h: 994 }, px_per_point: 2.0, png_w: 3456, png_h: 1988 };
        // png (300, 500) → point (150, 33+250=283)
        assert_eq!(cap.click_point(300, 500), Some((150, 283)));
        assert_eq!(cap.click_point(0, 0), Some((1, 34))); // clamped off the very edge
        assert_eq!(cap.click_point(3456, 500), None); // outside the png
        assert_eq!(cap.click_point(10, 10), cap.click_point(10, 10)); // stable
    }

    #[test]
    fn escaping_is_safe_for_applescript() {
        assert_eq!(escape_applescript(r#"say "hi""#).unwrap(), r#"say \"hi\""#);
        assert_eq!(escape_applescript("back\\slash").unwrap(), r#"back\\slash"#);
        assert_eq!(escape_applescript("héllo → 世界").unwrap(), "héllo → 世界");
        for bad in ["a\nb", "tab\there", "\0", "\u{7f}"] {
            assert!(escape_applescript(bad).is_err(), "{bad:?} must be refused");
        }
    }

    #[test]
    fn capture_failure_names_the_screen_recording_grant() {
        // bounds work, but the capture itself is denied → the Capture variant with guidance
        let err = capture_window(&FakeShell::capture_denied(), "Google Chrome").unwrap_err();
        assert!(err.to_string().contains("Screen Recording"), "{err}");
        assert!(matches!(err, ScreenError::Capture(..)));
    }

    #[test]
    fn decode_and_scale_derives_retina_scale() {
        // draw a 2728x1941 png (weird size on purpose); window is 1728x994 points
        let img = image::GrayImage::from_pixel(2728, 1941, image::Luma([120u8]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        let rect = WindowRect { x: 0, y: 33, w: 1728, h: 994 };
        let cap = decode_and_scale(buf.into_inner(), rect).unwrap();
        assert!(cap.png_w <= MAX_SIDE && cap.png_h <= MAX_SIDE);
        // scale = scaled_width / 1728; long side was 2728 → scaled to 1344 → 1344/1728 = 0.777…
        assert!((cap.px_per_point - 1344.0 / 1728.0).abs() < 0.01, "scale {}", cap.px_per_point);
        assert_eq!(cap.png_w, 1344);
    }

    #[test]
    fn non_macos_is_unsupported_cleanly() {
        if cfg!(target_os = "macos") {
            return; // this host is macOS; the branch is compiled elsewhere
        }
        assert!(matches!(capture_window(&OsShell, "Any").unwrap_err(), ScreenError::Unsupported(_)));
    }
}
