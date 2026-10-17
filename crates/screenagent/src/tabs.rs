//! The browser tabs the Google Photos picker lives in: find them, close them (the earlier ones
//! before an agent opens a new picking session, so one tab is open at a time), and answer the
//! browser's "Leave site?" dialog — only ever for tabs whose address is Google's picker.
//!
//! Every pick used to open a new tab and nothing closed it: forty searches left forty tabs, and
//! closing one by hand raises "Leave site?" (the picker keeps its selection behind a
//! beforeunload handler). Chrome's AppleScript dictionary lists every tab with its window, its
//! id and its address, and closes a tab found by id *and* address, so a tab that went somewhere
//! else since it was listed is never touched. A tab that asks "Leave site?" is closed once more,
//! which Chrome takes as Leave; where eyes are needed, the dialog's buttons are listed by their
//! text and only the one that reads "Leave" is clicked. macOS only, like the rest of the crate:
//! elsewhere no tab is found and nothing is closed.

use crate::act::Act;
use crate::capture::{Shell, escape_applescript};
use crate::ground::Ground;
use crate::{ScreenError, picker};

/// The browser whose tabs are listed and tidied.
pub const CHROME: &str = "Google Chrome";

/// Most tabs read from one listing.
const MAX_TABS: usize = 2000;
/// Longest address accepted.
const MAX_URL: usize = 8192;
/// How long an AppleScript call to the browser may take before it is an error (a close that
/// waits on a dialog must not hang the caller).
const SCRIPT_TIMEOUT_SECS: u32 = 15;
/// How long a closed tab gets before it is looked at again.
const SETTLE_MS: u64 = 1500;
/// Presses of Leave per tab.
const LEAVE_TRIES: u32 = 2;

/// Whether `url` is Google's Photos picker: `https://photos.google.com/` (or `/u/N/`, another
/// signed-in account) followed by `picker` or `integration/picker`. Nothing else is a picker
/// tab — not Google Photos itself, not another Google page.
pub fn is_picker_url(url: &str) -> bool {
    if url.len() > MAX_URL || url.chars().any(|c| c.is_whitespace() || c.is_control()) {
        return false;
    }
    let Some(mut path) = url.strip_prefix("https://photos.google.com/") else { return false };
    if let Some(after) = path.strip_prefix("u/") {
        let digits = after.chars().take_while(char::is_ascii_digit).count();
        match after.get(digits..).and_then(|r| r.strip_prefix('/')) {
            Some(rest) if digits > 0 => path = rest,
            _ => return false,
        }
    }
    let is = |word: &str| path == word || path.strip_prefix(word).is_some_and(|r| r.starts_with(['/', '?', '#']));
    is("picker") || is("integration/picker")
}

/// One browser tab: its window's id, its own id, and its address.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tab {
    pub window: i64,
    pub id: i64,
    pub url: String,
}

fn timed(app: &str, body: &str) -> String {
    format!("with timeout of {SCRIPT_TIMEOUT_SECS} seconds\ntell application \"{app}\"\n{body}\nend tell\nend timeout")
}

/// Every tab of every window, one `window<TAB>tab<TAB>url` line each.
fn list_script(app: &str) -> String {
    timed(
        app,
        "set out to \"\"\nrepeat with w in windows\nrepeat with t in tabs of w\nset out to out & (id of w as text) & (character id 9) & (id of t as text) & (character id 9) & (URL of t) & (character id 10)\nend repeat\nend repeat\nreturn out",
    )
}

/// `body` run on the tab with `tab`'s window id, id and address (`w`, `k` its index, `t`
/// itself); answers `ok`, or `gone` when no such tab is there any more. Ids are compared as
/// text: Chrome 155 answers them as text (`tab id "180477276"`), which never equals a number,
/// and older versions as integers, which `as text` turns into the same digits.
fn with_tab_script(app: &str, tab: &Tab, body: &str) -> Result<String, ScreenError> {
    let url = escape_applescript(&tab.url)?;
    Ok(timed(
        app,
        &format!(
            "repeat with w in windows\nif (id of w as text) is \"{win}\" then\nrepeat with k from 1 to (count of tabs of w)\nset t to tab k of w\nif (id of t as text) is \"{id}\" and (URL of t) is \"{url}\" then\n{body}\nreturn \"ok\"\nend if\nend repeat\nend if\nend repeat\nreturn \"gone\"",
            win = tab.window,
            id = tab.id
        ),
    ))
}

/// The tabs in a [`list_script`] answer; malformed lines are skipped.
pub fn parse_tabs(out: &str) -> Vec<Tab> {
    out.lines()
        .take(MAX_TABS)
        .filter_map(|line| {
            let mut it = line.splitn(3, '\t');
            let window = it.next()?.trim().parse().ok()?;
            let id = it.next()?.trim().parse().ok()?;
            let url = it.next()?.trim();
            (url.len() <= MAX_URL).then(|| Tab { window, id, url: url.to_string() })
        })
        .collect()
}

/// Whether Chrome is running (asking never launches it).
pub fn chrome_running(shell: &dyn Shell) -> bool {
    shell.run("osascript", &["-e", &format!("application \"{CHROME}\" is running")]).is_ok_and(|o| o.trim() == "true")
}

/// Chrome's picker tabs, in window and tab order; none when Chrome is not running.
pub fn picker_tabs(shell: &dyn Shell) -> Result<Vec<Tab>, ScreenError> {
    if !chrome_running(shell) {
        return Ok(Vec::new());
    }
    let out = shell.run("osascript", &["-e", &list_script(CHROME)]).map_err(|e| ScreenError::Other(format!("cannot list {CHROME}'s tabs: {e}")))?;
    Ok(parse_tabs(&out).into_iter().filter(|t| is_picker_url(&t.url)).collect())
}

/// Chrome's front window's active tab, named by its ids; `None` when Chrome is not running or
/// has no window.
pub fn front_tab(shell: &dyn Shell) -> Result<Option<Tab>, ScreenError> {
    if !chrome_running(shell) {
        return Ok(None);
    }
    let script = timed(
        CHROME,
        "if (count of windows) is 0 then return \"\"\nset w to front window\nset t to active tab of w\nreturn (id of w as text) & (character id 9) & (id of t as text) & (character id 9) & (URL of t)",
    );
    let out = shell.run("osascript", &["-e", &script]).map_err(|e| ScreenError::Other(format!("cannot read {CHROME}'s front tab: {e}")))?;
    Ok(parse_tabs(&out).into_iter().next())
}

/// The address Chrome's tab with `tab`'s window and tab id shows now; `None` when it is gone.
pub fn tab_url(shell: &dyn Shell, tab: &Tab) -> Result<Option<String>, ScreenError> {
    let script = timed(
        CHROME,
        &format!(
            "repeat with w in windows\nif (id of w as text) is \"{win}\" then\nrepeat with t in tabs of w\nif (id of t as text) is \"{id}\" then\nreturn \"url\" & (character id 9) & (URL of t)\nend if\nend repeat\nend if\nend repeat\nreturn \"gone\"",
            win = tab.window,
            id = tab.id
        ),
    );
    let out = shell.run("osascript", &["-e", &script])?;
    Ok(out.trim_end_matches(['\n', '\r']).strip_prefix("url\t").filter(|u| u.len() <= MAX_URL).map(str::to_string))
}

/// Run `body` on `tab`, refusing any tab that is not a picker; `Ok(false)` when it is gone.
fn on_tab(shell: &dyn Shell, tab: &Tab, body: &str) -> Result<bool, ScreenError> {
    if !is_picker_url(&tab.url) {
        return Err(ScreenError::Other(format!("refusing to touch a tab that is not the Google Photos picker: {}", tab.url)));
    }
    let out = shell.run("osascript", &["-e", &with_tab_script(CHROME, tab, body)?])?;
    Ok(out.trim() == "ok")
}

/// Close a picker tab. `Ok(false)`: it was already gone (or went elsewhere).
pub fn close(shell: &dyn Shell, tab: &Tab) -> Result<bool, ScreenError> {
    on_tab(shell, tab, "close t")
}

/// Bring a picker tab to the front: its window first, it the active tab, Chrome frontmost.
pub fn focus(shell: &dyn Shell, tab: &Tab) -> Result<bool, ScreenError> {
    on_tab(shell, tab, "set active tab index of w to k\nset index of w to 1\nactivate")
}

/// Answer the "Leave site?" dialog on `actor`'s front window: list its buttons by their text
/// and click the one that reads "Leave". Anything not labelled "Leave" (and button-sized) is
/// refused, so nothing else is ever clicked. `Ok(false)`: no Leave button showed.
pub fn answer_leave<A: Act + ?Sized, G: Ground + ?Sized>(actor: &mut A, grounder: &G, progress: &mut dyn FnMut(&str)) -> Result<bool, ScreenError> {
    let cap = actor.capture()?;
    let buttons = grounder.inventory(&cap.png, picker::LEAVE_DIALOG_BUTTONS)?;
    let Some(found) = picker::leave_button(&buttons, cap.png_w, cap.png_h) else {
        if !buttons.is_empty() {
            let read: Vec<String> = buttons.iter().take(4).map(|b| b.label.trim().chars().take(40).collect()).collect();
            progress(&format!("screenagent: no Leave button among the dialog's buttons ({}); not clicking", read.join(", ")));
        }
        return Ok(false);
    };
    progress("screenagent: answering the browser's \"Leave site?\" with Leave");
    let (px, py) = found.center();
    actor.click_png(&cap, px, py)?;
    Ok(true)
}

/// Close the picker tab `actor` drove, when the front tab is still it: Cmd+W, then Leave when
/// the browser asks. A front tab that is not the picker (the picker closed itself, or the user
/// moved on) is left alone, and so is one whose address the browser does not tell. `Ok(true)`:
/// the tab is closed.
pub fn close_front_picker<A: Act + ?Sized, G: Ground + ?Sized>(actor: &mut A, grounder: &G, progress: &mut dyn FnMut(&str)) -> Result<bool, ScreenError> {
    let Some(url) = actor.front_url()? else {
        progress("screenagent: leaving the picker tab open: this browser does not tell its address (`filmcraft-cli photos tidy` closes picker tabs in Chrome)");
        return Ok(false);
    };
    if !is_picker_url(&url) {
        return Ok(false); // the picker closed itself, or the front tab is something else
    }
    progress("screenagent: closing the picker tab");
    actor.activate()?;
    // the front tab may have changed while the browser came forward
    if actor.front_url()?.as_deref() != Some(url.as_str()) {
        return Ok(false);
    }
    actor.close_tab()?;
    for _ in 0..=LEAVE_TRIES {
        actor.sleep(SETTLE_MS);
        if actor.front_url()?.as_deref() != Some(url.as_str()) {
            return Ok(true);
        }
        // still there: the browser asks "Leave site?"
        if !answer_leave(actor, grounder, progress)? {
            progress("screenagent: the picker tab did not close and no Leave button showed; leaving it open");
            return Ok(false);
        }
    }
    actor.sleep(SETTLE_MS);
    Ok(actor.front_url()?.as_deref() != Some(url.as_str()))
}

/// Close the picker tab an agent drove: `driven`, the tab it named by its ids when it began
/// (else the front tab now), closed by those ids — never "whatever tab is in front" when the
/// agent finishes, since by then another tab may be (live: another session was working in
/// Chrome alongside). Only while its address is still the picker: a tab that went elsewhere
/// is left be. A tab holding a selection asks "Leave site?" on the first close and goes on the
/// second (Chrome takes it as Leave); one still open after three, if it is in front, gets its
/// Leave clicked. A browser that does not name its tabs gets [`close_front_picker`]. `Ok(true)`:
/// the agent closed it.
pub fn close_driven_picker<A: Act + ?Sized, G: Ground + ?Sized>(
    actor: &mut A,
    grounder: &G,
    driven: Option<&Tab>,
    progress: &mut dyn FnMut(&str),
) -> Result<bool, ScreenError> {
    let tab = match driven {
        Some(t) => Some(t.clone()),
        None => actor.front_tab()?.filter(|t| is_picker_url(&t.url)),
    };
    let Some(mut tab) = tab else {
        return close_front_picker(actor, grounder, progress);
    };
    let mut announced = false;
    for _ in 0..3 {
        match actor.tab_url(&tab)? {
            None => return Ok(announced), // closed (by us, or the picker closed itself)
            Some(url) if !is_picker_url(&url) => {
                progress("screenagent: the tab it drove shows something else now; leaving it open");
                return Ok(false);
            }
            Some(url) => tab.url = url,
        }
        if !announced {
            progress("screenagent: closing the picker tab");
            announced = true;
        }
        actor.close_that_tab(&tab)?;
        actor.sleep(SETTLE_MS);
    }
    if actor.tab_url(&tab)?.is_none() {
        return Ok(true);
    }
    // still asking: Leave, only on this very tab in front
    if actor.front_tab()?.is_some_and(|f| f.window == tab.window && f.id == tab.id) && answer_leave(actor, grounder, progress)? {
        actor.sleep(SETTLE_MS);
        if actor.tab_url(&tab)?.is_none() {
            return Ok(true);
        }
    }
    progress("screenagent: the picker tab did not close; leaving it open (`filmcraft-cli photos tidy` closes picker tabs)");
    Ok(false)
}

/// What `photos tidy` did.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Tidied {
    /// Picker tabs found.
    pub found: usize,
    /// Picker tabs closed.
    pub closed: usize,
    /// The addresses of picker tabs still open.
    pub left: Vec<String>,
    pub notes: Vec<String>,
}

/// Close `tabs` (picker tabs just listed): each once, then once more those still open — a tab
/// holding a selection asks "Leave site?" instead of closing, and Chrome takes a second close
/// while it asks as Leave (live, Chrome 155). Answers those of `tabs` still open.
fn close_twice(shell: &dyn Shell, tabs: &[Tab], pause: &mut dyn FnMut(u64), notes: &mut Vec<String>) -> Result<Vec<Tab>, ScreenError> {
    let ours = |all: Vec<Tab>| -> Vec<Tab> { all.into_iter().filter(|x| tabs.iter().any(|t| t.window == x.window && t.id == x.id)).collect() };
    let mut open: Vec<Tab> = tabs.to_vec();
    for _ in 0..2 {
        if open.is_empty() {
            break;
        }
        for t in &open {
            if let Err(e) = close(shell, t) {
                notes.push(format!("{}: {e}", t.url));
            }
        }
        pause(SETTLE_MS);
        open = ours(picker_tabs(shell)?);
    }
    Ok(open)
}

/// Close every picker tab before an agent opens a new picking session, so the new session's tab
/// is the only one; nothing is clicked and no eyes are needed. (Pointing an open picker tab at
/// the new session instead was tried: a tab holding a selection asks "Leave site?" while Chrome
/// already reports the new address — live, Chrome 155 — so the agent met the dialog, not the
/// picker.) Only for a session an agent drives: a tab someone is picking in by hand is theirs.
pub fn close_picker_tabs(shell: &dyn Shell, pause: &mut dyn FnMut(u64), progress: &mut dyn FnMut(&str)) -> Result<Tidied, ScreenError> {
    let mut out = Tidied::default();
    if !chrome_running(shell) {
        return Ok(out);
    }
    let tabs = picker_tabs(shell)?;
    out.found = tabs.len();
    if tabs.is_empty() {
        return Ok(out);
    }
    progress(&format!("screenagent: closing {} earlier Google Photos picker tab(s)", tabs.len()));
    out.left = close_twice(shell, &tabs, pause, &mut out.notes)?.into_iter().map(|t| t.url).collect();
    out.closed = out.found.saturating_sub(out.left.len());
    Ok(out)
}

/// Close every Google Photos picker tab in Chrome — those tabs and nothing else. A tab that asks
/// "Leave site?" is closed once more (Chrome takes that as Leave); one still open after that is
/// brought to the front and answered with Leave through `leave` (`actor` must be hands on
/// Chrome); without eyes it is left open and listed.
pub fn tidy<A: Act + ?Sized>(
    shell: &dyn Shell,
    actor: &mut A,
    leave: Option<&dyn Ground>,
    pause: &mut dyn FnMut(u64),
    progress: &mut dyn FnMut(&str),
) -> Result<Tidied, ScreenError> {
    let mut out = Tidied::default();
    if !chrome_running(shell) {
        out.notes.push(format!("{CHROME} is not running"));
        return Ok(out);
    }
    let tabs = picker_tabs(shell)?;
    out.found = tabs.len();
    if tabs.is_empty() {
        return Ok(out);
    }
    progress(&format!("screenagent: closing {} Google Photos picker tab(s)", tabs.len()));
    let ours = |all: Vec<Tab>| -> Vec<Tab> { all.into_iter().filter(|x| tabs.iter().any(|t| t.window == x.window && t.id == x.id)).collect() };
    let waiting = close_twice(shell, &tabs, pause, &mut out.notes)?;
    if !waiting.is_empty() {
        match leave {
            None => out.notes.push(format!(
                "{} picker tab(s) ask \"Leave site?\": click Leave in each (or start Ollama with the grounding model and run this again)",
                waiting.len()
            )),
            Some(grounder) => {
                for t in &waiting {
                    for _ in 0..LEAVE_TRIES {
                        if !focus(shell, t)? {
                            break; // closed meanwhile
                        }
                        pause(SETTLE_MS / 2);
                        match answer_leave(actor, grounder, progress) {
                            Ok(true) => pause(SETTLE_MS),
                            Ok(false) => {
                                // the close may not have reached it: once more, then look
                                let _ = close(shell, t);
                                pause(SETTLE_MS);
                            }
                            Err(e) => {
                                out.notes.push(format!("{}: {e}", t.url));
                                break;
                            }
                        }
                        if !ours(picker_tabs(shell)?).iter().any(|x| x.window == t.window && x.id == t.id) {
                            break;
                        }
                    }
                }
            }
        }
    }
    out.left = ours(picker_tabs(shell)?).into_iter().map(|t| t.url).collect();
    out.closed = out.found.saturating_sub(out.left.len());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capture::{Capture, WindowRect};
    use crate::ground::Element;
    use std::cell::RefCell;
    use std::sync::{Arc, Mutex, PoisonError};

    const P1: &str = "https://photos.google.com/picker/s1/autoclose";
    const P2: &str = "https://photos.google.com/integration/picker?sid=2";

    #[test]
    fn only_the_picker_is_a_picker_tab() {
        for yes in [
            P1,
            P2,
            "https://photos.google.com/picker",
            "https://photos.google.com/u/1/picker/abc",
            "https://photos.google.com/integration/picker/x#y",
            "https://photos.google.com/picker?x=1",
        ] {
            assert!(is_picker_url(yes), "{yes}");
        }
        for no in [
            "",
            "https://photos.google.com/",
            "https://photos.google.com/album/picker",
            "https://photos.google.com/pickers",
            "https://photos.google.com/u/picker/x",
            "https://photos.google.com/u/x/picker",
            "http://photos.google.com/picker/x",
            "https://photos.google.com.evil.example/picker/x",
            "https://evil.example/https://photos.google.com/picker",
            "https://mail.google.com/mail/u/0/",
            "https://photos.google.com/picker/a b",
            "https://photos.google.com/picker/a\nb",
        ] {
            assert!(!is_picker_url(no), "{no:?}");
        }
        assert!(!is_picker_url(&format!("https://photos.google.com/picker/{}", "x".repeat(MAX_URL))));
    }

    #[test]
    fn tab_listings_parse_and_skip_junk() {
        let tabs = parse_tabs(&format!("1\t10\t{P1}\n1\t11\thttps://example.com/\nnot a line\n2\tx\t{P2}\n3\t30\t{P2}\n\n"));
        assert_eq!(
            tabs,
            vec![
                Tab { window: 1, id: 10, url: P1.into() },
                Tab { window: 1, id: 11, url: "https://example.com/".into() },
                Tab { window: 3, id: 30, url: P2.into() },
            ]
        );
        assert!(parse_tabs("").is_empty());
        assert_eq!(parse_tabs(&"1\t2\thttps://a/\n".repeat(MAX_TABS + 10)).len(), MAX_TABS);
    }

    /// A fake Chrome: a tab list that `close` scripts change, and every script kept.
    /// `stubborn` tabs ask "Leave site?" instead: the close waits until their id is in `left`
    /// (Leave was clicked) — or, with `second_close_leaves` (Chrome 155, live), until the tab is
    /// closed again while it asks.
    struct FakeChrome {
        running: bool,
        tabs: Mutex<Vec<Tab>>,
        stubborn: Vec<i64>,
        second_close_leaves: bool,
        left: Arc<Mutex<Vec<i64>>>,
        /// The tabs whose close waits on "Leave site?".
        waiting: Mutex<Vec<i64>>,
        scripts: Mutex<Vec<String>>,
    }

    impl FakeChrome {
        fn new(tabs: Vec<Tab>) -> Self {
            Self {
                running: true,
                tabs: Mutex::new(tabs),
                stubborn: vec![],
                second_close_leaves: true,
                left: Arc::default(),
                waiting: Mutex::default(),
                scripts: Mutex::new(vec![]),
            }
        }
        /// What Leave let through.
        fn settle(&self, tabs: &mut Vec<Tab>) {
            let left = self.left.lock().unwrap_or_else(PoisonError::into_inner).clone();
            let mut waiting = self.waiting.lock().unwrap_or_else(PoisonError::into_inner);
            tabs.retain(|t| !(waiting.contains(&t.id) && left.contains(&t.id)));
            waiting.retain(|id| !left.contains(id));
        }
        fn scripts(&self) -> Vec<String> {
            self.scripts.lock().unwrap_or_else(PoisonError::into_inner).clone()
        }
        fn ids(&self) -> Vec<i64> {
            self.tabs.lock().unwrap_or_else(PoisonError::into_inner).iter().map(|t| t.id).collect()
        }
    }

    /// The window id, tab id and address a script names. Like Chrome 155, whose ids are text, a
    /// script that compares an id with a number names nothing.
    fn target(script: &str) -> Option<(i64, i64, String)> {
        let id = |key: &str| script.split(key).nth(1)?.split('"').next()?.parse::<i64>().ok();
        let url = script.split("(URL of t) is \"").nth(1)?.split('"').next()?.to_string();
        Some((id("if (id of w as text) is \"")?, id("if (id of t as text) is \"")?, url))
    }

    impl Shell for FakeChrome {
        fn run(&self, program: &str, args: &[&str]) -> Result<String, ScreenError> {
            assert_eq!(program, "osascript");
            let s = args.last().copied().unwrap_or_default().to_string();
            self.scripts.lock().unwrap_or_else(PoisonError::into_inner).push(s.clone());
            if s.contains("is running") {
                return Ok(format!("{}\n", self.running));
            }
            let mut tabs = self.tabs.lock().unwrap_or_else(PoisonError::into_inner);
            self.settle(&mut tabs);
            if s.contains("set out to") {
                return Ok(tabs.iter().map(|t| format!("{}\t{}\t{}\n", t.window, t.id, t.url)).collect());
            }
            if s.contains("set w to front window") {
                return Ok(tabs.first().map(|t| format!("{}\t{}\t{}\n", t.window, t.id, t.url)).unwrap_or_default());
            }
            if s.contains("return \"url\"") {
                let id = |key: &str| s.split(key).nth(1)?.split('"').next()?.parse::<i64>().ok();
                let (w, t) = (id("if (id of w as text) is \""), id("if (id of t as text) is \""));
                return Ok(match tabs.iter().find(|x| Some(x.window) == w && Some(x.id) == t) {
                    Some(x) => format!("url\t{}\n", x.url),
                    None => "gone\n".into(),
                });
            }
            let Some((w, id, url)) = target(&s) else { return Err(ScreenError::Other(format!("unexpected script: {s}"))) };
            let Some(pos) = tabs.iter().position(|t| t.window == w && t.id == id && t.url == url) else { return Ok("gone\n".into()) };
            if !s.contains("close t") {
                return Ok("ok\n".into()); // focus
            }
            let mut waiting = self.waiting.lock().unwrap_or_else(PoisonError::into_inner);
            if waiting.contains(&id) {
                // closed again while it asks "Leave site?": Chrome leaves
                if self.second_close_leaves {
                    waiting.retain(|w| *w != id);
                    tabs.remove(pos);
                }
            } else if self.stubborn.contains(&id) {
                waiting.push(id);
            } else {
                tabs.remove(pos);
            }
            Ok("ok\n".into())
        }
    }

    fn tab(window: i64, id: i64, url: &str) -> Tab {
        Tab { window, id, url: url.into() }
    }

    fn png() -> Vec<u8> {
        let img = image::GrayImage::from_pixel(200, 150, image::Luma([9]));
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    /// Hands on Chrome: clicks recorded; a click answers the front stubborn tab's dialog. The
    /// front tab's address is scripted per call.
    struct Hands {
        clicks: Vec<(u32, u32)>,
        urls: RefCell<std::collections::VecDeque<Option<String>>>,
        closed: u32,
        left: Arc<Mutex<Vec<i64>>>,
        answers: Vec<i64>,
    }
    impl Hands {
        fn new() -> Self {
            Self { clicks: vec![], urls: RefCell::default(), closed: 0, left: Arc::default(), answers: vec![] }
        }
    }
    impl Act for Hands {
        fn activate(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn click_png(&mut self, _cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            self.clicks.push((px, py));
            self.left.lock().unwrap_or_else(PoisonError::into_inner).extend(self.answers.iter().copied());
            Ok(())
        }
        fn shift_click_png(&mut self, _cap: &Capture, _px: u32, _py: u32) -> Result<(), ScreenError> {
            Err(ScreenError::Other("no shift-clicks here".into()))
        }
        fn type_text(&mut self, _text: &str) -> Result<(), ScreenError> {
            Err(ScreenError::Other("no typing here".into()))
        }
        fn key_return(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn key_escape(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn scroll(&mut self, _lines: i32) -> Result<(), ScreenError> {
            Ok(())
        }
        fn sleep(&mut self, _ms: u64) {}
        fn capture(&mut self) -> Result<Capture, ScreenError> {
            Ok(Capture { png: png(), rect: WindowRect { x: 0, y: 0, w: 200, h: 150 }, px_per_point: 1.0, png_w: 200, png_h: 150 })
        }
        fn front_url(&mut self) -> Result<Option<String>, ScreenError> {
            Ok(self.urls.borrow_mut().pop_front().flatten())
        }
        fn close_tab(&mut self) -> Result<(), ScreenError> {
            self.closed += 1;
            Ok(())
        }
    }

    /// Eyes that see these buttons for the Leave dialog's ask (or nothing).
    struct Sees(Vec<Element>);
    impl Ground for Sees {
        fn inventory(&self, _png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
            assert_eq!(task, picker::LEAVE_DIALOG_BUTTONS);
            Ok(self.0.clone())
        }
        fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            Err(ScreenError::Other(format!("not asked to find anything, asked for {target}")))
        }
    }

    fn cancel_button() -> Element {
        Element { label: "Cancel".into(), kind: "button".into(), x1: 70, y1: 90, x2: 110, y2: 104 }
    }

    /// The dialog's two buttons, as the model lists them live.
    fn leave_button() -> Sees {
        Sees(vec![cancel_button(), Element { label: "Leave".into(), kind: "button".into(), x1: 120, y1: 90, x2: 160, y2: 104 }])
    }

    #[test]
    fn tidy_closes_picker_tabs_and_nothing_else() {
        let chrome = FakeChrome::new(vec![tab(1, 10, P1), tab(1, 11, "https://mail.google.com/"), tab(2, 20, P2), tab(2, 21, "https://photos.google.com/")]);
        let mut hands = Hands::new();
        let t = tidy(&chrome, &mut hands, Some(&leave_button() as &dyn Ground), &mut |_| {}, &mut |_| {}).unwrap();
        assert_eq!((t.found, t.closed, t.left.len()), (2, 2, 0), "{t:?}");
        assert_eq!(chrome.ids(), [11, 21], "only the picker tabs are gone");
        assert!(hands.clicks.is_empty(), "no dialog, nothing clicked");
        // every close names its tab by window, id and address
        assert!(chrome.scripts().iter().filter(|s| s.contains("close t")).all(|s| s.contains("with timeout of")));
        // nothing to do: nothing asked of Chrome beyond the listing
        let chrome = FakeChrome::new(vec![tab(1, 11, "https://example.com/")]);
        let t = tidy(&chrome, &mut Hands::new(), None, &mut |_| {}, &mut |_| {}).unwrap();
        assert_eq!((t.found, t.closed), (0, 0));
        assert!(!chrome.scripts().iter().any(|s| s.contains("close t")));
        // Chrome not running: never launched by asking
        let mut chrome = FakeChrome::new(vec![tab(1, 10, P1)]);
        chrome.running = false;
        let t = tidy(&chrome, &mut Hands::new(), None, &mut |_| {}, &mut |_| {}).unwrap();
        assert!(t.notes.iter().any(|n| n.contains("not running")) && t.found == 0);
        assert_eq!(chrome.scripts().len(), 1);
    }

    #[test]
    fn tidy_closes_a_tab_that_asks_by_closing_it_again() {
        // live, Chrome 155: a picker tab holding a selection asked "Leave site?" on the first
        // close and was gone after the second — no eyes, nothing clicked
        let mut chrome = FakeChrome::new(vec![tab(1, 11, "https://mail.google.com/"), tab(1, 12, P2)]);
        chrome.stubborn = vec![12];
        let mut hands = Hands::new();
        let t = tidy(&chrome, &mut hands, None, &mut |_| {}, &mut |_| {}).unwrap();
        assert_eq!((t.found, t.closed, t.left.len()), (1, 1, 0), "{t:?}");
        assert!(t.notes.is_empty(), "{t:?}");
        assert!(hands.clicks.is_empty());
        assert_eq!(chrome.ids(), [11]);
        assert_eq!(chrome.scripts().iter().filter(|s| s.contains("close t")).count(), 2);
    }

    #[test]
    fn tidy_answers_leave_site_with_leave_only() {
        // a tab a second close does not shut: the eyes find Leave among the dialog's buttons
        let left = Arc::new(Mutex::new(vec![]));
        let mut chrome = FakeChrome::new(vec![tab(1, 10, P1), tab(1, 12, P2)]);
        chrome.stubborn = vec![12];
        chrome.second_close_leaves = false;
        chrome.left = Arc::clone(&left);
        let mut hands = Hands::new();
        hands.left = Arc::clone(&left);
        hands.answers = vec![12];
        let mut lines = vec![];
        let t = tidy(&chrome, &mut hands, Some(&leave_button() as &dyn Ground), &mut |_| {}, &mut |l| lines.push(l.to_string())).unwrap();
        assert_eq!((t.found, t.closed, t.left.len()), (2, 2, 0), "{t:?}");
        assert_eq!(hands.clicks, [(140, 97)], "the Leave button, once");
        assert!(chrome.scripts().iter().any(|s| s.contains("set index of w to 1")), "the asking tab was brought to the front");
        assert!(lines.iter().any(|l| l.contains("Leave site?")), "{lines:?}");
        // eyes that offer another button: nothing is clicked and the tab is reported
        let mut chrome = FakeChrome::new(vec![tab(1, 12, P2)]);
        chrome.stubborn = vec![12];
        chrome.second_close_leaves = false;
        let mut hands = Hands::new();
        let cancel = Sees(vec![cancel_button()]);
        let t = tidy(&chrome, &mut hands, Some(&cancel as &dyn Ground), &mut |_| {}, &mut |_| {}).unwrap();
        assert!(hands.clicks.is_empty(), "Cancel is never clicked");
        assert_eq!((t.closed, t.left.as_slice()), (0, [P2.to_string()].as_slice()));
        // no eyes at all: listed, with what to do
        let mut chrome = FakeChrome::new(vec![tab(1, 12, P2)]);
        chrome.stubborn = vec![12];
        chrome.second_close_leaves = false;
        let t = tidy(&chrome, &mut Hands::new(), None, &mut |_| {}, &mut |_| {}).unwrap();
        assert_eq!(t.left, [P2]);
        assert!(t.notes.iter().any(|n| n.contains("click Leave")), "{t:?}");
    }

    #[test]
    fn earlier_picker_tabs_are_closed_before_a_new_session() {
        // one picker tab holding a selection (it asks "Leave site?" on the first close), one
        // without, and two other tabs that are never touched
        let mut chrome = FakeChrome::new(vec![tab(1, 9, "https://example.com/"), tab(1, 10, P1), tab(1, 11, "https://mail.google.com/"), tab(2, 20, P2)]);
        chrome.stubborn = vec![10];
        let mut lines = vec![];
        let t = close_picker_tabs(&chrome, &mut |_| {}, &mut |l| lines.push(l.to_string())).unwrap();
        assert_eq!((t.found, t.closed, t.left.len()), (2, 2, 0), "{t:?}");
        assert_eq!(chrome.ids(), [9, 11]);
        assert!(lines.iter().any(|l| l.contains("closing 2 earlier")), "{lines:?}");
        assert!(!chrome.scripts().iter().any(|s| s.contains("set URL")), "no tab is pointed anywhere");
        // a tab a second close does not shut is reported, not clicked at (there are no hands)
        let mut chrome = FakeChrome::new(vec![tab(1, 10, P1)]);
        chrome.stubborn = vec![10];
        chrome.second_close_leaves = false;
        let t = close_picker_tabs(&chrome, &mut |_| {}, &mut |_| {}).unwrap();
        assert_eq!((t.closed, t.left.as_slice()), (0, [P1.to_string()].as_slice()));
        // none open: nothing closed; Chrome not running: only asked whether it runs
        let chrome = FakeChrome::new(vec![tab(1, 9, "https://example.com/")]);
        assert_eq!(close_picker_tabs(&chrome, &mut |_| {}, &mut |_| {}).unwrap(), Tidied::default());
        assert!(!chrome.scripts().iter().any(|s| s.contains("close t")));
        let mut chrome = FakeChrome::new(vec![tab(1, 10, P1)]);
        chrome.running = false;
        assert_eq!(close_picker_tabs(&chrome, &mut |_| {}, &mut |_| {}).unwrap(), Tidied::default());
        assert_eq!(chrome.scripts().len(), 1);
    }

    #[test]
    fn the_front_tab_and_a_named_tab_are_read_by_their_ids() {
        let chrome = FakeChrome::new(vec![tab(7, 70, P2), tab(7, 71, "https://example.com/a,b")]);
        assert_eq!(front_tab(&chrome).unwrap(), Some(tab(7, 70, P2)));
        assert_eq!(tab_url(&chrome, &tab(7, 71, "")).unwrap().as_deref(), Some("https://example.com/a,b"));
        assert_eq!(tab_url(&chrome, &tab(7, 72, P2)).unwrap(), None, "gone");
        assert_eq!(tab_url(&chrome, &tab(8, 70, P2)).unwrap(), None, "another window's id");
        // Chrome not running: no front tab, and it is not launched by asking
        let mut chrome = FakeChrome::new(vec![tab(7, 70, P2)]);
        chrome.running = false;
        assert_eq!(front_tab(&chrome).unwrap(), None);
        assert_eq!(chrome.scripts().len(), 1);
    }

    #[test]
    fn tab_ids_are_compared_as_text() {
        // live, Chrome 155: `(id of t) is 180477276` was false for that very tab (its id is
        // text), so `photos tidy` closed nothing
        let s = with_tab_script(CHROME, &tab(180_476_798, 180_477_276, P2), "close t").unwrap();
        assert!(s.contains("if (id of w as text) is \"180476798\" then"), "{s}");
        assert!(s.contains("if (id of t as text) is \"180477276\" and (URL of t) is \""), "{s}");
        assert!(!s.contains("(id of t) is 1") && !s.contains("(id of w) is 1"), "{s}");
    }

    #[test]
    fn only_picker_tabs_are_ever_closed_or_focused() {
        let chrome = FakeChrome::new(vec![tab(1, 11, "https://mail.google.com/")]);
        let mail = tab(1, 11, "https://mail.google.com/");
        assert!(close(&chrome, &mail).is_err());
        assert!(focus(&chrome, &mail).is_err());
        assert_eq!(chrome.scripts().len(), 0);
        // a picker tab that has gone since it was listed is reported gone, not guessed at
        let chrome = FakeChrome::new(vec![]);
        assert!(!close(&chrome, &tab(1, 10, P1)).unwrap());
    }

    #[test]
    fn the_driven_tab_is_closed_when_it_is_still_the_picker() {
        // in front: the picker (looked at, and again right before Cmd+W); after it: another tab
        let mut hands = Hands::new();
        hands.urls = RefCell::new([Some(P1.to_string()), Some(P1.to_string()), Some("https://example.com/".to_string())].into());
        assert!(close_front_picker(&mut hands, &Sees(vec![]), &mut |_| {}).unwrap());
        assert_eq!(hands.closed, 1);
        assert!(hands.clicks.is_empty());
        // "Leave site?" after Cmd+W: Leave is clicked, then the tab is gone
        let mut hands = Hands::new();
        hands.urls = RefCell::new([Some(P1.to_string()), Some(P1.to_string()), Some(P1.to_string()), None].into());
        assert!(close_front_picker(&mut hands, &leave_button(), &mut |_| {}).unwrap());
        assert_eq!((hands.closed, hands.clicks.len()), (1, 1));
        // the front tab is not the picker (it closed itself): nothing is closed
        let mut hands = Hands::new();
        hands.urls = RefCell::new([Some("https://mail.google.com/".to_string())].into());
        assert!(!close_front_picker(&mut hands, &leave_button(), &mut |_| {}).unwrap());
        assert_eq!((hands.closed, hands.clicks.len()), (0, 0));
        // a browser that does not tell its address: nothing is closed
        let mut hands = Hands::new();
        let mut lines = vec![];
        assert!(!close_front_picker(&mut hands, &leave_button(), &mut |l| lines.push(l.to_string())).unwrap());
        assert_eq!(hands.closed, 0);
        assert!(lines.iter().any(|l| l.contains("photos tidy")), "{lines:?}");
        // still there and the eyes offer Cancel: refused, left open
        let mut hands = Hands::new();
        hands.urls = RefCell::new([Some(P1.to_string()), Some(P1.to_string()), Some(P1.to_string())].into());
        let cancel = Sees(vec![cancel_button()]);
        assert!(!close_front_picker(&mut hands, &cancel, &mut |_| {}).unwrap());
        assert!(hands.clicks.is_empty());
        // the front tab changed while the browser came forward: no Cmd+W
        let mut hands = Hands::new();
        hands.urls = RefCell::new([Some(P1.to_string()), Some(PHOTO.to_string())].into());
        assert!(!close_front_picker(&mut hands, &leave_button(), &mut |_| {}).unwrap());
        assert_eq!(hands.closed, 0);
    }

    /// Another session's Google Photos tab (live: it was in front when an agent finished).
    const PHOTO: &str = "https://photos.google.com/photo/AF1QipNw0yVEeutKy";

    /// A browser that names its tabs: a tab list, the front tab, closes by id (a `stubborn` tab
    /// asks "Leave site?" on its first close and goes on the second), and every Cmd+W counted.
    struct Named {
        tabs: Vec<Tab>,
        front: (i64, i64),
        stubborn: Vec<i64>,
        asking: Vec<i64>,
        cmd_w: u32,
        closes: Vec<i64>,
    }
    impl Named {
        fn new(tabs: Vec<Tab>, front: (i64, i64)) -> Self {
            Self { tabs, front, stubborn: vec![], asking: vec![], cmd_w: 0, closes: vec![] }
        }
        fn ids(&self) -> Vec<i64> {
            self.tabs.iter().map(|t| t.id).collect()
        }
    }
    impl Act for Named {
        fn activate(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn click_png(&mut self, _cap: &Capture, _px: u32, _py: u32) -> Result<(), ScreenError> {
            Err(ScreenError::Other("nothing is clicked here".into()))
        }
        fn shift_click_png(&mut self, _cap: &Capture, _px: u32, _py: u32) -> Result<(), ScreenError> {
            Err(ScreenError::Other("no shift-clicks here".into()))
        }
        fn type_text(&mut self, _text: &str) -> Result<(), ScreenError> {
            Err(ScreenError::Other("no typing here".into()))
        }
        fn key_return(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn key_escape(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn scroll(&mut self, _lines: i32) -> Result<(), ScreenError> {
            Ok(())
        }
        fn sleep(&mut self, _ms: u64) {}
        fn capture(&mut self) -> Result<Capture, ScreenError> {
            Ok(Capture { png: png(), rect: WindowRect { x: 0, y: 0, w: 200, h: 150 }, px_per_point: 1.0, png_w: 200, png_h: 150 })
        }
        fn front_url(&mut self) -> Result<Option<String>, ScreenError> {
            Ok(self.tabs.iter().find(|t| (t.window, t.id) == self.front).map(|t| t.url.clone()))
        }
        fn close_tab(&mut self) -> Result<(), ScreenError> {
            self.cmd_w += 1;
            Ok(())
        }
        fn front_tab(&mut self) -> Result<Option<Tab>, ScreenError> {
            Ok(self.tabs.iter().find(|t| (t.window, t.id) == self.front).cloned())
        }
        fn tab_url(&mut self, tab: &Tab) -> Result<Option<String>, ScreenError> {
            Ok(self.tabs.iter().find(|t| t.window == tab.window && t.id == tab.id).map(|t| t.url.clone()))
        }
        fn close_that_tab(&mut self, tab: &Tab) -> Result<bool, ScreenError> {
            assert!(is_picker_url(&tab.url), "only a picker is ever closed: {}", tab.url);
            let Some(pos) = self.tabs.iter().position(|t| *t == *tab) else { return Ok(false) };
            self.closes.push(tab.id);
            if self.stubborn.contains(&tab.id) && !self.asking.contains(&tab.id) {
                self.asking.push(tab.id); // "Leave site?"
            } else {
                self.tabs.remove(pos);
            }
            Ok(true)
        }
    }

    #[test]
    fn the_driven_tab_is_closed_by_its_ids_whatever_tab_is_in_front() {
        // live: another session's photo tab was in front when the agent finished; Cmd+W would
        // have closed it. The tab the agent drove (named when it began) is closed instead.
        let driven = tab(1, 10, P1);
        let mut b = Named::new(vec![tab(1, 9, "https://example.com/"), driven.clone(), tab(1, 11, PHOTO)], (1, 11));
        let mut lines = vec![];
        assert!(close_driven_picker(&mut b, &Sees(vec![]), Some(&driven), &mut |l| lines.push(l.to_string())).unwrap());
        assert_eq!(b.ids(), [9, 11], "only the picker went");
        assert_eq!(b.cmd_w, 0, "never Cmd+W");
        assert!(lines.iter().any(|l| l.contains("closing the picker tab")), "{lines:?}");
        // one holding a selection asks "Leave site?" and goes on the second close
        let mut b = Named::new(vec![driven.clone(), tab(1, 11, PHOTO)], (1, 11));
        b.stubborn = vec![10];
        assert!(close_driven_picker(&mut b, &Sees(vec![]), Some(&driven), &mut |_| {}).unwrap());
        assert_eq!((b.ids(), b.closes.as_slice()), (vec![11], [10, 10].as_slice()));
        // its address moved on within the picker (auth → session): still that tab, still closed
        let moved = tab(1, 10, "https://photos.google.com/integration/picker/session/abc?x=1");
        let mut b = Named::new(vec![moved.clone()], (1, 10));
        assert!(close_driven_picker(&mut b, &Sees(vec![]), Some(&driven), &mut |_| {}).unwrap());
        assert!(b.ids().is_empty());
        // it went somewhere else (a photo opened in it): left open, nothing closed
        let mut b = Named::new(vec![tab(1, 10, PHOTO)], (1, 10));
        assert!(!close_driven_picker(&mut b, &leave_button(), Some(&driven), &mut |_| {}).unwrap());
        assert!(b.closes.is_empty() && b.ids() == [10]);
        // already gone (the picker closed itself after Done): nothing to do, nothing closed
        let mut b = Named::new(vec![tab(1, 11, PHOTO)], (1, 11));
        assert!(!close_driven_picker(&mut b, &Sees(vec![]), Some(&driven), &mut |_| {}).unwrap());
        assert!(b.closes.is_empty() && b.cmd_w == 0);
        // none recorded: the front tab now, by its ids — only when it is a picker
        let mut b = Named::new(vec![driven.clone(), tab(1, 11, PHOTO)], (1, 10));
        assert!(close_driven_picker(&mut b, &Sees(vec![]), None, &mut |_| {}).unwrap());
        assert_eq!(b.ids(), [11]);
        let mut b = Named::new(vec![driven.clone(), tab(1, 11, PHOTO)], (1, 11));
        assert!(!close_driven_picker(&mut b, &leave_button(), None, &mut |_| {}).unwrap());
        assert_eq!((b.ids(), b.cmd_w), (vec![10, 11], 0), "a photo tab in front: nothing closed");
    }
}
