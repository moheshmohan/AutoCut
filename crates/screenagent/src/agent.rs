//! The loop: see (capture + ground), decide, act, verify — with hard budgets on actions
//! and time so the agent can never spin forever (§0). It drives one application's front
//! window, which had better be the Google Photos picker the caller just opened.

use std::time::{Duration, Instant};

use crate::ScreenError;
use crate::act::Act;
use crate::ground::Ground;
use crate::picker;
use crate::verify;

/// What the agent should accomplish in the picker.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Free text typed into the picker's own search field (None = take the grid as-is).
    pub query: Option<String>,
    /// How many tiles to select.
    pub max_items: u32,
    /// Only click tiles that look like videos (duration badges).
    pub videos_only: bool,
    /// Hard cap on state-changing actions; the loop refuses to exceed it.
    pub max_actions: u32,
    /// Hard wall-clock budget.
    pub timeout: Duration,
}

impl Default for Plan {
    fn default() -> Self {
        Self { query: None, max_items: 6, videos_only: true, max_actions: 40, timeout: Duration::from_secs(300) }
    }
}

/// What the agent did.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Summary {
    pub actions: u32,
    pub selected: u32,
    pub searched: bool,
    pub done_pressed: bool,
    pub note: String,
}

/// Consecutive captures that all look the same before the agent gives up.
const MAX_STALLED: u32 = 3;
/// One tile click needs this long to land before re-capturing.
const CLICK_SETTLE_MS: u64 = 700;
/// Search results need this long to render.
const SEARCH_SETTLE_MS: u64 = 2500;
/// One "page" of the picker grid: ~40 pt per wheel line, ~3/4 of a laptop-sized window.
const SCROLL_LINES: i32 = 18;
/// Two click points closer than this (capture pixels, per axis) are the same tile.
const SAME_SPOT: u32 = 16;

/// Convenience for callers that just opened the picker in the default browser: spawns a
/// thread that waits for the picker to load, learns which browser is frontmost, checks the
/// grounding model is present, then drives the picker (search, select, Done). The join
/// handle reports what the agent did; the authoritative outcome stays the pick session.
///
/// Needs Screen Recording + Accessibility for the host process and the model in Ollama —
/// all failures come back as readable strings on the handle.
pub fn spawn(query: String, max_items: u32) -> std::thread::JoinHandle<Result<Summary, String>> {
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(2500)); // let the picker load
        let shell = crate::capture::OsShell;
        let app = crate::capture::frontmost_app(&shell).map_err(|e| e.to_string())?;
        let cfg = crate::Config::from_env();
        let grounder = crate::ground::QwenGrounder::from_config(&cfg);
        if !grounder.model_available().map_err(|e| e.to_string())? {
            return Err(format!("the grounding model `{}` is not in Ollama — `ollama pull {}` first", cfg.model, cfg.model));
        }
        let plan = Plan { query: Some(query), max_items, ..Plan::default() };
        let mut actor = crate::act::MacActor::new(app, Box::new(shell));
        drive(&mut actor, &grounder, &plan, &mut |l| eprintln!("{l}")).map_err(|e| e.to_string())
    })
}

/// The clickable tiles in one capture. A reply that groups the grid into one box or loops
/// on one label gets one retry with a stricter ask; a second failure counts as "none here"
/// (the loop then scrolls), never as tiles to click blind.
fn visible_tiles<G: Ground>(
    grounder: &G,
    cap: &crate::capture::Capture,
    videos_only: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Vec<crate::ground::Element>, ScreenError> {
    let task = if videos_only { picker::VIDEO_TILES } else { picker::ALL_TILES };
    let (w, h) = (cap.png_w, cap.png_h);
    let mut els = grounder.inventory(&cap.png, task)?;
    if picker::enumeration_failed(&els, w, h) {
        progress("screenagent: the model lumped the grid together; asking again for one box per thumbnail");
        els = grounder.inventory(&cap.png, &format!("{task} {}", picker::STRICT))?;
        if picker::enumeration_failed(&els, w, h) {
            return Ok(Vec::new());
        }
    }
    Ok(picker::tile_candidates(&els, videos_only, w, h))
}

/// Run the picker plan against one app. `progress` receives one line per step, the gphotos
/// convention (`&mut dyn FnMut(&str)`).
pub fn drive<A: Act, G: Ground>(actor: &mut A, grounder: &G, plan: &Plan, progress: &mut dyn FnMut(&str)) -> Result<Summary, ScreenError> {
    match drive_inner(actor, grounder, plan, progress) {
        Ok(s) => Ok(s),
        Err(e) => {
            // every exit is spoken: a silent agent thread is undebuggable from outside
            progress(&format!("screenagent: giving up: {e}"));
            Err(e)
        }
    }
}

fn drive_inner<A: Act, G: Ground>(actor: &mut A, grounder: &G, plan: &Plan, progress: &mut dyn FnMut(&str)) -> Result<Summary, ScreenError> {
    let started = Instant::now();
    let mut summary = Summary::default();
    let mut stalled = 0u32;
    let deadline = || started.elapsed() >= plan.timeout;

    actor.activate()?;
    actor.sleep(600);
    let mut cap = actor.capture()?;

    // 1. search — for videos always, so Google's own filter leaves only videos in the grid
    if let Some(q) = picker::search_text(plan.query.as_deref(), plan.videos_only) {
        let q = q.as_str();
        if spend(&mut summary, plan)? || deadline() {
            return Err(budget(&summary));
        }
        match grounder.find(&cap.png, picker::SEARCH_FIELD) {
            Ok(Some(field)) => {
                let (px, py) = field.center();
                progress(&format!("screenagent: typing `{q}` into the picker search"));
                actor.click_png(&cap, px, py)?;
                actor.sleep(400);
                actor.type_text(q)?;
                actor.key_return()?;
                summary.searched = true;
                actor.sleep(SEARCH_SETTLE_MS);
                cap = actor.capture()?;
            }
            Ok(None) => progress("screenagent: no search field visible; using the grid as-is"),
            Err(e) => return Err(e),
        }
    }

    // 2. select tiles
    let mut tried: Vec<(u32, u32)> = Vec::new();
    while summary.selected < plan.max_items {
        if spend(&mut summary, plan)? || deadline() {
            return Err(budget(&summary));
        }
        let tiles = visible_tiles(grounder, &cap, plan.videos_only, progress)?;
        let fresh: Vec<&crate::ground::Element> = tiles
            .iter()
            .filter(|t| {
                let (px, py) = picker::click_point(t);
                !tried.iter().any(|&(x, y)| x.abs_diff(px) < SAME_SPOT && y.abs_diff(py) < SAME_SPOT)
            })
            .take(plan.max_items.saturating_sub(summary.selected).max(1) as usize)
            .collect();
        if fresh.is_empty() {
            // nothing new at this scroll position: page down once, or finish
            if spend(&mut summary, plan)? {
                return Err(budget(&summary));
            }
            let before = cap.png.clone();
            actor.scroll(SCROLL_LINES)?;
            actor.sleep(900);
            let next = actor.capture()?;
            if !verify::changed(&before, &next.png)? {
                stalled += 1;
                if stalled >= MAX_STALLED {
                    break; // bottom of the grid: go confirm what we have
                }
            } else {
                stalled = 0;
            }
            cap = next;
            continue;
        }
        for t in fresh {
            if summary.selected >= plan.max_items || spend(&mut summary, plan)? {
                break;
            }
            let (px, py) = picker::click_point(t);
            tried.push((px, py));
            let before = cap.png.clone();
            if let Err(e) = actor.click_png(&cap, px, py) {
                // a mis-grounded tile is one bad candidate, not a dead run: skip it
                progress(&format!("screenagent: skipped a tile it could not click: {e}"));
                stalled += 1;
                continue;
            }
            actor.sleep(CLICK_SETTLE_MS);
            let after = match actor.capture() {
                Ok(c) => c,
                Err(_) => {
                    // the window vanishing mid-selection is the autoclose page doing its thing
                    summary.selected += 1;
                    summary.done_pressed = true;
                    summary.note = "the picker window closed during selection".into();
                    return Ok(summary);
                }
            };
            if verify::changed(&before, &after.png)? {
                summary.selected += 1;
                stalled = 0;
                progress(&format!("screenagent: selected {} item(s)", summary.selected));
            } else {
                stalled += 1;
                if stalled >= MAX_STALLED {
                    summary.note = "clicks stopped changing the screen (if none ever did: is Accessibility granted to this terminal?)".into();
                    break;
                }
            }
            cap = after;
        }
    }

    // 3. confirm
    if summary.selected > 0 && !summary.done_pressed {
        if spend(&mut summary, plan)? {
            return Err(budget(&summary));
        }
        let mut pressed = false;
        for label in picker::CONFIRM_LABELS {
            if deadline() {
                return Err(budget(&summary));
            }
            match grounder.find(&cap.png, label) {
                Ok(Some(btn)) => {
                    let (px, py) = btn.center();
                    progress("screenagent: pressing the confirm button");
                    actor.click_png(&cap, px, py)?;
                    pressed = true;
                    break;
                }
                Ok(None) => {}
                Err(e) => return Err(e),
            }
        }
        if !pressed {
            return Err(ScreenError::Stalled("could not find the confirm (Done) button".into()));
        }
        summary.done_pressed = true;
        actor.sleep(1200);
    }

    if summary.note.is_empty() {
        summary.note = if summary.done_pressed { format!("selected {} item(s) and pressed Done", summary.selected) } else { "nothing selected".into() };
    }
    progress(&format!("screenagent: {}", summary.note));
    Ok(summary)
}

/// Count an action; `Ok(true)` = budget exhausted.
fn spend(summary: &mut Summary, plan: &Plan) -> Result<bool, ScreenError> {
    summary.actions += 1;
    if summary.actions > plan.max_actions {
        return Ok(true);
    }
    Ok(false)
}

fn budget(summary: &Summary) -> ScreenError {
    ScreenError::Stalled(format!("ran out of budget after {} actions ({} selected)", summary.actions, summary.selected))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ground::Element;
    use crate::{capture::Capture, capture::WindowRect};

    /// A deterministic PNG the fakes can hand around (content is irrelevant to the loop).
    fn frame(tag: u8) -> Vec<u8> {
        let mut img = image::GrayImage::from_pixel(200, 150, image::Luma([tag]));
        img.get_pixel_mut(5, 5).0[0] = tag;
        let mut buf = std::io::Cursor::new(Vec::new());
        img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        buf.into_inner()
    }

    fn cap_of(tag: u8) -> Capture {
        Capture { png: frame(tag), rect: WindowRect { x: 0, y: 0, w: 200, h: 150 }, px_per_point: 1.0, png_w: 200, png_h: 150 }
    }

    struct FakeActor {
        caps: Vec<Capture>,
        i: usize,
        clicks: Vec<(u32, u32)>,
        typed: Vec<String>,
        returns: u32,
        scrolls: i32,
        fail_capture_at: Option<usize>,
    }
    impl FakeActor {
        fn new(caps: Vec<Capture>) -> Self {
            Self { caps, i: 0, clicks: vec![], typed: vec![], returns: 0, scrolls: 0, fail_capture_at: None }
        }
    }
    impl Act for FakeActor {
        fn activate(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn click_png(&mut self, _cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            self.clicks.push((px, py));
            self.i = (self.i + 1).min(self.caps.len() - 1);
            Ok(())
        }
        fn type_text(&mut self, text: &str) -> Result<(), ScreenError> {
            self.typed.push(text.into());
            Ok(())
        }
        fn key_return(&mut self) -> Result<(), ScreenError> {
            self.returns += 1;
            Ok(())
        }
        fn key_escape(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn scroll(&mut self, n: i32) -> Result<(), ScreenError> {
            self.scrolls += n;
            self.i = (self.i + 1).min(self.caps.len() - 1);
            Ok(())
        }
        fn sleep(&mut self, _ms: u64) {}
        fn capture(&mut self) -> Result<Capture, ScreenError> {
            if self.fail_capture_at == Some(self.i) {
                return Err(ScreenError::Capture("x".into(), "window closed".into()));
            }
            Ok(self.caps[self.i].clone())
        }
    }

    /// Serves a fixed inventory and scripted `find`s; unscripted targets are absent.
    struct FakeGrounder {
        inventory: Vec<Element>,
        finds: Vec<(String, Option<Element>)>,
    }
    impl Ground for FakeGrounder {
        fn inventory(&self, _png: &[u8], _hint: &str) -> Result<Vec<Element>, ScreenError> {
            Ok(self.inventory.clone())
        }
        fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            Ok(self.finds.iter().find(|(t, _)| t == target).and_then(|(_, e)| e.clone()))
        }
    }

    fn tile(label: &str, x: u32) -> Element {
        Element { label: label.into(), kind: "thumbnail".into(), x1: x, y1: 10, x2: x + 90, y2: 100 }
    }

    #[test]
    fn selects_tiles_and_presses_done() {
        // script: capture 0 = grid; each click advances to a changed capture; confirm find scripted
        let g = FakeGrounder {
            inventory: vec![tile("0:30", 10), tile("0:06", 120)],
            finds: vec![(
                "the Done button (top right, blue/enabled when items are selected)".to_string(),
                Some(Element { label: "Done".into(), kind: "button".into(), x1: 180, y1: 0, x2: 199, y2: 20 }),
            )],
        };
        // three visually distinct captures so each click visibly lands
        let caps = vec![cap_of(10), cap_of(200), cap_of(90)];
        let mut a = FakeActor::new(caps);
        let plan = Plan { max_items: 2, ..Plan::default() };
        let mut lines: Vec<String> = vec![];
        let s = drive(&mut a, &g, &plan, &mut |l| lines.push(l.into())).unwrap();
        assert_eq!(s.selected, 2);
        assert!(s.done_pressed);
        assert_eq!(a.clicks.len(), 3); // 2 tiles + Done
        assert!(a.clicks.iter().all(|(x, y)| *x < 200 && *y < 150));
        assert!(lines.iter().any(|l| l.contains("pressing the confirm button")));
    }

    #[test]
    fn search_types_query_first() {
        let g = FakeGrounder {
            inventory: vec![],
            finds: vec![(
                crate::picker::SEARCH_FIELD.to_string(),
                Some(Element { label: "search".into(), kind: "field".into(), x1: 0, y1: 0, x2: 100, y2: 20 }),
            )],
        };
        let mut a = FakeActor::new(vec![cap_of(10), cap_of(11), cap_of(12)]);
        let plan = Plan { query: Some("beach".into()), max_items: 1, ..Plan::default() };
        // no tiles anywhere → the loop scrolls, stalls, breaks, and returns with nothing
        let s = drive(&mut a, &g, &plan, &mut |_| {}).unwrap();
        assert!(s.searched);
        assert_eq!(a.typed, vec!["beach videos".to_string()]); // Google filters the grid to videos
        assert!(a.returns == 1);
        assert_eq!(s.selected, 0);
        assert!(!s.done_pressed);
    }

    #[test]
    fn garbage_grounder_gives_up_within_budget() {
        struct Garbage;
        impl Ground for Garbage {
            fn inventory(&self, _p: &[u8], _h: &str) -> Result<Vec<Element>, ScreenError> {
                Ok(vec![]) // nothing ever found
            }
            fn find(&self, _p: &[u8], _t: &str) -> Result<Option<Element>, ScreenError> {
                Ok(None)
            }
        }
        let mut a = FakeActor::new(vec![cap_of(10), cap_of(11)]);
        let plan = Plan { max_actions: 10, ..Plan::default() };
        let s = drive(&mut a, &Garbage, &plan, &mut |_| {}).unwrap();
        assert_eq!(s.selected, 0);
        assert!(a.clicks.is_empty(), "nothing may be clicked blind");
        assert!(s.actions <= 10, "budget respected: {}", s.actions);
    }

    #[test]
    fn hostile_actor_failures_are_errors_not_panics() {
        struct Exploding;
        impl Act for Exploding {
            fn activate(&mut self) -> Result<(), ScreenError> {
                Err(ScreenError::Input("Google Chrome".into(), "no assistive access".into()))
            }
            fn click_png(&mut self, _: &Capture, _: u32, _: u32) -> Result<(), ScreenError> {
                Err(ScreenError::Input("Google Chrome".into(), "refused".into()))
            }
            fn type_text(&mut self, _: &str) -> Result<(), ScreenError> {
                Err(ScreenError::Input("Google Chrome".into(), "refused".into()))
            }
            fn key_return(&mut self) -> Result<(), ScreenError> {
                Err(ScreenError::Input("Google Chrome".into(), "refused".into()))
            }
            fn key_escape(&mut self) -> Result<(), ScreenError> {
                Ok(())
            }
            fn scroll(&mut self, _: i32) -> Result<(), ScreenError> {
                Ok(())
            }
            fn sleep(&mut self, _: u64) {}
            fn capture(&mut self) -> Result<Capture, ScreenError> {
                Ok(cap_of(1))
            }
        }
        let g = FakeGrounder { inventory: vec![], finds: vec![] };
        let mut a = Exploding;
        let err = drive(&mut a, &g, &Plan::default(), &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("Accessibility"), "{err}");
    }

    /// Answers each inventory call with the next scripted list (the last one repeats) and
    /// records the asks.
    struct SequencedGrounder {
        rounds: Vec<Vec<Element>>,
        asks: std::cell::RefCell<Vec<String>>,
    }
    impl Ground for SequencedGrounder {
        fn inventory(&self, _png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
            let mut asks = self.asks.borrow_mut();
            asks.push(task.to_string());
            let i = (asks.len() - 1).min(self.rounds.len() - 1);
            Ok(self.rounds[i].clone())
        }
        fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            Ok(target.contains("Done").then(|| Element { label: "Done".into(), kind: String::new(), x1: 180, y1: 0, x2: 199, y2: 20 }))
        }
    }

    #[test]
    fn a_tile_down_right_of_a_clicked_one_is_still_fresh() {
        // regression: `tried` compared positions with one-sided saturating_sub, so once one
        // tile was clicked every tile below/right of it counted as tried and was never clicked
        let a = Element { label: "0:30".into(), kind: String::new(), x1: 10, y1: 10, x2: 60, y2: 60 };
        let b = Element { label: "0:06".into(), kind: String::new(), x1: 100, y1: 80, x2: 150, y2: 130 };
        let g = SequencedGrounder { rounds: vec![vec![a.clone()], vec![a, b]], asks: Default::default() };
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(200), cap_of(90)]);
        let plan = Plan { max_items: 2, ..Plan::default() };
        let s = drive(&mut act, &g, &plan, &mut |_| {}).unwrap();
        assert_eq!(s.selected, 2, "{s:?}");
        assert_eq!(act.clicks.len(), 3); // a, b, Done
        assert_eq!(act.clicks[1], crate::picker::click_point(&Element { label: String::new(), kind: String::new(), x1: 100, y1: 80, x2: 150, y2: 130 }));
    }

    #[test]
    fn a_lumped_grid_is_asked_again_once() {
        // the live failure: one box around the whole grid labelled with the task
        let lump = Element { label: "media thumbnails to select".into(), kind: String::new(), x1: 0, y1: 0, x2: 200, y2: 150 };
        let tile = Element { label: "0:30".into(), kind: String::new(), x1: 10, y1: 10, x2: 60, y2: 60 };
        let g = SequencedGrounder { rounds: vec![vec![lump], vec![tile]], asks: Default::default() };
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(200), cap_of(90)]);
        let plan = Plan { max_items: 1, ..Plan::default() };
        let mut lines: Vec<String> = vec![];
        let s = drive(&mut act, &g, &plan, &mut |l| lines.push(l.into())).unwrap();
        assert_eq!(s.selected, 1);
        assert!(s.done_pressed);
        let asks = g.asks.borrow();
        assert_eq!(asks[0], crate::picker::VIDEO_TILES);
        assert!(asks[1].ends_with(crate::picker::STRICT), "{asks:?}");
        assert!(lines.iter().any(|l| l.contains("asking again")));
        // the lump itself was never clicked
        assert_eq!(act.clicks[0], crate::picker::click_point(&g.rounds[1][0]));
    }

    #[test]
    fn window_vanishing_mid_selection_counts_as_done() {
        let g = FakeGrounder { inventory: vec![tile("0:30", 10)], finds: vec![] };
        let mut a = FakeActor::new(vec![cap_of(10), cap_of(20)]);
        a.fail_capture_at = Some(1); // the re-capture after the first click fails
        let plan = Plan::default();
        let s = drive(&mut a, &g, &plan, &mut |_| {}).unwrap();
        assert!(s.done_pressed);
        assert_eq!(s.selected, 1);
    }
}
