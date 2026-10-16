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
    /// Select every result instead of `max_items` tiles: click the first tile, wheel to the
    /// end of the grid and Shift+click the last one (the picker's own range selection).
    pub select_all: bool,
    /// Pick every item of the user's album with this title instead of searching: open the
    /// picker's Albums view (or search the title, Google's documented way to an album), open
    /// the album whose label matches, select all of it, press Done. No matching album → an
    /// error, with nothing selected. `query` is ignored; `select_all` is implied.
    pub album: Option<String>,
}

impl Default for Plan {
    fn default() -> Self {
        Self { query: None, max_items: 6, videos_only: true, max_actions: 40, timeout: Duration::from_secs(300), select_all: false, album: None }
    }
}

/// What a caller asks of [`spawn`].
#[derive(Debug, Clone, Default)]
pub struct Request {
    pub query: String,
    pub max_items: u32,
    pub select_all: bool,
    /// Pick the whole album with this title instead (see [`Plan::album`]); `query`,
    /// `max_items` and `select_all` are then unused.
    pub album: Option<String>,
    /// Only video tiles (searches ask Google for videos; in an album, the range is grounded
    /// on video tiles — the caller drops any photo in between when it downloads).
    pub videos_only: bool,
}

/// What the agent did.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Summary {
    pub actions: u32,
    pub selected: u32,
    pub searched: bool,
    /// The album of [`Plan::album`] was found and opened.
    pub album_opened: bool,
    pub done_pressed: bool,
    pub note: String,
}

/// How long a pick keeps waiting for Google to report the selection after the agent pressed Done.
pub const DONE_GRACE: Duration = Duration::from_secs(120);

impl Summary {
    /// Whether waiting for the picker's selection is pointless, the agent having finished
    /// `since` ago: it never pressed Done (the search matched nothing, or nothing could be
    /// selected), or it did and Google still has not reported a selection after [`DONE_GRACE`].
    pub fn stop_waiting(&self, since: Duration) -> bool {
        !self.done_pressed || since >= DONE_GRACE
    }
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
pub fn spawn(req: Request) -> std::thread::JoinHandle<Result<Summary, String>> {
    std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(2500)); // let the picker load
        let shell = crate::capture::OsShell;
        let app = crate::capture::frontmost_app(&shell).map_err(|e| e.to_string())?;
        let cfg = crate::Config::from_env();
        let grounder = crate::ground::QwenGrounder::from_config(&cfg);
        if !grounder.model_available().map_err(|e| e.to_string())? {
            return Err(format!("the grounding model `{}` is not in Ollama — `ollama pull {}` first", cfg.model, cfg.model));
        }
        let plan = plan_for(req);
        let mut actor = crate::act::MacActor::new(app, Box::new(shell));
        drive(&mut actor, &grounder, &plan, &mut |l| eprintln!("{l}")).map_err(|e| e.to_string())
    })
}

/// The plan for a caller's request: an album pick selects the whole album (and has a few more
/// actions to get there); otherwise the query is searched.
pub fn plan_for(req: Request) -> Plan {
    match req.album {
        Some(album) => Plan { query: None, album: Some(album), select_all: true, videos_only: req.videos_only, max_actions: 60, ..Plan::default() },
        None => Plan { query: Some(req.query), max_items: req.max_items, select_all: req.select_all, videos_only: req.videos_only, ..Plan::default() },
    }
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

/// Looks for the picker's search field before going on without it, and the wait between them.
const SEARCH_TRIES: usize = 4;
const SEARCH_RETRY_MS: u64 = 1500;

fn drive_inner<A: Act, G: Ground>(actor: &mut A, grounder: &G, plan: &Plan, progress: &mut dyn FnMut(&str)) -> Result<Summary, ScreenError> {
    let started = Instant::now();
    let mut summary = Summary::default();
    let mut stalled = 0u32;
    let deadline = || started.elapsed() >= plan.timeout;

    actor.activate()?;
    actor.sleep(600);
    let mut cap = actor.capture()?;

    // 1. the album, opened (or an error with nothing selected) — or the search, for videos
    // always, so Google's own filter leaves only videos in the grid
    if let Some(title) = plan.album.as_deref() {
        open_album(actor, grounder, plan, &mut summary, &mut cap, title, started, progress)?;
    } else if let Some(q) = picker::search_text(plan.query.as_deref(), plan.videos_only) {
        let q = q.as_str();
        if spend(&mut summary, plan)? || deadline() {
            return Err(budget(&summary));
        }
        // the picker may still be loading: look again a few times before going on without it
        let mut field = None;
        for attempt in 0..SEARCH_TRIES {
            field = grounder.find(&cap.png, picker::SEARCH_FIELD)?;
            if field.is_some() || attempt + 1 == SEARCH_TRIES {
                break;
            }
            progress("screenagent: no search field yet; waiting for the picker");
            actor.sleep(SEARCH_RETRY_MS);
            cap = actor.capture()?;
        }
        match field {
            Some(field) => {
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
            // selecting "everything" without the search would select the whole library
            None if plan.select_all => {
                return Err(ScreenError::Other(format!(
                    "the picker's search field never appeared, so `{q}` could not be searched; refusing to select every item of the whole library (selecting all needs a search)"
                )));
            }
            None => progress("screenagent: no search field visible; using the grid as-is"),
        }
    }

    // 2. select tiles — every result (of the search, or of the album), or up to max_items
    let select_all = plan.select_all || plan.album.is_some();
    if select_all {
        // only ever inside a search result or an opened album: never the whole library
        if !summary.searched && !summary.album_opened {
            return Err(ScreenError::Other(
                "nothing was searched and no album was opened; refusing to select every item of the whole library (selecting all needs a search or an album)"
                    .into(),
            ));
        }
        select_everything(actor, grounder, plan, &mut summary, &mut cap, started, progress)?;
    }
    let mut tried: Vec<(u32, u32)> = Vec::new();
    while !select_all && summary.selected < plan.max_items {
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

    // 3. confirm, and check the picker took it
    if summary.selected > 0 && !summary.done_pressed {
        let mut button = None;
        for label in picker::CONFIRM_LABELS {
            if deadline() {
                return Err(budget(&summary));
            }
            if let Some(btn) = grounder.find(&cap.png, label)? {
                button = Some((label, btn));
                break;
            }
        }
        let Some((label, btn)) = button else {
            return Err(ScreenError::Stalled("could not find the confirm (Done) button".into()));
        };
        confirm(actor, grounder, plan, &mut summary, &mut cap, label, btn, started, progress)?;
        summary.done_pressed = true;
    }

    if summary.note.is_empty() {
        summary.note = if summary.done_pressed { format!("selected {} item(s) and pressed Done", summary.selected) } else { "nothing selected".into() };
    }
    progress(&format!("screenagent: {}", summary.note));
    Ok(summary)
}

/// How long the picker gets to act on Done before the agent looks whether it did.
const DONE_SETTLE_MS: u64 = 1800;
/// Presses of Done after the first, when the picker did not take it.
const DONE_RETRIES: u32 = 2;

/// Press the confirm button `btn` (found by `label`) and check the picker took it: after
/// [`DONE_SETTLE_MS`] the window is gone (the autoclose page closed it), or the button is, or
/// the "N selected" counter is (the picker left its grid; its finished page says "Done!", which
/// could pass for the button, but never "N selected"). While both are still there the press did
/// not take — the button not yet enabled after a range click, the header still animating — and
/// it is pressed again, at most [`DONE_RETRIES`] more times, each from the budget.
#[allow(clippy::too_many_arguments)]
fn confirm<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    label: &str,
    mut btn: crate::ground::Element,
    started: Instant,
    progress: &mut dyn FnMut(&str),
) -> Result<(), ScreenError> {
    for press in 0..=DONE_RETRIES {
        if spend(summary, plan)? || started.elapsed() >= plan.timeout {
            return Err(budget(summary));
        }
        progress(if press == 0 { "screenagent: pressing the confirm button" } else { "screenagent: the picker did not take Done; pressing it again" });
        let (px, py) = btn.center();
        actor.click_png(cap, px, py)?;
        actor.sleep(DONE_SETTLE_MS);
        *cap = match actor.capture() {
            Ok(c) => c,
            // the window vanishing is the autoclose page doing its thing: taken
            Err(_) => return Ok(()),
        };
        let Some(still) = grounder.find(&cap.png, label)? else { return Ok(()) };
        let counter = grounder.inventory(&cap.png, picker::SELECTED_COUNTER)?;
        if !counter.iter().any(|e| picker::selected_count(&e.label).is_some()) {
            return Ok(());
        }
        btn = still;
    }
    Err(ScreenError::Stalled(format!(
        "the picker did not accept Done: pressed {} times and it still shows the selection with its Done button",
        DONE_RETRIES + 1
    )))
}

/// An album view (or an opened album) needs this long to render.
const ALBUM_SETTLE_MS: u64 = 2000;
/// Search suggestions need this long to appear under the field.
const SUGGEST_SETTLE_MS: u64 = 1500;
/// Pages of an albums view looked through for the album.
const ALBUM_PAGES: u32 = 4;

/// How the picker leads to an album.
enum WayIn {
    /// An Albums view (tab or button) to switch to.
    Tab(crate::ground::Element),
    /// The search field: Google's documented way ("search for the album").
    Search(crate::ground::Element),
}

/// The albums named `title` in the current capture (see [`picker::album_candidates`]).
fn albums_named<G: Ground>(
    grounder: &G,
    cap: &crate::capture::Capture,
    title: &str,
    avoid: Option<&crate::ground::Element>,
) -> Result<Vec<crate::ground::Element>, ScreenError> {
    let els = grounder.inventory(&cap.png, picker::ALBUM_TILES)?;
    Ok(picker::album_candidates(&els, title, avoid, cap.png_w, cap.png_h))
}

/// Open the user's album titled `title`: through the picker's Albums view when it has one
/// (paging down it), else by searching the title (its suggestions, then its results). Only an
/// element whose label names the album is ever clicked; no match, or two albums with that
/// title, is an error before anything is selected. The opened screen must be headed by the
/// title, so a click that opened something else selects nothing.
#[allow(clippy::too_many_arguments)]
fn open_album<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    title: &str,
    started: Instant,
    progress: &mut dyn FnMut(&str),
) -> Result<(), ScreenError> {
    if picker::normalize_title(title).is_empty() {
        return Err(ScreenError::Other(format!("the album title `{title}` has no letters or digits to look for")));
    }
    let out_of_time = || started.elapsed() >= plan.timeout;
    // 1. the way in, looked for a few times while the picker loads
    let mut way = None;
    for attempt in 0..SEARCH_TRIES {
        if let Some(tab) = grounder.find(&cap.png, picker::ALBUMS_TAB)?.filter(|e| picker::is_albums_tab(e, cap.png_w, cap.png_h)) {
            way = Some(WayIn::Tab(tab));
            break;
        }
        if let Some(field) = grounder.find(&cap.png, picker::SEARCH_FIELD)? {
            way = Some(WayIn::Search(field));
            break;
        }
        if attempt + 1 < SEARCH_TRIES {
            progress("screenagent: no Albums view or search field yet; waiting for the picker");
            actor.sleep(SEARCH_RETRY_MS);
            *cap = actor.capture()?;
        }
    }
    // 2. the albums named `title`
    let found = match way {
        None => {
            return Err(ScreenError::Other(format!(
                "the picker showed neither an Albums view nor its search field, so the album `{title}` could not be opened; nothing was selected"
            )));
        }
        Some(WayIn::Tab(tab)) => {
            if spend(summary, plan)? || out_of_time() {
                return Err(budget(summary));
            }
            progress("screenagent: opening the picker's Albums view");
            let (px, py) = tab.center();
            actor.click_png(cap, px, py)?;
            actor.sleep(ALBUM_SETTLE_MS);
            *cap = actor.capture()?;
            let mut found = Vec::new();
            for page in 0..ALBUM_PAGES {
                found = albums_named(grounder, cap, title, None)?;
                if !found.is_empty() || page + 1 == ALBUM_PAGES {
                    break;
                }
                if spend(summary, plan)? || out_of_time() {
                    return Err(budget(summary));
                }
                let before = cap.png.clone();
                actor.scroll(SCROLL_LINES)?;
                actor.sleep(900);
                *cap = actor.capture()?;
                if !verify::changed(&before, &cap.png)? {
                    break; // the end of the albums
                }
            }
            found
        }
        Some(WayIn::Search(field)) => {
            if spend(summary, plan)? || out_of_time() {
                return Err(budget(summary));
            }
            progress(&format!("screenagent: searching the picker for the album `{title}`"));
            let (px, py) = field.center();
            actor.click_png(cap, px, py)?;
            actor.sleep(400);
            actor.type_text(title)?;
            summary.searched = true;
            actor.sleep(SUGGEST_SETTLE_MS);
            *cap = actor.capture()?;
            // the album among the suggestions under the field, else among the results
            let mut found = albums_named(grounder, cap, title, Some(&field))?;
            if found.is_empty() {
                if spend(summary, plan)? || out_of_time() {
                    return Err(budget(summary));
                }
                actor.key_return()?;
                actor.sleep(SEARCH_SETTLE_MS);
                *cap = actor.capture()?;
                found = albums_named(grounder, cap, title, Some(&field))?;
            }
            found
        }
    };
    let album = match found.as_slice() {
        [] => {
            return Err(ScreenError::Other(format!(
                "no album titled `{title}` showed up in the picker; nothing was selected (the title must match the album's; is it in this Google account?)"
            )));
        }
        [one] => one.clone(),
        many => {
            return Err(ScreenError::Other(format!(
                "{} albums titled `{title}` are on screen; not choosing between them (rename one, or pick by hand); nothing was selected",
                many.len()
            )));
        }
    };
    // 3. open it, and check it is the album
    if spend(summary, plan)? || out_of_time() {
        return Err(budget(summary));
    }
    progress(&format!("screenagent: opening the album `{title}`"));
    let before = cap.png.clone();
    let (px, py) = album.center();
    actor.click_png(cap, px, py)?;
    actor.sleep(ALBUM_SETTLE_MS);
    *cap = actor.capture()?;
    if !verify::changed(&before, &cap.png)? {
        return Err(ScreenError::Stalled(format!("clicking the album `{title}` changed nothing; nothing was selected")));
    }
    let heading = grounder.inventory(&cap.png, picker::ALBUM_HEADING)?;
    if !heading.iter().any(|e| picker::heading_names(&e.label, title)) {
        return Err(ScreenError::Other(format!("the screen that opened is not headed `{title}`; nothing was selected")));
    }
    summary.album_opened = true;
    progress(&format!("screenagent: opened the album `{title}`; selecting all of it"));
    Ok(())
}

/// Wheel lines per scroll while looking for the end of the grid.
const END_SCROLL_LINES: i32 = SCROLL_LINES * 3;

/// Select every result with the picker's range gesture: click the first tile, wheel to the end
/// of the grid (until two scrolls change nothing — the grid loads lazily), Shift+click the
/// last tile. Two clicks select a whole search result, where tile-by-tile selection misses
/// the tiles the model does not box. The selected count is read off the picker's header.
fn select_everything<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    progress: &mut dyn FnMut(&str),
) -> Result<(), ScreenError> {
    let out_of_time = || started.elapsed() >= plan.timeout;
    // 1. the first tile, in reading order
    let mut first = None;
    let mut first_page = 0usize;
    for _ in 0..MAX_STALLED {
        let tiles = visible_tiles(grounder, cap, plan.videos_only, progress)?;
        first_page = tiles.len();
        if let Some(t) = tiles.into_iter().next() {
            first = Some(t);
            break;
        }
        if spend(summary, plan)? || out_of_time() {
            return Err(budget(summary));
        }
        actor.scroll(SCROLL_LINES)?;
        actor.sleep(900);
        *cap = actor.capture()?;
    }
    let Some(first) = first else {
        summary.note = "no tiles to select".into();
        return Ok(());
    };
    if spend(summary, plan)? {
        return Err(budget(summary));
    }
    let (px, py) = picker::click_point(&first);
    let before = cap.png.clone();
    actor.click_png(cap, px, py)?;
    actor.sleep(CLICK_SETTLE_MS);
    *cap = actor.capture()?;
    if !verify::changed(&before, &cap.png)? {
        return Err(ScreenError::Stalled("clicking the first tile changed nothing (is Accessibility granted to this terminal?)".into()));
    }
    summary.selected = 1;
    progress("screenagent: selected the first result; scrolling to the end of the grid");
    // 2. to the end of the grid
    let mut still = 0u32;
    let mut scrolled = false;
    while still < 2 {
        if spend(summary, plan)? || out_of_time() {
            progress("screenagent: out of budget before the end of the grid; selecting up to here");
            break;
        }
        let before = cap.png.clone();
        actor.scroll(END_SCROLL_LINES)?;
        actor.sleep(1200);
        *cap = actor.capture()?;
        if verify::changed(&before, &cap.png)? {
            still = 0;
            scrolled = true;
        } else {
            still += 1;
        }
    }
    // 3. Shift+click the last tile
    let tiles = visible_tiles(grounder, cap, plan.videos_only, progress)?;
    let band = (cap.png_h / 16).max(1);
    let Some(last) = tiles.iter().max_by_key(|e| (e.y1 / band, e.x1)) else {
        summary.note = "found no tile at the end of the grid; only the first is selected".into();
        return Ok(());
    };
    // the last tile is the first one (a grid of one): a Shift+click on it would toggle the
    // selection back off — the first click already selected it
    if picker::same_tile(&first, last, scrolled) || (!scrolled && first_page == 1 && tiles.len() == 1) {
        summary.note = "the grid holds one item; the first click selected it".into();
        progress("screenagent: the grid holds one item, already selected");
        return Ok(());
    }
    if spend(summary, plan)? {
        return Err(budget(summary));
    }
    let (px, py) = picker::click_point(last);
    let before = cap.png.clone();
    actor.shift_click_png(cap, px, py)?;
    actor.sleep(CLICK_SETTLE_MS * 2);
    *cap = actor.capture()?;
    if !verify::changed(&before, &cap.png)? {
        summary.note = "the range click changed nothing; only the first tile is selected".into();
        return Ok(());
    }
    summary.selected = 2;
    // the header says how many: "37 selected"
    if let Ok(els) = grounder.inventory(&cap.png, picker::SELECTED_COUNTER)
        && let Some(n) = els.iter().find_map(|e| picker::selected_count(&e.label))
    {
        summary.selected = n.max(2);
    }
    progress(&format!("screenagent: selected {} item(s) with a range click", summary.selected));
    Ok(())
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
        shift_clicks: Vec<(u32, u32)>,
        /// What the screen shows after a Shift+click (the range selected).
        after_shift: Option<Capture>,
        typed: Vec<String>,
        returns: u32,
        scrolls: i32,
        fail_capture_at: Option<usize>,
    }
    impl FakeActor {
        fn new(caps: Vec<Capture>) -> Self {
            Self { caps, i: 0, clicks: vec![], shift_clicks: vec![], after_shift: None, typed: vec![], returns: 0, scrolls: 0, fail_capture_at: None }
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
        fn shift_click_png(&mut self, _cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            self.shift_clicks.push((px, py));
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
            if !self.shift_clicks.is_empty()
                && let Some(c) = &self.after_shift
            {
                return Ok(c.clone());
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
            fn shift_click_png(&mut self, _: &Capture, _: u32, _: u32) -> Result<(), ScreenError> {
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
    fn select_all_clicks_first_scrolls_to_the_end_and_shift_clicks_last() {
        /// Counts the counter reads: the range's, then (once Done took) none — the picker left
        /// its grid.
        struct Grid(std::cell::Cell<u32>);
        impl Ground for Grid {
            fn inventory(&self, _png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
                if task == crate::picker::SELECTED_COUNTER {
                    self.0.set(self.0.get() + 1);
                    if self.0.get() > 1 {
                        return Ok(vec![]);
                    }
                    return Ok(vec![Element { label: "37 selected".into(), kind: String::new(), x1: 5, y1: 2, x2: 60, y2: 12 }]);
                }
                let tile = |l: &str, x1: u32, y1: u32| Element { label: l.into(), kind: String::new(), x1, y1, x2: x1 + 40, y2: y1 + 40 };
                // three rows; the last row's rightmost tile is the end of the range
                Ok(vec![tile("0:12", 120, 100), tile("0:05", 10, 10), tile("1:01", 60, 100), tile("0:30", 120, 10)])
            }
            fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
                if target == crate::picker::SEARCH_FIELD {
                    return Ok(Some(Element { label: "Search".into(), kind: String::new(), x1: 40, y1: 0, x2: 160, y2: 8 }));
                }
                Ok(target.contains("Done").then(|| Element { label: "Done".into(), kind: String::new(), x1: 180, y1: 0, x2: 199, y2: 20 }))
            }
        }
        // the search, the grid, first tile selected, two scrolled pages, then the end (captures
        // stop changing)
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(5), cap_of(200), cap_of(90), cap_of(150)]);
        act.after_shift = Some(cap_of(30));
        let plan = Plan { select_all: true, ..Plan::default() };
        let mut lines: Vec<String> = vec![];
        let grid = Grid(Default::default());
        let s = drive(&mut act, &grid, &plan, &mut |l| lines.push(l.into())).unwrap();
        let tile = |x1: u32, y1: u32| Element { label: String::new(), kind: String::new(), x1, y1, x2: x1 + 40, y2: y1 + 40 };
        // the search field, then the first tile
        assert_eq!(act.clicks.get(1), Some(&crate::picker::click_point(&tile(10, 10))), "the first tile: {:?}", act.clicks);
        assert_eq!(act.shift_clicks, vec![crate::picker::click_point(&tile(120, 100))], "the last tile");
        assert!(act.scrolls > 0, "it scrolled to the end");
        assert_eq!(s.selected, 37, "the header count: {s:?}");
        assert!(s.done_pressed);
        // search field, first tile, Done once (it took: the counter was gone after it)
        assert_eq!(act.clicks.len(), 3, "{:?}", act.clicks);
        assert_eq!(grid.0.get(), 2, "the range's count, then the check after Done");
        assert!(lines.iter().any(|l| l.contains("range click")), "{lines:?}");
    }

    /// A search result of one video (and a photo): the header reads "1 selected" until the
    /// capture after Done (`gone`), when the picker has left its grid.
    struct OneVideo {
        gone: Vec<u8>,
    }
    impl Ground for OneVideo {
        fn inventory(&self, png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
            if task == crate::picker::SELECTED_COUNTER {
                return Ok(if png == self.gone.as_slice() {
                    vec![]
                } else {
                    vec![Element { label: "1 selected".into(), kind: String::new(), x1: 5, y1: 2, x2: 60, y2: 12 }]
                });
            }
            let t = |l: &str, x1: u32| Element { label: l.into(), kind: String::new(), x1, y1: 40, x2: x1 + 40, y2: 80 };
            Ok(vec![t("0:07", 10), t("photo", 60)])
        }
        fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            if target == crate::picker::SEARCH_FIELD {
                return Ok(Some(Element { label: "Search".into(), kind: String::new(), x1: 40, y1: 0, x2: 160, y2: 8 }));
            }
            Ok(target.contains("Done").then(|| Element { label: "Done".into(), kind: String::new(), x1: 180, y1: 0, x2: 199, y2: 20 }))
        }
    }

    #[test]
    fn a_grid_of_one_is_not_range_clicked_off_again() {
        // the live failure (an "October 15 2018" search): one video tile; the range ended on the
        // tile it started on, the Shift+click toggled it off, and Done was pressed with nothing
        // selected. The grid never scrolls (captures stop changing after the first click).
        let g = OneVideo { gone: frame(40) };
        // the search, the grid, the tile selected, then (after Done) the picker gone
        let mut act = ScrollStays(FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30), cap_of(40)]));
        let plan = Plan { select_all: true, query: Some("October 15 2018".into()), ..Plan::default() };
        let mut lines: Vec<String> = vec![];
        let s = drive(&mut act, &g, &plan, &mut |l| lines.push(l.into())).unwrap();
        assert_eq!(s.selected, 1, "{s:?}");
        assert!(s.done_pressed, "{s:?}");
        let video = Element { label: String::new(), kind: String::new(), x1: 10, y1: 40, x2: 50, y2: 80 };
        assert_eq!(act.0.clicks, vec![(100, 4), crate::picker::click_point(&video), (189, 10)], "search, the one tile, Done");
        assert!(act.0.scrolls > 0, "it looked for the end of the grid");
        assert!(lines.iter().any(|l| l.contains("one item")), "{lines:?}");
    }

    /// A FakeActor whose scrolls change nothing (a grid that fits the window), and which records
    /// Shift+clicks: any is a failure here.
    struct ScrollStays(FakeActor);
    impl Act for ScrollStays {
        fn activate(&mut self) -> Result<(), ScreenError> {
            self.0.activate()
        }
        fn click_png(&mut self, cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            self.0.click_png(cap, px, py)
        }
        fn shift_click_png(&mut self, _cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            Err(ScreenError::Other(format!("a Shift+click at ({px}, {py}) would toggle the only selected tile off")))
        }
        fn type_text(&mut self, text: &str) -> Result<(), ScreenError> {
            self.0.type_text(text)
        }
        fn key_return(&mut self) -> Result<(), ScreenError> {
            self.0.key_return()
        }
        fn key_escape(&mut self) -> Result<(), ScreenError> {
            self.0.key_escape()
        }
        fn scroll(&mut self, n: i32) -> Result<(), ScreenError> {
            self.0.scrolls += n; // the screen stays where it is
            Ok(())
        }
        fn sleep(&mut self, _ms: u64) {}
        fn capture(&mut self) -> Result<Capture, ScreenError> {
            self.0.capture()
        }
    }

    #[test]
    fn select_all_without_a_search_field_refuses_instead_of_taking_the_library() {
        // the live failure: the picker had not loaded its search box yet, the agent went on with
        // the whole library grid and began a range selection from its first tile
        let g = FakeGrounder { inventory: vec![tile("0:30", 10)], finds: vec![] };
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30), cap_of(40), cap_of(50)]);
        let plan = Plan { select_all: true, query: Some("videos October 14 2022".into()), ..Plan::default() };
        let mut lines: Vec<String> = vec![];
        let err = drive(&mut act, &g, &plan, &mut |l| lines.push(l.into())).unwrap_err();
        assert!(err.to_string().contains("refusing to select every item"), "{err}");
        assert!(act.clicks.is_empty() && act.shift_clicks.is_empty(), "nothing was clicked: {:?}", act.clicks);
        assert_eq!(lines.iter().filter(|l| l.contains("waiting for the picker")).count(), SEARCH_TRIES - 1, "{lines:?}");
    }

    #[test]
    fn a_search_that_matches_nothing_stops_the_wait_at_once() {
        // the live failure: "Dubai Police robot videos" had no results, the agent found no tiles,
        // and the pick waited out its whole timeout for a selection that could never come
        struct NoResults;
        impl Ground for NoResults {
            fn inventory(&self, _png: &[u8], _task: &str) -> Result<Vec<Element>, ScreenError> {
                Ok(vec![])
            }
            fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
                Ok((target == crate::picker::SEARCH_FIELD).then(|| Element { label: "Search".into(), kind: String::new(), x1: 40, y1: 0, x2: 160, y2: 8 }))
            }
        }
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30), cap_of(40), cap_of(50)]);
        let plan = Plan { select_all: true, query: Some("Dubai Police robot".into()), ..Plan::default() };
        let s = drive(&mut act, &NoResults, &plan, &mut |_| {}).unwrap();
        assert!(!s.done_pressed && s.selected == 0, "{s:?}");
        assert!(act.shift_clicks.is_empty());
        assert!(s.stop_waiting(Duration::ZERO), "nothing will be picked: stop now");
        // Done pressed: Google gets a moment to report the selection, not forever
        let done = Summary { done_pressed: true, selected: 3, ..Summary::default() };
        assert!(!done.stop_waiting(Duration::from_secs(5)));
        assert!(done.stop_waiting(DONE_GRACE));
    }

    /// A picker with (or without) an Albums view: scripted by ask, recording every ask.
    struct AlbumPicker {
        tab: Option<Element>,
        search: bool,
        /// What each ALBUM_TILES ask sees, in order (the last repeats).
        albums: Vec<Vec<Element>>,
        heading: &'static str,
        asks: std::cell::RefCell<Vec<String>>,
    }

    fn album_card(label: &str, x1: u32, y1: u32) -> Element {
        Element { label: label.into(), kind: "album".into(), x1, y1, x2: x1 + 50, y2: y1 + 50 }
    }

    const TAB: (u32, u32) = (65, 10);
    const FIELD: (u32, u32) = (100, 4);

    impl AlbumPicker {
        fn new(tab: bool, albums: Vec<Vec<Element>>) -> Self {
            let tab = tab.then(|| Element { label: "Albums".into(), kind: "tab".into(), x1: 50, y1: 5, x2: 80, y2: 15 });
            Self { tab, search: true, albums, heading: "Robot showcase", asks: Default::default() }
        }
        fn asked(&self, task: &str) -> usize {
            self.asks.borrow().iter().filter(|a| *a == task).count()
        }
        fn asked_for_tiles(&self) -> bool {
            self.asks.borrow().iter().any(|a| a.starts_with(crate::picker::VIDEO_TILES) || a.starts_with(crate::picker::ALL_TILES))
        }
    }

    impl Ground for AlbumPicker {
        fn inventory(&self, _png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
            self.asks.borrow_mut().push(task.to_string());
            if task == crate::picker::ALBUM_TILES {
                let n = self.asked(task);
                return Ok(self.albums.get(n - 1).or(self.albums.last()).cloned().unwrap_or_default());
            }
            if task == crate::picker::ALBUM_HEADING {
                return Ok(vec![Element { label: self.heading.into(), kind: "heading".into(), x1: 5, y1: 20, x2: 120, y2: 32 }]);
            }
            if task == crate::picker::SELECTED_COUNTER {
                // the range's count; once Done took, the picker has left its grid
                if self.asked(task) > 1 {
                    return Ok(vec![]);
                }
                return Ok(vec![Element { label: "12 selected".into(), kind: String::new(), x1: 5, y1: 2, x2: 60, y2: 12 }]);
            }
            let tile = |l: &str, x1: u32, y1: u32| Element { label: l.into(), kind: String::new(), x1, y1, x2: x1 + 40, y2: y1 + 40 };
            Ok(vec![tile("0:05", 10, 40), tile("photo", 60, 40), tile("0:12", 110, 100)])
        }
        fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            self.asks.borrow_mut().push(target.to_string());
            Ok(match target {
                t if t == crate::picker::ALBUMS_TAB => self.tab.clone(),
                t if t == crate::picker::SEARCH_FIELD && self.search => {
                    Some(Element { label: "Search".into(), kind: "field".into(), x1: 40, y1: 0, x2: 160, y2: 8 })
                }
                t if t.contains("Done") => Some(Element { label: "Done".into(), kind: String::new(), x1: 180, y1: 0, x2: 199, y2: 20 }),
                _ => None,
            })
        }
    }

    fn album_actor() -> FakeActor {
        let mut a = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30), cap_of(40), cap_of(50), cap_of(60)]);
        a.after_shift = Some(cap_of(90));
        a
    }

    fn album_plan(title: &str, videos_only: bool) -> Plan {
        plan_for(Request { album: Some(title.into()), videos_only, query: "ignored".into(), max_items: 3, select_all: false })
    }

    #[test]
    fn an_album_is_opened_from_the_albums_view_and_selected_whole() {
        let wanted = album_card("Robot showcase", 60, 60);
        let g = AlbumPicker::new(true, vec![vec![album_card("Cars", 0, 60), wanted.clone(), album_card("Robot showcase 2", 120, 60)]]);
        let mut act = album_actor();
        let plan = album_plan("Robot showcase", false);
        assert!(plan.select_all && plan.query.is_none() && !plan.videos_only, "{plan:?}");
        let mut lines: Vec<String> = vec![];
        let s = drive(&mut act, &g, &plan, &mut |l| lines.push(l.into())).unwrap();
        assert!(s.album_opened && s.done_pressed && !s.searched, "{s:?}");
        assert_eq!(s.selected, 12, "the header count");
        // the tab, the album, the first tile; the last tile by Shift+click; then Done
        assert_eq!(act.clicks[0], TAB);
        assert_eq!(act.clicks[1], wanted.center(), "{:?}", act.clicks);
        let tile = |x1: u32, y1: u32| Element { label: String::new(), kind: String::new(), x1, y1, x2: x1 + 40, y2: y1 + 40 };
        assert_eq!(act.clicks[2], crate::picker::click_point(&tile(10, 40)));
        assert_eq!(act.shift_clicks, vec![crate::picker::click_point(&tile(110, 100))]);
        assert_eq!(act.clicks.len(), 4, "tab, album, first tile, Done: {:?}", act.clicks);
        assert!(act.typed.is_empty(), "nothing typed");
        assert!(lines.iter().any(|l| l.contains("opened the album `Robot showcase`")), "{lines:?}");
        // the whole album: photos too
        assert!(g.asks.borrow().iter().any(|a| a == crate::picker::ALL_TILES));
    }

    #[test]
    fn without_an_albums_view_the_title_is_searched_and_the_album_opened() {
        // no suggestion under the field; the album card shows among the results
        let wanted = album_card("Robot showcase", 60, 60);
        let g = AlbumPicker::new(false, vec![vec![], vec![wanted.clone()]]);
        let mut act = album_actor();
        let s = drive(&mut act, &g, &album_plan("Robot showcase", true), &mut |_| {}).unwrap();
        assert!(s.album_opened && s.searched && s.done_pressed, "{s:?}");
        assert_eq!(act.typed, vec!["Robot showcase".to_string()], "the title as is: no ` videos`");
        assert_eq!(act.returns, 1);
        assert_eq!(act.clicks[0], FIELD);
        assert_eq!(act.clicks[1], wanted.center());
        // videos only: the range is grounded on video tiles
        assert!(g.asks.borrow().iter().any(|a| a == crate::picker::VIDEO_TILES));
        // a suggestion under the field is taken without pressing Return
        let g = AlbumPicker::new(false, vec![vec![wanted.clone()]]);
        let mut act = album_actor();
        let s = drive(&mut act, &g, &album_plan("robot showcase", false), &mut |_| {}).unwrap();
        assert!(s.album_opened && act.returns == 0, "{s:?}");
    }

    #[test]
    fn an_album_that_is_not_there_selects_nothing() {
        for tab in [true, false] {
            let g = AlbumPicker::new(tab, vec![vec![album_card("Cars", 0, 60), album_card("Robot showcase 2", 60, 60)]]);
            let mut act = album_actor();
            let err = drive(&mut act, &g, &album_plan("Robot showcase", false), &mut |_| {}).unwrap_err().to_string();
            assert!(err.contains("no album titled `Robot showcase`") && err.contains("nothing was selected"), "{err}");
            // only the way in was clicked: no album, no tile, no Done
            assert_eq!(act.clicks, vec![if tab { TAB } else { FIELD }], "tab {tab}");
            assert!(act.shift_clicks.is_empty());
            assert!(!g.asked_for_tiles(), "never even looked at tiles: {:?}", g.asks.borrow());
        }
        // the albums view is paged through, a bounded number of times
        let g = AlbumPicker::new(true, vec![vec![]]);
        let mut act = FakeActor::new((0..12).map(|i| cap_of(10 + i * 10)).collect());
        assert!(drive(&mut act, &g, &album_plan("Robot showcase", false), &mut |_| {}).is_err());
        assert_eq!(act.scrolls, SCROLL_LINES * (ALBUM_PAGES as i32 - 1));
        assert_eq!(act.clicks.len(), 1);
    }

    #[test]
    fn two_albums_with_the_title_are_not_chosen_between() {
        let g = AlbumPicker::new(true, vec![vec![album_card("Robots", 0, 60), album_card("Robots", 120, 60)]]);
        let mut act = album_actor();
        let err = drive(&mut act, &g, &album_plan("Robots", false), &mut |_| {}).unwrap_err().to_string();
        assert!(err.contains("2 albums titled `Robots`"), "{err}");
        assert_eq!(act.clicks, vec![TAB]);
        assert!(!g.asked_for_tiles());
    }

    #[test]
    fn a_click_that_opens_something_else_selects_nothing() {
        let mut g = AlbumPicker::new(true, vec![vec![album_card("Robot showcase", 60, 60)]]);
        g.heading = "Recent photos";
        let mut act = album_actor();
        let err = drive(&mut act, &g, &album_plan("Robot showcase", false), &mut |_| {}).unwrap_err().to_string();
        assert!(err.contains("not headed `Robot showcase`"), "{err}");
        assert_eq!(act.clicks.len(), 2, "the tab and the album only: {:?}", act.clicks);
        assert!(act.shift_clicks.is_empty() && !g.asked_for_tiles());
        // and a click that changed nothing is a stall, not a selection
        let g = AlbumPicker::new(true, vec![vec![album_card("Robot showcase", 60, 60)]]);
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(20)]);
        let err = drive(&mut act, &g, &album_plan("Robot showcase", false), &mut |_| {}).unwrap_err().to_string();
        assert!(err.contains("changed nothing"), "{err}");
        assert!(!g.asked_for_tiles());
    }

    #[test]
    fn a_picker_that_never_loads_or_a_tile_sized_tab_is_handled() {
        // neither an Albums view nor a search field: an error after a few looks, no clicks
        let mut g = AlbumPicker::new(false, vec![vec![]]);
        g.search = false;
        let mut act = album_actor();
        let mut lines: Vec<String> = vec![];
        let err = drive(&mut act, &g, &album_plan("Robot showcase", false), &mut |l| lines.push(l.into())).unwrap_err().to_string();
        assert!(err.contains("neither an Albums view nor its search field"), "{err}");
        assert!(act.clicks.is_empty() && act.typed.is_empty());
        assert_eq!(lines.iter().filter(|l| l.contains("waiting for the picker")).count(), SEARCH_TRIES - 1);
        // a misgrounded "Albums tab" the size of a thumbnail is never clicked: the search is used
        let mut g = AlbumPicker::new(false, vec![vec![album_card("Robot showcase", 60, 60)]]);
        g.tab = Some(Element { label: "album".into(), kind: String::new(), x1: 0, y1: 0, x2: 120, y2: 120 });
        let mut act = album_actor();
        let s = drive(&mut act, &g, &album_plan("Robot showcase", false), &mut |_| {}).unwrap();
        assert!(s.searched && s.album_opened);
        assert_eq!(act.clicks[0], FIELD);
        // a title with nothing to match is refused before anything
        let g = AlbumPicker::new(true, vec![vec![]]);
        let mut act = album_actor();
        assert!(drive(&mut act, &g, &album_plan(" !! ", false), &mut |_| {}).unwrap_err().to_string().contains("no letters or digits"));
        assert!(act.clicks.is_empty() && g.asks.borrow().is_empty());
    }

    #[test]
    fn select_all_never_takes_the_whole_library() {
        // photos allowed and no query: there is nothing to search, so no select-all
        let g = FakeGrounder { inventory: vec![tile("0:30", 10)], finds: vec![] };
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30)]);
        let plan = Plan { select_all: true, videos_only: false, query: None, ..Plan::default() };
        let err = drive(&mut act, &g, &plan, &mut |_| {}).unwrap_err().to_string();
        assert!(err.contains("refusing to select every item of the whole library"), "{err}");
        assert!(act.clicks.is_empty() && act.shift_clicks.is_empty());
    }

    #[test]
    fn requests_become_plans() {
        let p = plan_for(Request { query: "beach".into(), max_items: 4, select_all: true, album: None, videos_only: true });
        assert_eq!((p.query.as_deref(), p.max_items, p.select_all, p.videos_only, p.album), (Some("beach"), 4, true, true, None));
        let p = plan_for(Request { query: String::new(), max_items: 0, select_all: false, album: Some("Robots".into()), videos_only: false });
        assert_eq!((p.query, p.select_all, p.videos_only, p.album.as_deref()), (None, true, false, Some("Robots")));
        assert!(p.max_actions >= Plan::default().max_actions);
    }

    /// One video tile, a Done button that is always found, and a "1 selected" header that stays
    /// up for the first `stuck` checks after Done (the press did not take), then is gone.
    struct StickyDone {
        stuck: usize,
        counter_reads: std::cell::Cell<usize>,
    }
    impl Ground for StickyDone {
        fn inventory(&self, _png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
            if task == crate::picker::SELECTED_COUNTER {
                self.counter_reads.set(self.counter_reads.get() + 1);
                return Ok(if self.counter_reads.get() <= self.stuck {
                    vec![Element { label: "1 selected".into(), kind: String::new(), x1: 5, y1: 2, x2: 60, y2: 12 }]
                } else {
                    vec![]
                });
            }
            Ok(vec![tile("0:30", 10)])
        }
        fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            Ok(target.contains("Done").then(|| Element { label: "Done".into(), kind: "button".into(), x1: 180, y1: 0, x2: 199, y2: 20 }))
        }
    }

    const DONE_AT: (u32, u32) = (189, 10);

    #[test]
    fn a_done_press_that_did_not_take_is_pressed_again() {
        // the live failure: "pressed Done", yet Google never got a selection — the picker still
        // showed its grid, its counter and its Done button
        let g = StickyDone { stuck: 1, counter_reads: Default::default() };
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30), cap_of(40), cap_of(50)]);
        let mut lines: Vec<String> = vec![];
        let s = drive(&mut act, &g, &Plan { max_items: 1, ..Plan::default() }, &mut |l| lines.push(l.into())).unwrap();
        assert!(s.done_pressed && s.selected == 1, "{s:?}");
        assert_eq!(act.clicks[1..], [DONE_AT, DONE_AT], "pressed twice: {:?}", act.clicks);
        assert_eq!(g.counter_reads.get(), 2);
        assert!(lines.iter().any(|l| l.contains("did not take Done; pressing it again")), "{lines:?}");
    }

    #[test]
    fn a_picker_that_never_takes_done_is_an_error_not_a_loop() {
        let g = StickyDone { stuck: usize::MAX, counter_reads: Default::default() };
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30), cap_of(40), cap_of(50)]);
        let err = drive(&mut act, &g, &Plan { max_items: 1, ..Plan::default() }, &mut |_| {}).unwrap_err().to_string();
        assert!(err.contains("did not accept Done"), "{err}");
        assert_eq!(act.clicks.len() as u32, 1 + 1 + DONE_RETRIES, "the tile, then Done {} times: {:?}", 1 + DONE_RETRIES, act.clicks);
        // and the retries come out of the action budget
        let g = StickyDone { stuck: usize::MAX, counter_reads: Default::default() };
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30), cap_of(40), cap_of(50)]);
        let tight = Plan { max_items: 1, max_actions: 3, ..Plan::default() };
        let err = drive(&mut act, &g, &tight, &mut |_| {}).unwrap_err().to_string();
        assert!(err.contains("budget"), "{err}");
        assert!(act.clicks.len() <= 3, "{:?}", act.clicks);
    }

    #[test]
    fn done_is_taken_when_the_window_closes_or_the_button_goes() {
        // the autoclose page closing the window right after Done
        let g = StickyDone { stuck: usize::MAX, counter_reads: Default::default() };
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30), cap_of(40)]);
        act.fail_capture_at = Some(2); // the re-capture after the Done click
        let s = drive(&mut act, &g, &Plan { max_items: 1, ..Plan::default() }, &mut |_| {}).unwrap();
        assert!(s.done_pressed);
        assert_eq!(act.clicks[1..], [DONE_AT]);
        assert_eq!(g.counter_reads.get(), 0, "nothing left to read");
        // the Done button gone (the finished page): taken, without reading a counter
        let g = FakeGrounder {
            inventory: vec![tile("0:30", 10)],
            finds: vec![(
                crate::picker::CONFIRM_LABELS[0].to_string(),
                Some(Element { label: "Done".into(), kind: "button".into(), x1: 180, y1: 0, x2: 199, y2: 20 }),
            )],
        };
        struct GoneAfterPress(FakeGrounder, std::cell::Cell<u32>);
        impl Ground for GoneAfterPress {
            fn inventory(&self, png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
                assert_ne!(task, crate::picker::SELECTED_COUNTER, "no counter read once Done is gone");
                self.0.inventory(png, task)
            }
            fn find(&self, png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
                if target.contains("Done") {
                    self.1.set(self.1.get() + 1);
                    if self.1.get() > 1 {
                        return Ok(None);
                    }
                }
                self.0.find(png, target)
            }
        }
        let g = GoneAfterPress(g, Default::default());
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30), cap_of(40)]);
        let s = drive(&mut act, &g, &Plan { max_items: 1, ..Plan::default() }, &mut |_| {}).unwrap();
        assert!(s.done_pressed);
        assert_eq!(act.clicks[1..], [DONE_AT]);
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
