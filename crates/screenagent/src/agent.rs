//! The loop: see (capture + ground), decide, act, verify — with hard budgets on actions
//! and time so the agent can never spin forever (§0). It drives one application's front
//! window, which had better be the Google Photos picker the caller just opened.
//!
//! One picking session takes many searches ([`Plan::queries`]): the picker keeps its selection
//! while the search changes, so the agent clears the search box, types each query in turn and
//! selects its results — by their date groups for a date search, else by the range gesture —
//! and the header's "N selected" only grows. Done is pressed once, at the end. A search with no
//! results is skipped; when the session's item cap or its budget would be passed, the queries
//! not reached are left for a next session ([`Summary::left`]). Whatever happens, the agent then
//! closes the picker tab it drove ([`crate::tabs::close_front_picker`]).

use std::time::{Duration, Instant};

use crate::ScreenError;
use crate::act::Act;
use crate::ground::{Element, Ground};
use crate::picker;
use crate::verify;

/// Google's cap on one picking session's selection (`maxItemCount` at most).
pub const MAX_PICK: u32 = 2000;
/// Searches a caller should give one session (the budget grows with each; more are left).
pub const MAX_SEARCHES: usize = 20;
/// Actions and time each search adds to a many-search session's budget.
const ACTIONS_PER_SEARCH: u32 = 40;
const TIME_PER_SEARCH: Duration = Duration::from_secs(180);
/// The longest a many-search session may run (a pick waits at most an hour for it).
pub const MAX_SESSION_TIME: Duration = Duration::from_secs(3300);
/// Kept back from a many-search session's searches for Done.
const DONE_RESERVE_ACTIONS: u32 = 6;
const DONE_RESERVE_TIME: Duration = Duration::from_secs(60);
/// A search is not started with less than this left.
const SEARCH_MIN_ACTIONS: u32 = 12;
const SEARCH_MIN_TIME: Duration = Duration::from_secs(90);
/// The longest query typed (characters).
const MAX_QUERY_CHARS: usize = 200;

/// What the agent should accomplish in the picker.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Searches typed into the picker's own search field, in order, in this one picking session
    /// (none = take the grid as-is). The selection accumulates across them.
    pub queries: Vec<String>,
    /// How many tiles to select one by one — or, with `select_all`, the session's item cap.
    pub max_items: u32,
    /// Only click tiles that look like videos (duration badges).
    pub videos_only: bool,
    /// Hard cap on state-changing actions; the loop refuses to exceed it.
    pub max_actions: u32,
    /// Hard wall-clock budget.
    pub timeout: Duration,
    /// Select every result instead of `max_items` tiles: the date groups of a date search, else
    /// click the first tile, wheel to the end of the grid and Shift+click the last one (the
    /// picker's own range selection).
    pub select_all: bool,
    /// Pick every item of the user's album with this title instead of searching: open the
    /// picker's Albums view (or search the title, Google's documented way to an album), open
    /// the album whose label matches, select all of it, press Done. No matching album → an
    /// error, with nothing selected. `queries` is ignored; `select_all` is implied.
    pub album: Option<String>,
}

impl Default for Plan {
    fn default() -> Self {
        Self { queries: Vec::new(), max_items: 6, videos_only: true, max_actions: 40, timeout: Duration::from_secs(300), select_all: false, album: None }
    }
}

impl Plan {
    /// `(query, the text typed for it)` per search ([`picker::search_text`]); a videos-only
    /// plan with no query still searches "videos".
    fn searches(&self) -> Vec<(String, String)> {
        if self.queries.is_empty() {
            return picker::search_text(None, self.videos_only).map(|t| vec![(t.clone(), t)]).unwrap_or_default();
        }
        self.queries.iter().filter_map(|q| picker::search_text(Some(q), self.videos_only).map(|t| (q.clone(), t))).collect()
    }
}

/// What a caller asks of [`spawn`].
#[derive(Debug, Clone, Default)]
pub struct Request {
    /// What to search for, in order, in one picking session.
    pub queries: Vec<String>,
    pub max_items: u32,
    pub select_all: bool,
    /// Pick the whole album with this title instead (see [`Plan::album`]); `queries`,
    /// `max_items` and `select_all` are then unused.
    pub album: Option<String>,
    /// Only video tiles (searches ask Google for videos; in an album, the range is grounded
    /// on video tiles — the caller drops any photo in between when it downloads).
    pub videos_only: bool,
}

/// How one search of a session went.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Outcome {
    /// Its results were selected: the count grew.
    Selected,
    /// It found nothing to select (skipped, not an error).
    NoResults,
    /// The count did not grow: everything it found was selected already, or the gesture missed.
    NothingNew,
    /// The count reached the session's cap: its results may be cut short.
    AtCap,
    /// The count fell: its gesture cleared some of what earlier searches selected (it overlapped
    /// one of them). It, and the searches before it, are worth another try.
    Cleared,
    /// Its results took the count past the session's cap — the picker selects past it but then
    /// refuses Done — so its days were unselected again: for a next session, which has room.
    Deferred,
    /// Its results alone are more than a session holds (unselected again): no session fits it;
    /// split it (by day, or a narrower search) or raise the cap.
    TooBig,
    /// It could not be searched or selected (the note says why).
    Failed,
}

/// One search of a session.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Searched {
    pub query: String,
    pub outcome: Outcome,
    /// The header's "N selected" after it — the whole session's count so far (select-all).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub selected: Option<u32>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub note: String,
}

/// What the agent did.
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct Summary {
    pub actions: u32,
    /// Selected in the session: the header's count when selecting all, else tiles clicked.
    pub selected: u32,
    pub searched: bool,
    /// The album of [`Plan::album`] was found and opened.
    pub album_opened: bool,
    pub done_pressed: bool,
    pub note: String,
    /// Each search typed, in order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub searches: Vec<Searched>,
    /// Queries this session did not get to (its item cap, or its budget): for a next session.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub left: Vec<String>,
    /// The agent closed the picker tab when it finished.
    pub tab_closed: bool,
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

    /// The searches worth another try in a fresh session: those that failed, selected nothing
    /// new, reached the cap (their results may be cut short) or cleared earlier selections — and
    /// then also every search before that one that selected something (some of it may be what was
    /// cleared). [`Self::left`] are on top.
    pub fn retry(&self) -> Vec<String> {
        let cleared = self.searches.iter().rposition(|s| s.outcome == Outcome::Cleared);
        self.searches
            .iter()
            .enumerate()
            .filter(|(i, s)| {
                matches!(s.outcome, Outcome::Failed | Outcome::NothingNew | Outcome::AtCap | Outcome::Cleared)
                    || (s.outcome == Outcome::Selected && cleared.is_some_and(|c| *i < c))
            })
            .map(|(_, s)| s.query.clone())
            .collect()
    }

    /// The searches more than a whole session holds: never worth another try as they are.
    pub fn too_big(&self) -> Vec<String> {
        self.searches.iter().filter(|s| s.outcome == Outcome::TooBig).map(|s| s.query.clone()).collect()
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
/// grounding model is present, then drives the picker (search, select, Done) and closes its
/// tab. The join handle reports what the agent did; the authoritative outcome stays the pick
/// session.
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

/// Queries as they are typed: control characters made spaces, white space collapsed, at most
/// [`MAX_QUERY_CHARS`] each; no empty ones and no repeats (in any case), in their order.
pub fn clean_queries(queries: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for q in queries {
        let q: String = q.chars().map(|c| if c.is_control() { ' ' } else { c }).collect::<String>().split_whitespace().collect::<Vec<_>>().join(" ");
        let q: String = q.chars().take(MAX_QUERY_CHARS).collect::<String>().trim().to_string();
        if !q.is_empty() && !out.iter().any(|o| o.to_lowercase() == q.to_lowercase()) {
            out.push(q);
        }
    }
    out
}

/// The plan for a caller's request: an album pick selects the whole album (and has a few more
/// actions to get there); otherwise the queries are searched, one after another, with a budget
/// that grows with each.
pub fn plan_for(req: Request) -> Plan {
    match req.album {
        Some(album) => Plan {
            queries: Vec::new(),
            album: Some(album),
            select_all: true,
            videos_only: req.videos_only,
            max_actions: 60,
            // the session's item cap (none given: Google's most)
            max_items: if req.max_items == 0 { MAX_PICK } else { req.max_items.min(MAX_PICK) },
            ..Plan::default()
        },
        None => {
            let queries = clean_queries(req.queries);
            let d = Plan::default();
            let n = u32::try_from(queries.len()).unwrap_or(u32::MAX);
            let (max_actions, timeout) = if n <= 1 {
                (d.max_actions, d.timeout)
            } else {
                let more = TIME_PER_SEARCH.saturating_mul(n.saturating_sub(1));
                (ACTIONS_PER_SEARCH.saturating_mul(n).saturating_add(DONE_RESERVE_ACTIONS), d.timeout.saturating_add(more).min(MAX_SESSION_TIME))
            };
            Plan { queries, max_items: req.max_items, select_all: req.select_all, videos_only: req.videos_only, max_actions, timeout, album: None }
        }
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
) -> Result<Vec<Element>, ScreenError> {
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

/// Run the picker plan against one app, then close the picker tab it drove (after Done, or on
/// giving up). `progress` receives one line per step, the gphotos convention
/// (`&mut dyn FnMut(&str)`).
pub fn drive<A: Act, G: Ground>(actor: &mut A, grounder: &G, plan: &Plan, progress: &mut dyn FnMut(&str)) -> Result<Summary, ScreenError> {
    // the picker tab it drives, named by its ids now while it is in front (Chrome), so that the
    // close at the end is that tab's, whatever tab is in front by then
    let driven = actor.front_tab().ok().flatten().filter(|t| crate::tabs::is_picker_url(&t.url));
    let result = drive_inner(actor, grounder, plan, progress);
    if let Err(e) = &result {
        // every exit is spoken: a silent agent thread is undebuggable from outside
        progress(&format!("screenagent: giving up: {e}"));
    }
    let closed = crate::tabs::close_driven_picker(actor, grounder, driven.as_ref(), progress).unwrap_or_else(|e| {
        progress(&format!("screenagent: could not close the picker tab: {e}"));
        false
    });
    result.map(|mut s| {
        s.tab_closed = closed;
        s
    })
}

/// Looks for the picker's search field before going on without it, and the wait between them.
const SEARCH_TRIES: usize = 4;
const SEARCH_RETRY_MS: u64 = 1500;

fn drive_inner<A: Act, G: Ground>(actor: &mut A, grounder: &G, plan: &Plan, progress: &mut dyn FnMut(&str)) -> Result<Summary, ScreenError> {
    let started = Instant::now();
    let mut summary = Summary::default();
    let deadline = || started.elapsed() >= plan.timeout;

    actor.activate()?;
    actor.sleep(600);
    let mut cap = actor.capture()?;

    // 1–2. the album, opened and selected whole (or an error with nothing selected); or the
    // searches — for videos always, so Google's own filter leaves only videos in the grid
    if let Some(title) = plan.album.as_deref() {
        open_album(actor, grounder, plan, &mut summary, &mut cap, title, started, progress)?;
        if let Some(found) = take_all(actor, grounder, plan, &mut summary, &mut cap, started, false, progress)? {
            return Err(ScreenError::Other(format!("the album `{title}`: {}", too_big_note(found, plan.max_items.clamp(1, MAX_PICK)))));
        }
    } else {
        let searches = plan.searches();
        if searches.len() > 1 {
            search_many(actor, grounder, plan, &mut summary, &mut cap, started, &searches, progress)?;
        } else if search_one(actor, grounder, plan, &mut summary, &mut cap, started, searches.first(), progress)? == Tiles::WindowClosed {
            return Ok(summary);
        }
    }

    // the picker selects past the session's item cap (date groups, ranges) but then refuses
    // Done (live: "143 selected", "Select up to 40 items", Done greyed out): never press it
    let cap_items = plan.max_items.clamp(1, MAX_PICK);
    if plan.select_all && !summary.done_pressed && summary.selected > cap_items {
        return Err(ScreenError::Stalled(format!(
            "{} are selected, more than the session's {cap_items}-item maximum, and the picker refuses Done past it",
            summary.selected
        )));
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

/// The picker's search field, looked for a few times while the picker loads.
fn find_search_field<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    cap: &mut crate::capture::Capture,
    progress: &mut dyn FnMut(&str),
) -> Result<Option<Element>, ScreenError> {
    let mut field = None;
    for attempt in 0..SEARCH_TRIES {
        field = grounder.find(&cap.png, picker::SEARCH_FIELD)?;
        if field.is_some() || attempt + 1 == SEARCH_TRIES {
            break;
        }
        progress("screenagent: no search field yet; waiting for the picker");
        actor.sleep(SEARCH_RETRY_MS);
        *cap = actor.capture()?;
    }
    Ok(field)
}

/// Type `text` into the search `field` and run the search; `clear` empties the field first (it
/// holds the previous search, about that many characters).
fn type_search<A: Act>(
    actor: &mut A,
    cap: &mut crate::capture::Capture,
    field: &Element,
    text: &str,
    clear: Option<usize>,
    progress: &mut dyn FnMut(&str),
) -> Result<(), ScreenError> {
    let (px, py) = field.center();
    progress(&format!("screenagent: typing `{text}` into the picker search"));
    actor.click_png(cap, px, py)?;
    actor.sleep(400);
    if let Some(n) = clear {
        actor.clear_text(n.saturating_add(8))?;
        actor.sleep(200);
    }
    actor.type_text(text)?;
    actor.key_return()?;
    actor.sleep(SEARCH_SETTLE_MS);
    *cap = actor.capture()?;
    Ok(())
}

/// How selecting tile by tile ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Tiles {
    Done,
    /// The window went away mid-selection (the autoclose page closing it): taken as Done.
    WindowClosed,
}

/// One search (or none), then its selection: every result, or up to `max_items` tiles.
#[allow(clippy::too_many_arguments)]
fn search_one<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    search: Option<&(String, String)>,
    progress: &mut dyn FnMut(&str),
) -> Result<Tiles, ScreenError> {
    let mut dates = false;
    let mut over = None;
    if let Some((_, q)) = search {
        if spend(summary, plan)? || started.elapsed() >= plan.timeout {
            return Err(budget(summary));
        }
        // the picker may still be loading: look again a few times before going on without it
        match find_search_field(actor, grounder, cap, progress)? {
            Some(field) => {
                type_search(actor, cap, &field, q, None, progress)?;
                summary.searched = true;
                dates = picker::is_date_search(q);
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
    let tiles = if plan.select_all {
        // only ever inside a search result: never the whole library
        if !summary.searched {
            return Err(ScreenError::Other(
                "nothing was searched and no album was opened; refusing to select every item of the whole library (selecting all needs a search or an album)"
                    .into(),
            ));
        }
        over = take_all(actor, grounder, plan, summary, cap, started, dates, progress)?;
        Tiles::Done
    } else {
        pick_tiles(actor, grounder, plan, summary, cap, started, progress)?
    };
    if let Some((query, _)) = search.filter(|_| summary.searched) {
        let cap_items = plan.max_items.clamp(1, MAX_PICK);
        let (outcome, note) = match (over, summary.selected, plan.select_all) {
            (Some(found), _, _) => (Outcome::TooBig, too_big_note(found, cap_items)),
            (None, 0, _) => (Outcome::NoResults, String::new()),
            (None, n, true) if n >= cap_items => (Outcome::AtCap, String::new()),
            _ => (Outcome::Selected, String::new()),
        };
        if !note.is_empty() {
            summary.note = format!("`{query}`: {note}");
        }
        summary.searches.push(Searched { query: query.clone(), outcome, selected: plan.select_all.then_some(summary.selected), note });
    }
    Ok(tiles)
}

/// Select up to `max_items` tiles in all, one click each, paging down for more.
fn pick_tiles<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    progress: &mut dyn FnMut(&str),
) -> Result<Tiles, ScreenError> {
    let deadline = || started.elapsed() >= plan.timeout;
    let mut stalled = 0u32;
    let mut tried: Vec<(u32, u32)> = Vec::new();
    while summary.selected < plan.max_items {
        if spend(summary, plan)? || deadline() {
            return Err(budget(summary));
        }
        let tiles = visible_tiles(grounder, cap, plan.videos_only, progress)?;
        let fresh: Vec<&Element> = tiles
            .iter()
            .filter(|t| {
                let (px, py) = picker::click_point(t);
                !tried.iter().any(|&(x, y)| x.abs_diff(px) < SAME_SPOT && y.abs_diff(py) < SAME_SPOT)
            })
            .take(plan.max_items.saturating_sub(summary.selected).max(1) as usize)
            .collect();
        if fresh.is_empty() {
            // nothing new at this scroll position: page down once, or finish
            if spend(summary, plan)? {
                return Err(budget(summary));
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
            *cap = next;
            continue;
        }
        for t in fresh {
            if summary.selected >= plan.max_items || spend(summary, plan)? {
                break;
            }
            let (px, py) = picker::click_point(t);
            tried.push((px, py));
            let before = cap.png.clone();
            if let Err(e) = actor.click_png(cap, px, py) {
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
                    return Ok(Tiles::WindowClosed);
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
            *cap = after;
        }
        if stalled >= MAX_STALLED {
            break;
        }
    }
    Ok(Tiles::Done)
}

/// Errors that end a whole session (no grants, no model, no window): a search that merely
/// failed does not.
fn fatal(e: &ScreenError) -> bool {
    matches!(e, ScreenError::Capture(..) | ScreenError::Input(..) | ScreenError::Model(..) | ScreenError::Unsupported(..))
}

/// The outcome of a search from the header's count before and after it.
fn outcome_of(before: Option<u32>, after: Option<u32>, cap_items: u32) -> (Outcome, String) {
    let b = before.unwrap_or(0);
    match after {
        None if before.is_some() => (Outcome::Failed, "the header showed no count after it".into()),
        None => (Outcome::NothingNew, "nothing was selected".into()),
        Some(a) if a < b => {
            (Outcome::Cleared, format!("the count fell from {b} to {a}: this search overlapped an earlier one, and selecting it again cleared some"))
        }
        Some(a) if a == b => (Outcome::NothingNew, String::new()),
        Some(a) if a >= cap_items => (Outcome::AtCap, format!("the session's {cap_items}-item maximum was reached; its results may be cut short")),
        Some(_) => (Outcome::Selected, String::new()),
    }
}

/// One search of many and its selection.
struct OneSearch {
    /// Something was typed (the search field now holds it).
    typed: bool,
    outcome: Outcome,
    /// The header's count after it (select-all), else unchanged.
    count: Option<u32>,
    note: String,
    /// The picker window went away (tile-by-tile selection takes that as Done).
    closed: bool,
}

/// Search `typed` (after emptying the field when `clear`), then select its results: everything
/// on top of the `before` already selected, or tiles up to `max_items` in all.
#[allow(clippy::too_many_arguments)]
fn one_of_many<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    typed: &str,
    clear: Option<usize>,
    before: Option<u32>,
    progress: &mut dyn FnMut(&str),
) -> Result<OneSearch, ScreenError> {
    let mut one = OneSearch { typed: false, outcome: Outcome::Failed, count: before, note: String::new(), closed: false };
    if spend(summary, plan)? || started.elapsed() >= plan.timeout {
        return Err(budget(summary));
    }
    let Some(field) = find_search_field(actor, grounder, cap, progress)? else {
        one.note = "the picker's search field did not show".into();
        return Ok(one);
    };
    let shown = cap.png.clone();
    type_search(actor, cap, &field, typed, clear, progress)?;
    one.typed = true;
    summary.searched = true;
    // the grid still shows the last search's results: selecting them again would clear them
    if clear.is_some() && !verify::changed(&shown, &cap.png)? {
        one.note = "the grid did not change after the search; not selecting the previous results again".into();
        return Ok(one);
    }
    if !plan.select_all {
        let had = summary.selected;
        one.closed = pick_tiles(actor, grounder, plan, summary, cap, started, progress)? == Tiles::WindowClosed;
        one.outcome = if summary.selected > had { Outcome::Selected } else { Outcome::NoResults };
        return Ok(one);
    }
    let cap_items = plan.max_items.clamp(1, MAX_PICK);
    let after = match by_day_groups(actor, grounder, plan, summary, cap, started, before, picker::is_date_search(typed), progress)? {
        ByDays::Count(n) => n,
        ByDays::OverCap { found, after } => {
            let (b, now) = (before.unwrap_or(0), after.unwrap_or(0));
            if now == b {
                // undone: with nothing selected before it, no session has room for it
                (one.outcome, one.note) = if b == 0 {
                    (Outcome::TooBig, too_big_note(found, cap_items))
                } else {
                    (
                        Outcome::Deferred,
                        format!(
                            "it took the count to {found}, past the session's {cap_items}-item maximum (the picker refuses Done past it); its days were unselected again, for a next session"
                        ),
                    )
                };
                one.count = before;
            } else {
                one.outcome = Outcome::Failed;
                one.note = format!("it took the count to {found}, past the session's {cap_items}-item maximum, and unselecting its days left {now}, not {b}");
                one.count = after;
            }
            return Ok(one);
        }
        ByDays::Range => match before {
            // nothing selected yet: the whole gesture, with its retry from the top
            None => match select_everything(actor, grounder, plan, summary, cap, started, progress)? {
                None => {
                    one.outcome = Outcome::NoResults;
                    return Ok(one);
                }
                Some(n) => Some(n),
            },
            Some(_) => {
                if range_once(actor, grounder, plan, summary, cap, started, progress)? == Ranged::NoTiles {
                    one.outcome = Outcome::NoResults;
                    return Ok(one);
                }
                read_selected(grounder, cap)?
            }
        },
    };
    (one.outcome, one.note) = outcome_of(before, after, cap_items);
    one.count = after.or(before);
    Ok(one)
}

/// Many searches in one session: each typed over the last (the field emptied first), its results
/// selected on top of what is selected already, the header's count read after each — it only
/// grows. A search with no results is skipped; a search that fails is noted and the next one
/// tried; an error that ends the session (no grants, no model) ends it. Before each search the
/// room is checked: the session's item cap (with the biggest search so far as the guess of the
/// next one's size) and the budget, part of which stays back for Done. What does not fit is
/// left for a next session.
#[allow(clippy::too_many_arguments)]
fn search_many<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    searches: &[(String, String)],
    progress: &mut dyn FnMut(&str),
) -> Result<(), ScreenError> {
    let inner =
        Plan { max_actions: plan.max_actions.saturating_sub(DONE_RESERVE_ACTIONS), timeout: plan.timeout.saturating_sub(DONE_RESERVE_TIME), ..plan.clone() };
    let room = if plan.select_all { plan.max_items.clamp(1, MAX_PICK) } else { plan.max_items };
    let mut count: Option<u32> = None;
    let mut biggest = 0u32;
    let mut holds: Option<usize> = None; // characters the search field holds
    let mut deferred: Vec<String> = Vec::new();
    for (i, (query, typed)) in searches.iter().enumerate() {
        let so_far = if plan.select_all { count.unwrap_or(0) } else { summary.selected };
        if i > 0 {
            let actions_left = inner.max_actions.saturating_sub(summary.actions);
            let time_left = inner.timeout.saturating_sub(started.elapsed());
            let why = if so_far >= room {
                Some(format!("the session holds its {room}-item maximum"))
            } else if plan.select_all && so_far.saturating_add(biggest) > room {
                Some(format!("{so_far} are selected and the next search could pass the session's {room}-item maximum"))
            } else if actions_left < SEARCH_MIN_ACTIONS || time_left < SEARCH_MIN_TIME {
                Some("this session's budget is nearly spent".to_string())
            } else {
                None
            };
            if let Some(why) = why {
                summary.left = searches.iter().skip(i).map(|(q, _)| q.clone()).collect();
                progress(&format!("screenagent: {why}; leaving {} search(es) for a next session", summary.left.len()));
                break;
            }
        }
        progress(&format!("screenagent: search {} of {}: `{typed}`", i + 1, searches.len()));
        let result = one_of_many(actor, grounder, &inner, summary, cap, started, typed, holds, count, progress);
        let one = match result {
            Ok(one) => one,
            Err(e) if fatal(&e) => return Err(e),
            Err(e) => {
                // the gesture may have changed the selection part way: read where it stands
                let now = if plan.select_all { read_selected(grounder, cap).ok().flatten().or(count) } else { count };
                OneSearch { typed: true, outcome: Outcome::Failed, count: now, note: e.to_string(), closed: false }
            }
        };
        if one.typed {
            holds = Some(typed.chars().count());
        }
        if plan.select_all {
            if let (Some(a), b) = (one.count, count.unwrap_or(0))
                && one.outcome == Outcome::Selected
            {
                biggest = biggest.max(a.saturating_sub(b));
            }
            count = one.count;
        }
        if one.outcome == Outcome::Deferred {
            deferred.push(query.clone());
        }
        let selected = plan.select_all.then_some(count).flatten();
        progress(&format!("screenagent: `{query}`: {}", outcome_line(one.outcome, selected, &one.note)));
        summary.searches.push(Searched { query: query.clone(), outcome: one.outcome, selected, note: one.note });
        if one.closed {
            summary.left = searches.iter().skip(i + 1).map(|(q, _)| q.clone()).collect();
            break;
        }
    }
    if plan.select_all {
        summary.selected = count.unwrap_or(0);
    }
    if !deferred.is_empty() {
        // first in the next session, which has room for them
        deferred.append(&mut summary.left);
        summary.left = deferred;
    }
    let with = |o: Outcome| summary.searches.iter().filter(|s| s.outcome == o).count();
    let past = with(Outcome::Deferred) + with(Outcome::TooBig);
    summary.note = format!(
        "{} search(es): {} selected from, {} without results, {} with nothing new, {} failed{}{}; {} selected in all",
        summary.searches.len(),
        with(Outcome::Selected) + with(Outcome::AtCap),
        with(Outcome::NoResults),
        with(Outcome::NothingNew),
        with(Outcome::Failed) + with(Outcome::Cleared),
        if past == 0 { String::new() } else { format!(", {past} past the session's maximum (unselected)") },
        if summary.left.is_empty() { String::new() } else { format!(", {} left for a next session", summary.left.len()) },
        summary.selected
    );
    Ok(())
}

fn outcome_line(o: Outcome, selected: Option<u32>, note: &str) -> String {
    let what = match o {
        Outcome::Selected => "selected",
        Outcome::NoResults => "no results, skipped",
        Outcome::NothingNew => "nothing new selected",
        Outcome::AtCap => "selected up to the session's maximum",
        Outcome::Cleared => "cleared some earlier selections",
        Outcome::Deferred => "past the session's maximum, unselected, left for a next session",
        Outcome::TooBig => "more than a session holds, unselected",
        Outcome::Failed => "failed",
    };
    let count = selected.map(|n| format!(" ({n} selected so far)")).unwrap_or_default();
    if note.is_empty() { format!("{what}{count}") } else { format!("{what}{count}: {note}") }
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
    mut btn: Element,
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
    Tab(Element),
    /// The search field: Google's documented way ("search for the album").
    Search(Element),
}

/// The albums named `title` in the current capture (see [`picker::album_candidates`]).
fn albums_named<G: Ground>(grounder: &G, cap: &crate::capture::Capture, title: &str, avoid: Option<&Element>) -> Result<Vec<Element>, ScreenError> {
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

/// Tries of the whole range gesture when the header shows nothing selected after it.
const RANGE_TRIES: u32 = 2;
/// Reads of the header's "N selected" before taking its absence as "nothing selected".
const COUNTER_READS: u32 = 2;
/// The most scrolls back to the top of the grid before a retry.
const MAX_TOP_SCROLLS: u32 = 20;

/// Select every result with the picker's range gesture: click the first tile, wheel to the end
/// of the grid (until two scrolls change nothing — the grid loads lazily), Shift+click the
/// last tile. Two clicks select a whole search result, where tile-by-tile selection misses
/// the tiles the model does not box. How many are selected is what the picker's header says
/// ("N selected", nothing else is trusted): a header with no selection gets the gesture once
/// more from the top, then an error — Done is never pressed with nothing selected. `None`:
/// there was no tile to select.
fn select_everything<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    progress: &mut dyn FnMut(&str),
) -> Result<Option<u32>, ScreenError> {
    for attempt in 0..RANGE_TRIES {
        if attempt > 0 {
            progress("screenagent: the picker shows nothing selected; selecting once more from the top of the grid");
            back_to_top(actor, plan, summary, cap, started)?;
        }
        summary.selected = 0;
        if range_once(actor, grounder, plan, summary, cap, started, progress)? == Ranged::NoTiles {
            return Ok(None);
        }
        if let Some(n) = read_selected(grounder, cap)? {
            summary.selected = n;
            progress(&format!("screenagent: the picker shows {n} selected"));
            return Ok(Some(n));
        }
    }
    summary.selected = 0;
    Err(ScreenError::Stalled(format!("the picker shows nothing selected after selecting all ({RANGE_TRIES} tries); not pressing its Done button")))
}

/// Select everything on screen — a search's results, or an opened album: by the date groups
/// of a date search first, else (or when they select nothing) by the range gesture with its
/// retry. `summary.selected` is the header's count after it. `Ok(Some(found))`: the date groups
/// selected `found`, more than the session's cap, and were unselected again.
#[allow(clippy::too_many_arguments)]
fn take_all<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    dates: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<Option<u32>, ScreenError> {
    match by_day_groups(actor, grounder, plan, summary, cap, started, None, dates, progress)? {
        ByDays::Count(Some(n)) => {
            summary.selected = n;
            return Ok(None);
        }
        ByDays::OverCap { found, after } => {
            summary.selected = after.unwrap_or(0);
            return Ok(Some(found));
        }
        ByDays::Count(None) | ByDays::Range => {}
    }
    if select_everything(actor, grounder, plan, summary, cap, started, progress)?.is_none() {
        summary.note = "no tiles to select".into();
    }
    Ok(None)
}

/// Why a search (or album) that alone passes the session's cap was not picked.
fn too_big_note(found: u32, cap_items: u32) -> String {
    format!(
        "its {found} results are more than the session's {cap_items}-item maximum (the picker refuses Done past it), so they were unselected again; split it (by day, or a narrower search) or raise --max-items"
    )
}

/// Pages of a date search's results looked through for its date groups.
const DAY_PAGES: u32 = 15;

/// What selecting by date groups came to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ByDays {
    /// No date group was clicked, or the groups selected nothing new: use the range gesture
    /// (the grid is back at its top).
    Range,
    /// The header's count after the groups (`None`: it could not be read).
    Count(Option<u32>),
    /// The groups took the count to `found`, past the session's cap — the picker selects past it
    /// but then refuses Done — so the same groups were clicked again to unselect them; `after` is
    /// the count then.
    OverCap { found: u32, after: Option<u32> },
}

/// Select a date search's results by their date groups (see [`select_day_groups`]) — only for a
/// date search (`dates`), and only while the header's count says they took: a count above
/// `before` is taken; the same count (or none at all on a first selection) means the groups did
/// nothing and the range gesture is used instead; a count that fell, or could not be read after
/// an earlier selection, is reported as it is (another gesture could clear more). A count past
/// the session's cap is undone: the same days clicked once more ([`ByDays::OverCap`]).
#[allow(clippy::too_many_arguments)]
fn by_day_groups<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    before: Option<u32>,
    dates: bool,
    progress: &mut dyn FnMut(&str),
) -> Result<ByDays, ScreenError> {
    if !dates {
        return Ok(ByDays::Range);
    }
    let (days, scrolled) = select_day_groups(actor, grounder, plan, summary, cap, started, None, progress)?;
    let clicked = days.len();
    if clicked > 0 {
        let b = before.unwrap_or(0);
        let cap_items = plan.max_items.clamp(1, MAX_PICK);
        match read_selected(grounder, cap)? {
            Some(n) if n > cap_items => {
                progress(&format!(
                    "screenagent: the date groups took the count to {n}, past the session's {cap_items}-item maximum (the picker refuses Done past it); unselecting those {clicked} day(s)"
                ));
                rewind(actor, summary, cap)?;
                let (undone, _) = select_day_groups(actor, grounder, plan, summary, cap, started, Some(&days), progress)?;
                if undone.len() < clicked {
                    progress(&format!("screenagent: unselected {} of the {clicked} day(s)", undone.len()));
                }
                let after = read_selected(grounder, cap)?;
                return Ok(ByDays::OverCap { found: n, after });
            }
            Some(n) if n > b => {
                progress(&format!("screenagent: {clicked} day(s) selected by their date groups; the picker shows {n} selected"));
                return Ok(ByDays::Count(Some(n)));
            }
            Some(n) if n < b => return Ok(ByDays::Count(Some(n))),
            None if before.is_some() => return Ok(ByDays::Count(None)),
            _ => progress("screenagent: the date groups selected nothing new; selecting the range instead"),
        }
    }
    if scrolled {
        back_to_top(actor, plan, summary, cap, started)?;
    }
    Ok(ByDays::Range)
}

/// Click the round check button of each date group of a date search's results, paging down for
/// more. Only a check labelled with a date is clicked, each date once ([`picker::date_key`]: a
/// heading still on screen after a scroll is the same day, and a second click would clear it),
/// and none already filled in. With `undo` (the keys of days this clicked before), exactly
/// those days are clicked once more — a second click unselects a whole day (live) — whatever
/// the model says of their fill, until all are; an undo never stops for the budget (it is
/// bounded by the pages and the days), since stopping half way would leave a selection the
/// picker refuses Done on. `(the keys of the days clicked, scrolled)`; no dated check on the
/// first page is `([], false)` at once.
#[allow(clippy::too_many_arguments)]
fn select_day_groups<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    undo: Option<&[String]>,
    progress: &mut dyn FnMut(&str),
) -> Result<(Vec<String>, bool), ScreenError> {
    let out_of_time = || started.elapsed() >= plan.timeout;
    // an action from the budget; `true`: none left (an undo only counts it)
    let stop = |summary: &mut Summary| -> Result<bool, ScreenError> {
        if undo.is_some() {
            summary.actions = summary.actions.saturating_add(1);
            return Ok(false);
        }
        Ok(spend(summary, plan)? || out_of_time())
    };
    let mut seen: Vec<String> = Vec::new();
    let mut clicked: Vec<String> = Vec::new();
    let (mut scrolled, mut still, mut look) = (false, 0u32, true);
    for page in 0..DAY_PAGES {
        if look {
            let els = grounder.inventory(&cap.png, picker::DAY_CHECKS)?;
            let checks = picker::day_checks(&els, cap.png_w, cap.png_h);
            if page == 0 && checks.is_empty() && undo.is_none() {
                return Ok((Vec::new(), false));
            }
            for c in checks {
                let Some(key) = picker::date_key(&c.label) else { continue };
                if seen.contains(&key) {
                    continue;
                }
                match undo {
                    Some(days) if !days.contains(&key) => continue,
                    None if picker::is_checked(&c) => {
                        seen.push(key);
                        continue;
                    }
                    _ => {}
                }
                seen.push(key.clone());
                if stop(summary)? {
                    return Err(budget(summary));
                }
                let (px, py) = picker::check_point(&c);
                let before = cap.png.clone();
                actor.click_png(cap, px, py)?;
                actor.sleep(CLICK_SETTLE_MS);
                *cap = actor.capture()?;
                if verify::changed(&before, &cap.png)? {
                    progress(&format!("screenagent: {} the day `{}`", if undo.is_some() { "unselected" } else { "selected" }, c.label.trim()));
                    clicked.push(key);
                }
            }
        }
        if undo.is_some_and(|days| days.iter().all(|d| clicked.contains(d))) {
            break; // every day undone
        }
        if stop(summary)? {
            return Err(budget(summary));
        }
        let before = cap.png.clone();
        actor.scroll(SCROLL_LINES)?;
        actor.sleep(900);
        *cap = actor.capture()?;
        look = verify::changed(&before, &cap.png)?;
        if look {
            (scrolled, still) = (true, 0);
        } else {
            still += 1;
            if still >= 2 {
                break; // the end of the results
            }
        }
    }
    Ok((clicked, scrolled))
}

/// The picker header's "N selected", read up to [`COUNTER_READS`] times; `None` when it shows
/// no selection.
fn read_selected<G: Ground>(grounder: &G, cap: &crate::capture::Capture) -> Result<Option<u32>, ScreenError> {
    for _ in 0..COUNTER_READS {
        let els = grounder.inventory(&cap.png, picker::SELECTED_COUNTER)?;
        if let Some(n) = els.iter().find_map(|e| picker::selected_count(&e.label)) {
            return Ok(Some(n));
        }
    }
    Ok(None)
}

/// Wheel back to the top of the grid (until two scrolls change nothing).
fn back_to_top<A: Act>(actor: &mut A, plan: &Plan, summary: &mut Summary, cap: &mut crate::capture::Capture, started: Instant) -> Result<(), ScreenError> {
    let mut still = 0u32;
    for _ in 0..MAX_TOP_SCROLLS {
        if still >= 2 {
            break;
        }
        if spend(summary, plan)? || started.elapsed() >= plan.timeout {
            return Err(budget(summary));
        }
        let before = cap.png.clone();
        actor.scroll(-END_SCROLL_LINES)?;
        actor.sleep(900);
        *cap = actor.capture()?;
        still = if verify::changed(&before, &cap.png)? { 0 } else { still + 1 };
    }
    Ok(())
}

/// [`back_to_top`] for an undo: its scrolls are counted but never stop for the budget.
fn rewind<A: Act>(actor: &mut A, summary: &mut Summary, cap: &mut crate::capture::Capture) -> Result<(), ScreenError> {
    let mut still = 0u32;
    for _ in 0..MAX_TOP_SCROLLS {
        if still >= 2 {
            break;
        }
        summary.actions = summary.actions.saturating_add(1);
        let before = cap.png.clone();
        actor.scroll(-END_SCROLL_LINES)?;
        actor.sleep(900);
        *cap = actor.capture()?;
        still = if verify::changed(&before, &cap.png)? { 0 } else { still + 1 };
    }
    Ok(())
}

/// What one range gesture did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ranged {
    /// There was no tile to select.
    NoTiles,
    /// It clicked (the header says what that selected).
    Clicked,
}

/// One range gesture (see [`select_everything`]).
fn range_once<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    progress: &mut dyn FnMut(&str),
) -> Result<Ranged, ScreenError> {
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
        return Ok(Ranged::NoTiles);
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
        return Ok(Ranged::Clicked);
    };
    // the last tile is the first one (a grid of one): a Shift+click on it would toggle the
    // selection back off — the first click already selected it
    if picker::same_tile(&first, last, scrolled) || (!scrolled && first_page == 1 && tiles.len() == 1) {
        summary.note = "the grid holds one item; the first click selected it".into();
        progress("screenagent: the grid holds one item, already selected");
        return Ok(Ranged::Clicked);
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
        return Ok(Ranged::Clicked);
    }
    progress("screenagent: range click done; reading how many are selected");
    Ok(Ranged::Clicked)
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
        let plan = Plan { queries: vec!["beach".into()], max_items: 1, ..Plan::default() };
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
        let plan = Plan { select_all: true, max_items: MAX_PICK, ..Plan::default() };
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
        let plan = Plan { select_all: true, max_items: MAX_PICK, queries: vec!["October 15 2018".into()], ..Plan::default() };
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

    /// Hands over a two-page grid: every click changes the screen, and the grid moves once per
    /// scroll direction (then it is at the end, or back at the top).
    #[derive(Default)]
    struct TwoPages {
        tag: u8,
        dir: i32,
        clicks: Vec<(u32, u32)>,
        shift_clicks: Vec<(u32, u32)>,
        ups: u32,
    }
    impl Act for TwoPages {
        fn activate(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn click_png(&mut self, _: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            self.clicks.push((px, py));
            self.tag = self.tag.wrapping_add(10);
            Ok(())
        }
        fn shift_click_png(&mut self, _: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            self.shift_clicks.push((px, py));
            self.tag = self.tag.wrapping_add(10);
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
        fn scroll(&mut self, n: i32) -> Result<(), ScreenError> {
            if n < 0 {
                self.ups += 1;
            }
            if n.signum() != self.dir {
                self.dir = n.signum();
                self.tag = self.tag.wrapping_add(7);
            }
            Ok(())
        }
        fn sleep(&mut self, _: u64) {}
        fn capture(&mut self) -> Result<Capture, ScreenError> {
            Ok(cap_of(self.tag))
        }
    }

    /// A search result in two rows; the header reads the scripted labels in turn (`""`: no
    /// counter at all), then nothing (the picker left its grid after Done).
    struct Header(std::cell::RefCell<std::collections::VecDeque<&'static str>>);
    impl Header {
        fn reads(labels: &[&'static str]) -> Self {
            Self(std::cell::RefCell::new(labels.iter().copied().collect()))
        }
    }
    impl Ground for Header {
        fn inventory(&self, _png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
            if task == crate::picker::SELECTED_COUNTER {
                let label = self.0.borrow_mut().pop_front().unwrap_or("");
                return Ok(if label.is_empty() { vec![] } else { vec![Element { label: label.into(), kind: String::new(), x1: 5, y1: 2, x2: 60, y2: 12 }] });
            }
            let t = |l: &str, x1: u32, y1: u32| Element { label: l.into(), kind: String::new(), x1, y1, x2: x1 + 40, y2: y1 + 40 };
            Ok(vec![t("0:05", 10, 30), t("0:30", 60, 30), t("0:12", 110, 90)])
        }
        fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            if target == crate::picker::SEARCH_FIELD {
                return Ok(Some(Element { label: "Search".into(), kind: String::new(), x1: 40, y1: 0, x2: 160, y2: 8 }));
            }
            Ok(target.contains("Done").then(|| Element { label: "Done".into(), kind: String::new(), x1: 180, y1: 0, x2: 199, y2: 20 }))
        }
    }

    fn range_plan() -> Plan {
        Plan { select_all: true, max_items: MAX_PICK, queries: vec!["robots".into()], ..Plan::default() }
    }

    #[test]
    fn a_header_without_a_selection_is_never_confirmed() {
        // the live failure: the header read "Select up to 300 items" (nothing selected), the
        // model answered with a count anyway, and a disabled Done was pressed
        let g = Header::reads(&["12", "Select up to 300 items", "12", "Add from Google Photos"]);
        let mut act = TwoPages::default();
        let mut lines: Vec<String> = vec![];
        let err = drive(&mut act, &g, &range_plan(), &mut |l| lines.push(l.into())).unwrap_err().to_string();
        assert!(err.contains("shows nothing selected"), "{err}");
        assert!(!act.clicks.contains(&DONE_AT), "Done never pressed: {:?}", act.clicks);
        // the gesture was tried twice, the second time from the top
        assert_eq!(act.shift_clicks.len() as u32, RANGE_TRIES, "{:?}", act.shift_clicks);
        assert!(act.ups > 0);
        assert!(lines.iter().any(|l| l.contains("selecting once more")), "{lines:?}");
    }

    #[test]
    fn only_an_n_selected_header_is_a_count() {
        // "1 selected" after a range is trusted as it is (not rounded up to the two clicks)
        let g = Header::reads(&["1 selected"]);
        let mut act = TwoPages::default();
        let s = drive(&mut act, &g, &range_plan(), &mut |_| {}).unwrap();
        assert_eq!(s.selected, 1, "{s:?}");
        assert!(s.done_pressed && act.clicks.iter().filter(|c| **c == DONE_AT).count() == 1, "{:?}", act.clicks);
        // a counter missed once is read again before anything is retried
        let g = Header::reads(&["", "37 selected"]);
        let mut act = TwoPages::default();
        let s = drive(&mut act, &g, &range_plan(), &mut |_| {}).unwrap();
        assert_eq!((s.selected, act.shift_clicks.len()), (37, 1), "{s:?}");
        // nothing on the first gesture, a count on the second: the retry took
        let g = Header::reads(&["", "", "5 selected"]);
        let mut act = TwoPages::default();
        let s = drive(&mut act, &g, &range_plan(), &mut |_| {}).unwrap();
        assert_eq!(s.selected, 5, "{s:?}");
        assert!(s.done_pressed && act.ups > 0 && act.shift_clicks.len() == 2, "{:?}", act.shift_clicks);
    }

    #[test]
    fn select_all_without_a_search_field_refuses_instead_of_taking_the_library() {
        // the live failure: the picker had not loaded its search box yet, the agent went on with
        // the whole library grid and began a range selection from its first tile
        let g = FakeGrounder { inventory: vec![tile("0:30", 10)], finds: vec![] };
        let mut act = FakeActor::new(vec![cap_of(10), cap_of(20), cap_of(30), cap_of(40), cap_of(50)]);
        let plan = Plan { select_all: true, max_items: MAX_PICK, queries: vec!["videos October 14 2022".into()], ..Plan::default() };
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
        let plan = Plan { select_all: true, max_items: MAX_PICK, queries: vec!["Dubai Police robot".into()], ..Plan::default() };
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
        plan_for(Request { album: Some(title.into()), videos_only, queries: vec!["ignored".into()], max_items: 0, select_all: false })
    }

    #[test]
    fn an_album_is_opened_from_the_albums_view_and_selected_whole() {
        let wanted = album_card("Robot showcase", 60, 60);
        let g = AlbumPicker::new(true, vec![vec![album_card("Cars", 0, 60), wanted.clone(), album_card("Robot showcase 2", 120, 60)]]);
        let mut act = album_actor();
        let plan = album_plan("Robot showcase", false);
        assert!(plan.select_all && plan.queries.is_empty() && !plan.videos_only, "{plan:?}");
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
        let plan = Plan { select_all: true, max_items: MAX_PICK, videos_only: false, queries: vec![], ..Plan::default() };
        let err = drive(&mut act, &g, &plan, &mut |_| {}).unwrap_err().to_string();
        assert!(err.contains("refusing to select every item of the whole library"), "{err}");
        assert!(act.clicks.is_empty() && act.shift_clicks.is_empty());
    }

    #[test]
    fn requests_become_plans() {
        let p = plan_for(Request { queries: vec!["beach".into()], max_items: 4, select_all: true, album: None, videos_only: true });
        assert_eq!((p.queries.as_slice(), p.max_items, p.select_all, p.videos_only, p.album), (["beach".to_string()].as_slice(), 4, true, true, None));
        let p = plan_for(Request { queries: vec![], max_items: 0, select_all: false, album: Some("Robots".into()), videos_only: false });
        assert_eq!((p.queries.is_empty(), p.select_all, p.videos_only, p.album.as_deref()), (true, true, false, Some("Robots")));
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

    use std::cell::RefCell;
    use std::rc::Rc;

    const PICKER_TAB: &str = "https://photos.google.com/picker/s1/autoclose";

    /// One picker tab driven through many searches. Every click but the search field's changes
    /// the screen; typing a search changes it (unless that search is `frozen`); the grid moves
    /// once per scroll direction (then it is at its end, or back at its top).
    #[derive(Default)]
    struct World {
        tag: u8,
        dir: i32,
        typed: Vec<String>,
        cleared: Vec<usize>,
        clicks: Vec<(u32, u32)>,
        shift_clicks: Vec<(u32, u32)>,
        frozen: Vec<&'static str>,
        /// Typing fails (no Accessibility) from this search on.
        fail_typing_at: Option<usize>,
        url: Option<String>,
        tabs_closed: u32,
    }

    #[derive(Clone)]
    struct Hands(Rc<RefCell<World>>);

    impl Act for Hands {
        fn activate(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn click_png(&mut self, _cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            let mut w = self.0.borrow_mut();
            w.clicks.push((px, py));
            if (px, py) != FIELD {
                w.tag = w.tag.wrapping_add(10);
            }
            Ok(())
        }
        fn shift_click_png(&mut self, _cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            let mut w = self.0.borrow_mut();
            w.shift_clicks.push((px, py));
            w.tag = w.tag.wrapping_add(10);
            Ok(())
        }
        fn type_text(&mut self, text: &str) -> Result<(), ScreenError> {
            let mut w = self.0.borrow_mut();
            if w.fail_typing_at.is_some_and(|n| w.typed.len() >= n) {
                return Err(ScreenError::Input("Google Chrome".into(), "not allowed assistive access".into()));
            }
            w.typed.push(text.to_string());
            if !w.frozen.contains(&text) {
                w.tag = w.tag.wrapping_add(3);
            }
            Ok(())
        }
        fn key_return(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn key_escape(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn scroll(&mut self, n: i32) -> Result<(), ScreenError> {
            let mut w = self.0.borrow_mut();
            if n.signum() != w.dir {
                w.dir = n.signum();
                w.tag = w.tag.wrapping_add(7);
            }
            Ok(())
        }
        fn sleep(&mut self, _ms: u64) {}
        fn capture(&mut self) -> Result<Capture, ScreenError> {
            Ok(cap_of(self.0.borrow().tag))
        }
        fn clear_text(&mut self, chars: usize) -> Result<(), ScreenError> {
            self.0.borrow_mut().cleared.push(chars);
            Ok(())
        }
        fn front_url(&mut self) -> Result<Option<String>, ScreenError> {
            Ok(self.0.borrow().url.clone())
        }
        fn close_tab(&mut self) -> Result<(), ScreenError> {
            let mut w = self.0.borrow_mut();
            w.tabs_closed += 1;
            w.url = None;
            Ok(())
        }
    }

    /// Eyes on the [`World`]: a search field and a Done button always; three video tiles unless
    /// the last search is `empty`; date groups for a date search (two at the top of the grid,
    /// one more further down, the second still showing); the header's counts in turn.
    struct Eyes {
        world: Rc<RefCell<World>>,
        counts: RefCell<std::collections::VecDeque<&'static str>>,
        empty: Vec<&'static str>,
        asks: RefCell<Vec<String>>,
    }

    impl Eyes {
        fn new(world: &Rc<RefCell<World>>, counts: &[&'static str]) -> Self {
            Self { world: Rc::clone(world), counts: RefCell::new(counts.iter().copied().collect()), empty: vec![], asks: RefCell::default() }
        }
    }

    fn check(label: &str, y: u32) -> Element {
        Element { label: label.into(), kind: "check".into(), x1: 4, y1: y, x2: 14, y2: y + 10 }
    }

    impl Ground for Eyes {
        fn inventory(&self, _png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
            self.asks.borrow_mut().push(task.to_string());
            let w = self.world.borrow();
            let last = w.typed.last().cloned().unwrap_or_default();
            if task == crate::picker::SELECTED_COUNTER {
                let label = self.counts.borrow_mut().pop_front().unwrap_or("");
                return Ok(if label.is_empty() { vec![] } else { vec![Element { label: label.into(), kind: String::new(), x1: 5, y1: 2, x2: 60, y2: 12 }] });
            }
            if task == crate::picker::DAY_CHECKS {
                if !crate::picker::is_date_search(&last) {
                    return Ok(vec![]);
                }
                return Ok(if w.dir <= 0 {
                    vec![check("Thu, Oct 13, 2022", 30), check("Wed, Oct 12, 2022", 90)]
                } else {
                    vec![check("Wed, Oct 12, 2022", 30), check("Tue, Oct 11, 2022", 90)]
                });
            }
            if self.empty.contains(&last.as_str()) {
                return Ok(vec![]);
            }
            let t = |l: &str, x1: u32, y1: u32| Element { label: l.into(), kind: String::new(), x1, y1, x2: x1 + 40, y2: y1 + 40 };
            Ok(vec![t("0:05", 20, 30), t("0:30", 70, 30), t("0:12", 120, 90)])
        }
        fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            if target == crate::picker::SEARCH_FIELD {
                return Ok(Some(Element { label: "Search".into(), kind: String::new(), x1: 40, y1: 0, x2: 160, y2: 8 }));
            }
            Ok(target.contains("Done").then(|| Element { label: "Done".into(), kind: String::new(), x1: 180, y1: 0, x2: 199, y2: 20 }))
        }
    }

    fn world() -> Rc<RefCell<World>> {
        Rc::new(RefCell::new(World { url: Some(PICKER_TAB.into()), ..World::default() }))
    }

    fn many(queries: &[&str], max_items: u32) -> Plan {
        plan_for(Request { queries: queries.iter().map(|q| q.to_string()).collect(), max_items, select_all: true, album: None, videos_only: true })
    }

    fn outcomes(s: &Summary) -> Vec<(Outcome, Option<u32>)> {
        s.searches.iter().map(|x| (x.outcome, x.selected)).collect()
    }

    #[test]
    fn many_searches_share_one_session_one_done_and_one_tab() {
        let w = world();
        let eyes = Eyes::new(&w, &["5 selected", "12 selected", "20 selected"]);
        let plan = many(&["robots", "drones", "Robots ", "cats"], MAX_PICK);
        assert_eq!(plan.queries, ["robots", "drones", "cats"], "repeats are searched once");
        let mut lines: Vec<String> = vec![];
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &plan, &mut |l| lines.push(l.into())).unwrap();
        let w = w.borrow();
        assert_eq!(w.typed, ["robots videos", "drones videos", "cats videos"]);
        // each search after the first empties the field of the last one first
        assert_eq!(w.cleared, ["robots videos".len() + 8, "drones videos".len() + 8]);
        assert_eq!(w.shift_clicks.len(), 3, "one range per search");
        assert_eq!(w.clicks.iter().filter(|c| **c == DONE_AT).count(), 1, "Done once, at the end: {:?}", w.clicks);
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(5)), (Outcome::Selected, Some(12)), (Outcome::Selected, Some(20))]);
        assert!(s.done_pressed && s.selected == 20 && s.left.is_empty() && s.retry().is_empty(), "{s:?}");
        assert!(s.note.contains("3 search(es)") && s.note.contains("20 selected in all"), "{}", s.note);
        // and the picker tab it drove is closed
        assert!(s.tab_closed && w.tabs_closed == 1);
        assert!(lines.iter().any(|l| l.contains("closing the picker tab")), "{lines:?}");
    }

    #[test]
    fn a_search_without_results_is_skipped_and_the_rest_go_on() {
        let w = world();
        let mut eyes = Eyes::new(&w, &["5 selected", "9 selected"]);
        eyes.empty = vec!["nothing here videos"];
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &many(&["robots", "nothing here", "cats"], MAX_PICK), &mut |_| {}).unwrap();
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(5)), (Outcome::NoResults, Some(5)), (Outcome::Selected, Some(9))]);
        assert!(s.done_pressed && s.selected == 9, "{s:?}");
        assert_eq!(w.borrow().shift_clicks.len(), 2);
        assert!(s.retry().is_empty(), "no results is not worth another try");
        // nothing anywhere: no Done, nothing to wait for
        let w = world();
        let mut eyes = Eyes::new(&w, &[]);
        eyes.empty = vec!["a videos", "b videos"];
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &many(&["a", "b"], MAX_PICK), &mut |_| {}).unwrap();
        assert!(!s.done_pressed && s.selected == 0 && s.stop_waiting(Duration::ZERO), "{s:?}");
        assert_eq!(w.borrow().clicks.iter().filter(|c| **c == DONE_AT).count(), 0);
    }

    #[test]
    fn the_session_cap_leaves_the_rest_for_a_next_session() {
        // 10 then 20 of 25: the next search could pass the cap, so it and the rest are left
        let w = world();
        let eyes = Eyes::new(&w, &["10 selected", "20 selected"]);
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &many(&["a", "b", "c", "d"], 25), &mut |_| {}).unwrap();
        assert_eq!(s.left, ["c", "d"]);
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(10)), (Outcome::Selected, Some(20))]);
        assert!(s.done_pressed && s.selected == 20);
        assert_eq!(w.borrow().typed.len(), 2, "c and d were never typed");
        // the cap reached exactly: that search may be cut short, so it is worth another try
        let w = world();
        let eyes = Eyes::new(&w, &["20 selected"]);
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &many(&["a", "b"], 20), &mut |_| {}).unwrap();
        assert_eq!(outcomes(&s), [(Outcome::AtCap, Some(20))]);
        assert_eq!((s.left.as_slice(), s.retry().as_slice()), (["b".to_string()].as_slice(), ["a".to_string()].as_slice()));
        assert!(s.done_pressed);
    }

    #[test]
    fn the_count_is_cumulative_and_a_fall_is_reported() {
        let w = world();
        let eyes = Eyes::new(&w, &["8 selected", "8 selected", "6 selected"]);
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &many(&["a", "b", "c"], MAX_PICK), &mut |_| {}).unwrap();
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(8)), (Outcome::NothingNew, Some(8)), (Outcome::Cleared, Some(6))]);
        assert!(s.searches[2].note.contains("fell from 8 to 6"), "{:?}", s.searches[2]);
        // c cleared some of what came before it: a (which selected) is worth another try too
        assert_eq!(s.retry(), ["a", "b", "c"]);
        assert!(s.done_pressed && s.selected == 6, "what is selected is still confirmed: {s:?}");
    }

    #[test]
    fn a_search_that_leaves_the_grid_as_it_was_selects_nothing() {
        // the search for b changed nothing: its "results" are a's, and a range over them would
        // clear them
        let w = world();
        w.borrow_mut().frozen = vec!["b videos"];
        let eyes = Eyes::new(&w, &["5 selected", "9 selected"]);
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &many(&["a", "b", "c"], MAX_PICK), &mut |_| {}).unwrap();
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(5)), (Outcome::Failed, Some(5)), (Outcome::Selected, Some(9))]);
        assert!(s.searches[1].note.contains("did not change"), "{:?}", s.searches[1]);
        assert_eq!(w.borrow().shift_clicks.len(), 2, "no range over b's unchanged grid");
    }

    #[test]
    fn losing_the_grants_mid_session_ends_it_and_still_closes_the_tab() {
        let w = world();
        w.borrow_mut().fail_typing_at = Some(1);
        let eyes = Eyes::new(&w, &["5 selected"]);
        let err = drive(&mut Hands(Rc::clone(&w)), &eyes, &many(&["a", "b", "c"], MAX_PICK), &mut |_| {}).unwrap_err();
        assert!(err.to_string().contains("Accessibility"), "{err}");
        let w = w.borrow();
        assert_eq!(w.clicks.iter().filter(|c| **c == DONE_AT).count(), 0, "no Done after giving up");
        assert_eq!(w.tabs_closed, 1, "the tab it gave up on is closed");
    }

    #[test]
    fn a_date_search_selects_its_day_groups_once_each() {
        let w = world();
        let eyes = Eyes::new(&w, &["14 selected"]);
        let plan = Plan { select_all: true, queries: vec!["October 2022".into()], max_items: MAX_PICK, ..Plan::default() };
        let mut lines: Vec<String> = vec![];
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &plan, &mut |l| lines.push(l.into())).unwrap();
        let w = w.borrow();
        assert_eq!(w.typed, ["October 2022 videos"]);
        let at = |y: u32| crate::picker::check_point(&check("", y));
        // Oct 13 and Oct 12 at the top; after a scroll Oct 12 (again: not clicked twice) and Oct 11
        assert_eq!(w.clicks, [FIELD, at(30), at(90), at(90), DONE_AT], "{:?}", w.clicks);
        assert!(w.shift_clicks.is_empty(), "no range gesture");
        assert_eq!((s.selected, s.done_pressed), (14, true), "{s:?}");
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(14))]);
        assert_eq!(lines.iter().filter(|l| l.contains("selected the day")).count(), 3, "{lines:?}");
        // never asked for date groups on a search that is not a date
        let w = world();
        let eyes = Eyes::new(&w, &["5 selected"]);
        drive(&mut Hands(Rc::clone(&w)), &eyes, &range_plan(), &mut |_| {}).unwrap();
        assert!(!eyes.asks.borrow().iter().any(|a| a == crate::picker::DAY_CHECKS));
    }

    #[test]
    fn day_groups_past_the_cap_are_unselected_and_left_for_a_next_session() {
        // live: the picker let a 40-item session's date groups reach 143 selected, then refused
        // Done three times and the whole session was lost. Now the same days are clicked again
        // (a second click unselects a day) and that search waits for a session with room.
        let w = world();
        let eyes = Eyes::new(&w, &["10 selected", "40 selected", "10 selected", "18 selected"]);
        let mut lines: Vec<String> = vec![];
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &many(&["robots", "October 2022", "cats"], 25), &mut |l| lines.push(l.into())).unwrap();
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(10)), (Outcome::Deferred, Some(10)), (Outcome::Selected, Some(18))]);
        assert_eq!(s.left, ["October 2022"], "{s:?}");
        assert!(s.retry().is_empty() && s.too_big().is_empty(), "{s:?}");
        assert!(s.done_pressed && s.selected == 18, "{s:?}");
        let w = w.borrow();
        let at = |y: u32| crate::picker::check_point(&check("", y));
        let days: Vec<(u32, u32)> = w.clicks.iter().copied().filter(|c| *c == at(30) || *c == at(90)).collect();
        // the grid was still scrolled from the range before it: Oct 12 and Oct 11 were selected;
        // from the top, exactly those are clicked again — Oct 12, (scroll) Oct 11 — never Oct 13
        assert_eq!(days, [at(30), at(90), at(90), at(90)], "{:?} {lines:#?}", w.clicks);
        assert_eq!(w.clicks.iter().filter(|c| **c == DONE_AT).count(), 1);
        let undone: Vec<&String> = lines.iter().filter(|l| l.contains("unselected the day")).collect();
        assert_eq!(undone.len(), 2, "{lines:?}");
        assert!(!undone.iter().any(|l| l.contains("Oct 13")), "{undone:?}");
        assert!(s.note.contains("1 past the session's maximum"), "{}", s.note);
        // the first search of a session already too big alone: given up, the next one goes on
        let w = world();
        let eyes = Eyes::new(&w, &["40 selected", "", "", "18 selected"]);
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &many(&["October 2022", "cats"], 25), &mut |_| {}).unwrap();
        assert_eq!(outcomes(&s), [(Outcome::TooBig, None), (Outcome::Selected, Some(18))]);
        assert_eq!(s.too_big(), ["October 2022"]);
        assert!(s.left.is_empty() && s.retry().is_empty() && s.done_pressed, "{s:?}");
        assert!(s.searches[0].note.contains("split it"), "{:?}", s.searches[0]);
    }

    #[test]
    fn a_lone_search_past_the_cap_is_unselected_and_never_confirmed() {
        let w = world();
        // 40 of 25 after the date groups; nothing after unselecting them (two reads)
        let eyes = Eyes::new(&w, &["40 selected", "", ""]);
        let plan = Plan { select_all: true, queries: vec!["October 2022".into()], max_items: 25, ..Plan::default() };
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &plan, &mut |_| {}).unwrap();
        assert_eq!(outcomes(&s), [(Outcome::TooBig, Some(0))]);
        assert!(!s.done_pressed && s.selected == 0 && s.stop_waiting(Duration::ZERO), "{s:?}");
        assert!(s.note.contains("more than the session's 25-item maximum"), "{}", s.note);
        assert_eq!(w.borrow().clicks.iter().filter(|c| **c == DONE_AT).count(), 0);
        // a range past the cap cannot be unselected (a second range toggles single tiles, live):
        // the agent stops before a Done the picker would refuse
        let g = Header::reads(&["37 selected"]);
        let mut act = TwoPages::default();
        let plan = Plan { max_items: 25, ..range_plan() };
        let err = drive(&mut act, &g, &plan, &mut |_| {}).unwrap_err().to_string();
        assert!(err.contains("more than the session's 25-item maximum"), "{err}");
        assert_eq!(act.clicks.iter().filter(|c| **c == DONE_AT).count(), 0, "{:?}", act.clicks);
    }

    #[test]
    fn day_groups_that_select_nothing_fall_back_to_the_range() {
        let w = world();
        // the header shows nothing after the groups (two reads), then the range's count
        let eyes = Eyes::new(&w, &["", "", "7 selected"]);
        let plan = Plan { select_all: true, queries: vec!["videos October 13 2022".into()], max_items: MAX_PICK, ..Plan::default() };
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &plan, &mut |_| {}).unwrap();
        assert_eq!((s.selected, w.borrow().shift_clicks.len()), (7, 1), "{s:?}");
        assert!(s.done_pressed);
    }

    #[test]
    fn many_queries_get_a_budget_that_grows_with_them() {
        let qs = |n: usize| (0..n).map(|i| format!("q{i}")).collect::<Vec<_>>();
        let p = plan_for(Request { queries: qs(5), max_items: MAX_PICK, select_all: true, album: None, videos_only: true });
        assert_eq!((p.max_actions, p.timeout), (ACTIONS_PER_SEARCH * 5 + DONE_RESERVE_ACTIONS, Duration::from_secs(300 + 4 * 180)));
        let p = plan_for(Request { queries: qs(500), max_items: MAX_PICK, select_all: true, album: None, videos_only: true });
        assert_eq!(p.timeout, MAX_SESSION_TIME, "bounded");
        // one query keeps the single-search budget
        let p = plan_for(Request { queries: qs(1), max_items: 6, select_all: false, album: None, videos_only: true });
        assert_eq!((p.max_actions, p.timeout), (Plan::default().max_actions, Plan::default().timeout));
        // queries are cleaned: no control characters, no blanks, no repeats, bounded
        let q = clean_queries(vec![" a\tb\nc ".into(), "".into(), "   ".into(), "A B C".into(), "x".repeat(500), "d\u{7}".into()]);
        assert_eq!(q.len(), 3, "{q:?}");
        assert_eq!(q[0], "a b c");
        assert_eq!(q[1].chars().count(), MAX_QUERY_CHARS);
        assert_eq!(q[2], "d");
    }

    #[test]
    fn many_tile_by_tile_searches_stop_at_max_items() {
        // without select-all: up to max_items tiles in all, over the searches
        let w = world();
        let eyes = Eyes::new(&w, &[]);
        let plan = plan_for(Request { queries: vec!["a".into(), "b".into(), "c".into()], max_items: 4, select_all: false, album: None, videos_only: true });
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &plan, &mut |_| {}).unwrap();
        assert_eq!(s.selected, 4, "{s:?}");
        assert_eq!(s.left, ["c"], "{s:?}");
        assert_eq!(s.searches.iter().map(|x| x.outcome).collect::<Vec<_>>(), [Outcome::Selected, Outcome::Selected]);
        assert!(s.done_pressed);
    }
}
