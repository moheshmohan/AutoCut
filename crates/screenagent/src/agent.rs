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
    /// Pick every item of the user's album with this title instead: search the picker for the
    /// exact title and press Return — Google's picker answers an album's exact title with the
    /// album's items (it has no Albums view and never suggests an album by name, live) — check
    /// the search visibly ran, select the whole result grid, press Done. A search that showed
    /// nothing new, a grid that never ends within the budget, or more than `max_items` selected
    /// is an error, and Done is never pressed. `queries` is ignored; `select_all` is implied.
    pub album: Option<String>,
    /// Pick exactly these items instead — ones a catalogue found, say by what they look like:
    /// each is searched by its day and ticked by its tile in the picker's page, matched by when it
    /// was taken ([`crate::items`]); no other tile is ticked. Needs the page (Chrome with View ▸
    /// Developer ▸ Allow JavaScript from Apple Events): by eyes the tiles' times cannot be read.
    /// `queries`, `album` and `select_all` are then unused.
    pub items: Vec<crate::items::Wanted>,
}

impl Default for Plan {
    fn default() -> Self {
        Self {
            queries: Vec::new(),
            max_items: 6,
            videos_only: true,
            max_actions: 40,
            timeout: Duration::from_secs(300),
            select_all: false,
            album: None,
            items: Vec::new(),
        }
    }
}

impl Plan {
    /// `(query, the text typed for it)` per search ([`picker::search_text`]; a resumed date
    /// search is typed without the day it resumes from, [`picker::resume_point`]); a videos-only
    /// plan with no query still searches "videos".
    fn searches(&self) -> Vec<(String, String)> {
        if self.queries.is_empty() {
            return picker::search_text(None, self.videos_only).map(|t| vec![(t.clone(), t)]).unwrap_or_default();
        }
        self.queries.iter().filter_map(|q| picker::search_text(Some(picker::resume_point(q).0), self.videos_only).map(|t| (q.clone(), t))).collect()
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
    /// Only videos: searches ask Google for videos. An album is selected whole, photos too (the
    /// picker cannot select by type): the caller skips its photos when it downloads.
    pub videos_only: bool,
    /// Pick exactly these items instead (see [`Plan::items`]); the rest is then unused.
    pub items: Vec<crate::items::Wanted>,
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
    /// Its days did not all fit under the session's cap: those that fit are selected, and the
    /// search resumes from the first that did not in a next session ([`Summary::left`],
    /// [`picker::resume_from`]).
    Split,
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
    /// The search for the album of [`Plan::album`] visibly ran — the view moved after Return and
    /// its grid is not the library's — and its result grid was taken for the album.
    pub album_shown: bool,
    pub done_pressed: bool,
    pub note: String,
    /// Each search typed, in order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub searches: Vec<Searched>,
    /// Searches for a next session: those this session did not get to (its item cap, or its
    /// budget), a date search deferred whole or resumed where it was split, and day searches for
    /// days no click selected.
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
    if !req.items.is_empty() {
        let items: Vec<crate::items::Wanted> = req.items.into_iter().take(crate::items::MAX_ITEMS).collect();
        let searches = item_searches(&items);
        let n = u32::try_from(searches).unwrap_or(u32::MAX);
        return Plan {
            max_items: u32::try_from(items.len()).unwrap_or(MAX_PICK).min(MAX_PICK),
            items,
            videos_only: false,
            max_actions: ITEM_ACTIONS_PER_SEARCH.saturating_mul(n).saturating_add(DONE_RESERVE_ACTIONS),
            timeout: ITEM_BASE_TIME.saturating_add(ITEM_TIME_PER_SEARCH.saturating_mul(n)).min(MAX_SESSION_TIME),
            ..Plan::default()
        };
    }
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
            Plan {
                queries,
                max_items: req.max_items,
                select_all: req.select_all,
                videos_only: req.videos_only,
                max_actions,
                timeout,
                album: None,
                items: Vec::new(),
            }
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
    let result = drive_inner(actor, grounder, plan, driven.as_ref(), progress);
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

fn drive_inner<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    driven: Option<&crate::tabs::Tab>,
    progress: &mut dyn FnMut(&str),
) -> Result<Summary, ScreenError> {
    let started = Instant::now();
    let mut summary = Summary::default();

    actor.activate()?;
    actor.sleep(600);
    let mut cap = actor.capture()?;

    // exact items: by their days and their tiles' times, in the page only
    if !plan.items.is_empty() {
        let Some(page) = open_page(actor, driven, progress) else {
            return Err(ScreenError::Other(ITEMS_NEED_PAGE.into()));
        };
        pick_items(actor, grounder, plan, &mut summary, &mut cap, &page, started, progress)?;
        return finish(actor, grounder, plan, summary, cap, started, progress);
    }

    // 1–2. the album's search result, selected whole (or an error with nothing selected): tile by
    // tile in the picker's page where the browser runs scripts in it, else by eyes that never see
    // anything in the search field, which holds its title; or the searches — for videos always,
    // so Google's own filter leaves only videos in the grid
    if let Some(title) = plan.album.as_deref() {
        let mut page = open_page(actor, driven, progress);
        let field = search_album(actor, grounder, plan, &mut summary, &mut cap, title, started, &mut page, progress)?;
        let eyes = OutsideField { eyes: grounder, field };
        let by_page = match &page {
            Some(p) => tick_album(actor, plan, &mut summary, p, title, started, progress)?,
            None => false,
        };
        if by_page {
            cap = actor.capture()?;
        } else if select_everything(actor, &eyes, plan, &mut summary, &mut cap, started, progress)?.is_none() {
            summary.note = no_item_note(title);
        }
        return finish(actor, &eyes, plan, summary, cap, started, progress);
    }
    let searches = plan.searches();
    if searches.len() > 1 {
        search_many(actor, grounder, plan, &mut summary, &mut cap, started, &searches, progress)?;
    } else if search_one(actor, grounder, plan, &mut summary, &mut cap, started, searches.first(), progress)? == Tiles::WindowClosed {
        return Ok(summary);
    }
    finish(actor, grounder, plan, summary, cap, started, progress)
}

/// Step 3: press Done on what is selected — never past the session's item cap — and check the
/// picker took it.
fn finish<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    mut summary: Summary,
    mut cap: crate::capture::Capture,
    started: Instant,
    progress: &mut dyn FnMut(&str),
) -> Result<Summary, ScreenError> {
    let deadline = || started.elapsed() >= plan.timeout;
    // the picker selects past the session's item cap (date groups, ranges) but then refuses
    // Done (live: "143 selected", "Select up to 40 items", Done greyed out): never press it. An
    // album's search past it is a runaway (Google's ordinary results for a title that is no
    // album's, or the library): never confirmed either
    let cap_items = plan.max_items.clamp(1, MAX_PICK);
    if plan.select_all && !summary.done_pressed && summary.selected > cap_items {
        return Err(ScreenError::Stalled(match plan.album.as_deref() {
            Some(title) => format!(
                "the search for the album `{title}` selected {}, more than this run's {cap_items}-item maximum — more than the album can be (is `{title}` an album's exact title? Google answers any other text with ordinary search results); Done was not pressed, nothing was picked",
                summary.selected
            ),
            None => format!("{} are selected, more than the session's {cap_items}-item maximum, and the picker refuses Done past it", summary.selected),
        }));
    }

    // confirm, and check the picker took it
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
        summary.note = match (summary.done_pressed, plan.album.as_deref()) {
            (true, Some(title)) => format!(
                "selected all {} item(s) of the album `{title}` and pressed Done{}",
                summary.selected,
                if plan.videos_only { " (its photos too — the picker cannot select by type; they are skipped when downloading)" } else { "" }
            ),
            (true, None) => format!("selected {} item(s) and pressed Done", summary.selected),
            (false, _) => "nothing selected".into(),
        };
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
    let mut dates = None;
    let mut days = None;
    if let Some((query, q)) = search {
        if spend(summary, plan)? || started.elapsed() >= plan.timeout {
            return Err(budget(summary));
        }
        // the picker may still be loading: look again a few times before going on without it
        match find_search_field(actor, grounder, cap, progress)? {
            Some(field) => {
                type_search(actor, cap, &field, q, None, progress)?;
                summary.searched = true;
                dates = Some(DateWalk { typed: q, resume: picker::resume_point(query).1, earlier: &[] });
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
                "nothing was searched; refusing to select every item of the whole library (selecting all needs a search, or an album's)".into(),
            ));
        }
        days = take_all(actor, grounder, plan, summary, cap, started, dates, progress)?;
        Tiles::Done
    } else {
        pick_tiles(actor, grounder, plan, summary, cap, started, progress)?
    };
    if let Some((query, typed)) = search.filter(|_| summary.searched) {
        let cap_items = plan.max_items.clamp(1, MAX_PICK);
        let (outcome, note, left) = match (&days, summary.selected, plan.select_all) {
            (Some(days), _, _) => days_outcome(days, query, typed, None, cap_items),
            (None, 0, _) => (Outcome::NoResults, String::new(), Vec::new()),
            (None, n, true) if n >= cap_items => (Outcome::AtCap, String::new(), Vec::new()),
            _ => (Outcome::Selected, String::new(), Vec::new()),
        };
        if !note.is_empty() {
            summary.note = format!("`{query}`: {note}");
        }
        summary.left = left;
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
    matches!(e, ScreenError::Capture(..) | ScreenError::Input(..) | ScreenError::NotInFront(_) | ScreenError::Model(..) | ScreenError::Unsupported(..))
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
    /// Searches it leaves for a next session: itself (deferred, or resumed where it was split),
    /// and day searches.
    left: Vec<String>,
    /// The days it selected by their date groups.
    days: Vec<String>,
}

/// Search `query`, typed as `typed` (after emptying the field when `clear`), then select its
/// results: everything on top of the `before` already selected, or tiles up to `max_items` in all.
/// `range_room`: a range gesture is not expected to pass the session's cap (a range cannot be
/// undone); without it, a date search whose day groups took nothing waits for a next session.
/// `earlier`: the days earlier searches of the session selected.
#[allow(clippy::too_many_arguments)]
fn one_of_many<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    (query, typed): (&str, &str),
    clear: Option<usize>,
    before: Option<u32>,
    (range_room, earlier): (bool, &[String]),
    progress: &mut dyn FnMut(&str),
) -> Result<OneSearch, ScreenError> {
    let mut one = OneSearch { typed: false, outcome: Outcome::Failed, count: before, note: String::new(), closed: false, left: Vec::new(), days: Vec::new() };
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
    let walk = DateWalk { typed, resume: picker::resume_point(query).1, earlier };
    let after = match by_day_groups(actor, grounder, plan, summary, cap, started, before, Some(walk), progress)? {
        ByDays::Days(days) => {
            (one.outcome, one.note, one.left) = days_outcome(&days, query, typed, before, cap_items);
            one.count = days.count.or(before);
            one.days = days.picked;
            return Ok(one);
        }
        ByDays::Range if !range_room && picker::is_date_search(typed) => {
            one.outcome = Outcome::Deferred;
            one.note = format!(
                "its date groups selected nothing, and its range could pass the session's {cap_items}-item maximum beside the {} selected (a range cannot be undone); it waits for a next session",
                before.unwrap_or(0)
            );
            one.left = vec![query.to_string()];
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
/// room is checked: the session's item cap and the budget, part of which stays back for Done.
/// A range cannot be undone, so before one the biggest search so far is the guess of its size;
/// a date search is taken day by day up to the cap instead, and the first search that does not
/// all fit (split, or deferred beside what is selected) ends the session. What does not fit is
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
    let mut full = false; // a search did not fit beside what is selected
    let mut held: Vec<String> = Vec::new(); // the days the session's date searches selected
    for (i, (query, typed)) in searches.iter().enumerate() {
        let so_far = if plan.select_all { count.unwrap_or(0) } else { summary.selected };
        if i > 0 {
            let actions_left = inner.max_actions.saturating_sub(summary.actions);
            let time_left = inner.timeout.saturating_sub(started.elapsed());
            let why = if so_far >= room {
                Some(format!("the session holds its {room}-item maximum"))
            } else if full {
                Some(format!("{so_far} are selected and the last search did not all fit under the session's {room}-item maximum"))
            } else if plan.select_all && !picker::is_date_search(typed) && so_far.saturating_add(biggest) > room {
                // a range cannot be undone; date groups are taken day by day up to the cap
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
        let range_room = so_far.saturating_add(biggest) <= room;
        let result = one_of_many(actor, grounder, &inner, summary, cap, started, (query, typed), holds, count, (range_room, &held), progress);
        let one = match result {
            Ok(one) => one,
            Err(e) if fatal(&e) => return Err(e),
            Err(e) => {
                // the gesture may have changed the selection part way: read where it stands
                let now = if plan.select_all { read_selected(grounder, cap).ok().flatten().or(count) } else { count };
                OneSearch { typed: true, outcome: Outcome::Failed, count: now, note: e.to_string(), closed: false, left: Vec::new(), days: Vec::new() }
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
        // a search deferred whole or resumed, and day searches: first in the next session
        deferred.extend(one.left);
        held.extend(one.days);
        full |= one.outcome == Outcome::Split || (one.outcome == Outcome::Deferred && count.unwrap_or(0) > 0);
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
    let split = with(Outcome::Split);
    summary.note = format!(
        "{} search(es): {} selected from, {} without results, {} with nothing new, {} failed{}{}{}; {} selected in all",
        summary.searches.len(),
        with(Outcome::Selected) + with(Outcome::AtCap) + split,
        with(Outcome::NoResults),
        with(Outcome::NothingNew),
        with(Outcome::Failed) + with(Outcome::Cleared),
        if split == 0 { String::new() } else { format!(", {split} split (to resume in a next session)") },
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
        Outcome::Split => "split: the days that fit are selected, and it resumes from the next in a next session",
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

/// Search suggestions need this long to appear under the field.
const SUGGEST_SETTLE_MS: u64 = 1500;

/// Eyes that never see anything inside the picker's search field ([`picker::in_field`]): in an
/// album pick the field holds the album's title, which the model took live for "the album", for
/// the album's heading and for a tile. Only the field itself is still found there.
struct OutsideField<'a, G: Ground> {
    eyes: &'a G,
    field: Element,
}

impl<G: Ground> Ground for OutsideField<'_, G> {
    fn inventory(&self, png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
        let mut els = self.eyes.inventory(png, task)?;
        els.retain(|e| !picker::in_field(e, &self.field));
        Ok(els)
    }

    fn find(&self, png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
        let found = self.eyes.find(png, target)?;
        if target == picker::SEARCH_FIELD {
            return Ok(found);
        }
        Ok(found.filter(|e| !picker::in_field(e, &self.field)))
    }
}

/// What a grid shows on one screen, to tell one grid from another: the words of its headings
/// ("thu oct 16 2025", "today": a heading of this week names no month) and its tiles (photos
/// too) — where each is clicked and what it reads (a duration, or "photo"), since in a uniform
/// grid every full screen has tiles at the same places.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GridLook {
    headings: Vec<String>,
    tiles: Vec<((u32, u32), String)>,
}

impl GridLook {
    fn of<G: Ground>(eyes: &G, cap: &crate::capture::Capture) -> Result<Self, ScreenError> {
        let (w, h) = (cap.png_w, cap.png_h);
        let band = (h / 16).max(1);
        let mut heads: Vec<Element> = eyes.inventory(&cap.png, picker::DAY_HEADINGS)?.into_iter().filter(|e| picker::heading_sized(e, w, h)).collect();
        heads.sort_by_key(|e| (e.y1 / band, e.x1));
        let headings = heads.iter().map(|e| picker::normalize_title(&e.label)).filter(|l| !l.is_empty()).collect();
        let tiles = picker::tile_candidates(&eyes.inventory(&cap.png, picker::ALL_TILES)?, false, w, h)
            .iter()
            .map(|t| (picker::click_point(t), picker::normalize_title(&t.label)))
            .collect();
        Ok(Self { headings, tiles })
    }

    /// Whether two looks are of one grid: the same headings, and the same tiles at the same places.
    fn same_as(&self, other: &Self) -> bool {
        self.headings == other.headings
            && self.tiles.len() == other.tiles.len()
            && self.tiles.iter().zip(&other.tiles).all(|((a, la), (b, lb))| a.0.abs_diff(b.0) < SAME_SPOT && a.1.abs_diff(b.1) < SAME_SPOT && la == lb)
    }
}

/// Looks at the grid after Return before an album's search counts as showing nothing new: in the
/// page a second apart ([`crate::dom::wait_for_results`]), by eyes [`SEARCH_SETTLE_MS`] apart.
/// Live (2026-10-24), the album "akku tesla toy" showed its items only after the two looks 2.5 s
/// apart the agent then made had given up on it.
const ALBUM_PAGE_LOOKS: u32 = 30;
const ALBUM_EYE_LOOKS: u32 = 8;

/// The picker's page, where the browser runs scripts in it: the tab, and the tiles the library
/// showed before the search (read once the picker has loaded, [`crate::dom::settled_tiles`]).
struct Page {
    tab: crate::tabs::Tab,
    library: Vec<String>,
}

/// The page of the picker tab the agent drives, when the browser runs scripts in it (Chrome with
/// View ▸ Developer ▸ Allow JavaScript from Apple Events); `None`: the eyes do it all.
fn open_page<A: Act>(actor: &mut A, driven: Option<&crate::tabs::Tab>, progress: &mut dyn FnMut(&str)) -> Option<Page> {
    let tab = driven?;
    match crate::dom::run(actor, tab, crate::dom::Step::Look) {
        Ok(_) => {
            progress("screenagent: reading the picker's page: its tiles are ticked by their checkboxes");
            Some(Page { tab: tab.clone(), library: Vec::new() })
        }
        Err(ScreenError::Unsupported(why)) => {
            progress(&format!("screenagent: {why}; selecting by eyes"));
            None
        }
        Err(e) => {
            progress(&format!("screenagent: cannot read the picker's page ({e}); selecting by eyes"));
            None
        }
    }
}

fn no_item_note(title: &str) -> String {
    format!("searching the picker for `{title}` showed no item to select; nothing was selected")
}

/// Tick every tile of the album's search result in the page ([`crate::dom::tick_all`]).
/// `Ok(false)`: the page stopped answering or its clicks selected nothing — the eyes go on.
/// More tiles than this run's maximum, or tiles that would not tick, end the pick unconfirmed.
fn tick_album<A: Act>(
    actor: &mut A,
    plan: &Plan,
    summary: &mut Summary,
    page: &Page,
    title: &str,
    started: Instant,
    progress: &mut dyn FnMut(&str),
) -> Result<bool, ScreenError> {
    let cap_items = plan.max_items.clamp(1, MAX_PICK);
    let out_of_time = || started.elapsed() >= plan.timeout;
    let ticked = match crate::dom::tick_all(actor, &page.tab, cap_items, &out_of_time, progress) {
        Ok(t) => t,
        Err(e @ ScreenError::Stalled(_)) => return Err(e),
        Err(e) => {
            progress(&format!("screenagent: the picker's page stopped answering ({e}); selecting by eyes instead"));
            return Ok(false);
        }
    };
    if ticked.tiles == 0 {
        summary.selected = 0;
        summary.note = no_item_note(title);
        return Ok(true);
    }
    if ticked.runaway {
        return Err(ScreenError::Stalled(format!(
            "the search for the album `{title}` showed more than this run's {cap_items}-item maximum — more than the album can be (is `{title}` an album's exact title? Google answers any other text with ordinary search results); Done was not pressed, nothing was picked"
        )));
    }
    let Some(n) = ticked.selected else {
        progress("screenagent: ticking the tiles in the page selected nothing; selecting by eyes instead");
        return Ok(false);
    };
    summary.selected = n;
    if !ticked.unticked.is_empty() {
        let shown: Vec<&str> = ticked.unticked.iter().take(3).map(String::as_str).collect();
        return Err(ScreenError::Stalled(format!(
            "{} of the album's {} tile(s) would not tick after {} walk(s) of the grid ({}); Done was not pressed, nothing was picked",
            ticked.unticked.len(),
            ticked.tiles,
            ticked.walks,
            shown.join("; ")
        )));
    }
    progress(&format!("screenagent: ticked all {} tile(s) of the album in the page; the picker shows {n} selected", ticked.tiles));
    Ok(true)
}

/// Search the picker for the album titled `title`, and check the search visibly ran. Google's
/// picker has no Albums view and never suggests an album by name (live, 2026-10-19: generic
/// suggestions and people only); Return on the album's exact title shows the album's items,
/// newest first, under date headings — the grid of any search result. So that grid is the album.
/// The search must visibly have run, and it answers after a delay: with the picker's `page`, its
/// tiles are no longer the library's and have settled ([`crate::dom::wait_for_results`], up to
/// [`ALBUM_PAGE_LOOKS`] s); by eyes, the view moved after Return ([`verify::moved`], looked at up
/// to [`ALBUM_EYE_LOOKS`] times while results load) and its first screen is not the unfiltered
/// library's ([`GridLook`]) — else nothing is selected: live, the model took the typed title for
/// the album and for its heading, and the agent selected the library. Nothing inside the search
/// field ever counts ([`OutsideField`]). Returns the field.
#[allow(clippy::too_many_arguments)]
fn search_album<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    title: &str,
    started: Instant,
    page: &mut Option<Page>,
    progress: &mut dyn FnMut(&str),
) -> Result<Element, ScreenError> {
    if picker::normalize_title(title).is_empty() {
        return Err(ScreenError::Other(format!("the album title `{title}` has no letters or digits to search for")));
    }
    let out_of_time = || started.elapsed() >= plan.timeout;
    if spend(summary, plan)? || out_of_time() {
        return Err(budget(summary));
    }
    let Some(field) = find_search_field(actor, grounder, cap, progress)? else {
        return Err(ScreenError::Other(format!(
            "the picker's search field never appeared, so the album `{title}` could not be searched; nothing was selected"
        )));
    };
    let eyes = OutsideField { eyes: grounder, field: field.clone() };
    // the unfiltered library, as the picker opened on it — in the page once its grid has loaded
    // and settled (a library still loading must not pass for the search's results)
    let library = GridLook::of(&eyes, cap)?;
    if let Some(p) = page.as_mut() {
        match crate::dom::settled_tiles(actor, &p.tab) {
            Ok(Some(tiles)) => p.library = tiles,
            Ok(None) => {
                progress("screenagent: the picker's page shows no settled grid; selecting by eyes");
                *page = None;
            }
            Err(e) => {
                progress(&format!("screenagent: cannot read the picker's page ({e}); selecting by eyes"));
                *page = None;
            }
        }
    }
    progress(&format!("screenagent: searching the picker for the album `{title}`"));
    let (px, py) = field.center();
    actor.click_png(cap, px, py)?;
    actor.sleep(400);
    actor.type_text(title)?;
    summary.searched = true;
    // the suggestions under the field never name an album: only what Return does counts
    actor.sleep(SUGGEST_SETTLE_MS);
    let typed = actor.capture()?;
    if spend(summary, plan)? || out_of_time() {
        return Err(budget(summary));
    }
    actor.key_return()?;
    let nothing_new = |why: &str| ScreenError::Other(format!("searching the picker for `{title}` showed nothing new ({why}); nothing was selected"));
    if let Some(p) = page.as_ref() {
        progress("screenagent: waiting for the search's results in the picker's page");
        match crate::dom::wait_for_results(actor, &p.tab, &p.library, ALBUM_PAGE_LOOKS) {
            Ok(crate::dom::Results::Shown(look)) => {
                summary.album_shown = true;
                progress(&format!(
                    "screenagent: the search for `{title}` shows {} tile(s) so far; taking them for the album and ticking all of them",
                    look.tiles.len()
                ));
                *cap = actor.capture()?;
                return Ok(field);
            }
            Ok(crate::dom::Results::Empty) => {
                summary.album_shown = true;
                *cap = actor.capture()?;
                return Ok(field);
            }
            Ok(crate::dom::Results::Unchanged) => {
                return Err(nothing_new(&format!("the grid showed the library's tiles for {ALBUM_PAGE_LOOKS} s after Return")));
            }
            Err(e) => progress(&format!("screenagent: the picker's page stopped answering ({e}); looking by eyes")),
        }
    }
    actor.sleep(SEARCH_SETTLE_MS);
    *cap = actor.capture()?;
    let mut looks = 1u32;
    while !verify::moved(&typed.png, &cap.png)? {
        if looks >= ALBUM_EYE_LOOKS {
            return Err(nothing_new(&format!("the view did not move in the {} s after Return", u64::from(looks) * SEARCH_SETTLE_MS / 1000)));
        }
        if looks == 1 {
            progress("screenagent: the view has not moved since Return; waiting for the results");
        }
        if out_of_time() {
            return Err(budget(summary));
        }
        actor.sleep(SEARCH_SETTLE_MS);
        *cap = actor.capture()?;
        looks += 1;
    }
    if GridLook::of(&eyes, cap)?.same_as(&library) {
        return Err(nothing_new("the grid is the library's, as it was before the search"));
    }
    summary.album_shown = true;
    progress(&format!("screenagent: the search for `{title}` shows its results; taking them for the album and selecting all of them"));
    Ok(field)
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

/// Select everything a search shows: by the date groups of a date search (`dates`) first, else (or when they select nothing) by the range gesture with
/// its retry. `summary.selected` is the header's count after it. `Some(days)`: the date groups
/// were walked, and [`days_outcome`] says what came of it.
#[allow(clippy::too_many_arguments)]
fn take_all<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    dates: Option<DateWalk>,
    progress: &mut dyn FnMut(&str),
) -> Result<Option<Days>, ScreenError> {
    if let ByDays::Days(days) = by_day_groups(actor, grounder, plan, summary, cap, started, None, dates, progress)? {
        summary.selected = days.count.unwrap_or(0);
        return Ok(Some(days));
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
/// Clicks on one day at most: the one that selects it, and those that put the count back where
/// it was when that went past the cap or the day was selected already.
const DAY_CLICKS: u32 = 3;
/// The wait before a second look for date headings when the first page showed none.
const DAY_LOOK_AGAIN_MS: u64 = 1500;

/// A date search to walk by its day groups.
#[derive(Debug, Clone, Copy)]
struct DateWalk<'a> {
    /// What was typed (the searched date).
    typed: &'a str,
    /// The day it resumes from ([`picker::resume_point`]): the days after it (newer) were picked
    /// by an earlier session.
    resume: Option<&'a str>,
    /// The days earlier searches of this session selected.
    earlier: &'a [String],
}

/// What selecting by date groups came to.
#[derive(Debug, Clone, PartialEq, Eq)]
enum ByDays {
    /// No date group took: use the range gesture (the grid is back at its top).
    Range,
    /// The walk over the date groups ([`select_day_groups`]).
    Days(Days),
}

/// What a walk over a date search's day groups did, day by day, by the header's count.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Days {
    /// A date heading was on the first page.
    found: bool,
    /// Headings of other dates than the searched one (Google answers a date it has nothing for
    /// with neighbouring days, and other years'), not selected.
    foreign: u32,
    /// Days of a resumed search newer than its resume point: an earlier session's, not selected.
    resumed_past: u32,
    /// The days it selected ([`picker::day_key`]), in order.
    picked: Vec<String>,
    /// Days an earlier search of the session selected: the click lowered the count, and one more
    /// put them back.
    had: Vec<String>,
    /// Days no click selected (no check button where the heading is, and no lone tile), or whose
    /// click hit another day selected already (a misplaced heading box): one more click put that
    /// back.
    missed: Vec<String>,
    /// Days whose items alone are more than the session's cap: unselected again, skipped.
    too_big: Vec<String>,
    /// The day that did not fit beside what was selected (clicked again at once, which unselects
    /// it): the walk stopped there, and the search resumes from it in a next session.
    over: Option<String>,
    /// The count the last day past the cap took the header to.
    passed: Option<u32>,
    /// The header's count after the walk (`None`: nothing selected).
    count: Option<u32>,
    scrolled: bool,
}

/// Select a date search's results by their date groups ([`select_day_groups`]) — only for a date
/// search (`dates`) — on top of the `before` already selected. Results that show only other dates'
/// days are reported as they are (a range would select another search's days); when no day took
/// otherwise (no dated heading, or every click selected nothing) the grid is put back at its top
/// for the range gesture.
#[allow(clippy::too_many_arguments)]
fn by_day_groups<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    before: Option<u32>,
    dates: Option<DateWalk>,
    progress: &mut dyn FnMut(&str),
) -> Result<ByDays, ScreenError> {
    let Some(walk) = dates.filter(|w| picker::is_date_search(w.typed)) else { return Ok(ByDays::Range) };
    let days = select_day_groups(actor, grounder, plan, summary, cap, started, before, walk, progress)?;
    if !days.found {
        return Ok(ByDays::Range);
    }
    let took = !(days.picked.is_empty() && days.had.is_empty() && days.over.is_none() && days.too_big.is_empty());
    if !took && days.missed.is_empty() && (days.foreign > 0 || days.resumed_past > 0) {
        progress(&format!("screenagent: `{}` shows no day of its own left to select", walk.typed));
        return Ok(ByDays::Days(days));
    }
    if !took {
        progress("screenagent: the date groups selected nothing; selecting the range instead");
        if days.scrolled {
            back_to_top(actor, plan, summary, cap, started)?;
        }
        return Ok(ByDays::Range);
    }
    progress(&format!(
        "screenagent: {} day(s) selected by their date groups{}; the picker shows {} selected",
        days.picked.len(),
        days.over.as_deref().map(|k| format!(", stopped at {k}, which did not fit")).unwrap_or_default(),
        days.count.unwrap_or(0)
    ));
    Ok(ByDays::Days(days))
}

/// Walk a date search's day headings ([`picker::DAY_HEADINGS`]), paging down, and select each day
/// of the searched date once ([`picker::day_key`]: a heading still on screen after a scroll is the
/// same day) on top of the `before` already selected. A day is selected by its round check button,
/// clicked on the heading's own row — only there: asked for check buttons instead, the model boxed
/// tiles' selection circles under other headings and labelled them with dates (live) — at the left
/// end of the heading's box, then just left of it ([`picker::check_points`]); a heading with no
/// check button (a day of one item, in compact rows) by its lone tile ([`picker::lone_tile`]). The
/// header's count is read after every click ([`click_day`]): hover alone changes the pixels, so
/// only the count tells a click that selected nothing, a day selected already (put back: an earlier
/// search's day, or else a misplaced box that hit another day), or a day past the session's cap
/// (taken off again at once): a day too big alone is skipped; any other ends the walk there, to
/// resume from in a next session. A resumed search skips the days newer than its resume point. No
/// dated heading on the first page (looked at twice): `found` false, nothing done.
#[allow(clippy::too_many_arguments)]
fn select_day_groups<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    started: Instant,
    before: Option<u32>,
    walk: DateWalk,
    progress: &mut dyn FnMut(&str),
) -> Result<Days, ScreenError> {
    let cap_items = plan.max_items.clamp(1, MAX_PICK);
    let out_of_time = || started.elapsed() >= plan.timeout;
    let mut days = Days { count: before, ..Days::default() };
    let mut seen: Vec<String> = Vec::new();
    let (mut still, mut look) = (0u32, true);
    'pages: for page in 0..DAY_PAGES {
        if look {
            let mut headings = picker::day_checks(&grounder.inventory(&cap.png, picker::DAY_HEADINGS)?, cap.png_w, cap.png_h);
            if page == 0 && headings.is_empty() {
                // the results may still be rendering (live, a day search's heading was missed
                // once and found on the next try): one more look before the range
                actor.sleep(DAY_LOOK_AGAIN_MS);
                *cap = actor.capture()?;
                headings = picker::day_checks(&grounder.inventory(&cap.png, picker::DAY_HEADINGS)?, cap.png_w, cap.png_h);
                if headings.is_empty() {
                    return Ok(days);
                }
            }
            days.found = true;
            // this page's tiles, grounded once a heading without a check button needs one
            let mut tiles: Option<Vec<Element>> = None;
            for c in &headings {
                let label = c.label.trim();
                let Some(key) = picker::day_key(label, walk.typed) else {
                    if !seen.iter().any(|s| s == label) {
                        progress(&format!("screenagent: `{label}` is not a day of `{}`; not selecting it", walk.typed));
                        seen.push(label.to_string());
                        days.foreign = days.foreign.saturating_add(1);
                    }
                    continue;
                };
                if seen.contains(&key) {
                    continue;
                }
                seen.push(key.clone());
                if let Some(from) = walk.resume
                    && (key.len() < from.len() || key.as_str() > from)
                {
                    days.resumed_past = days.resumed_past.saturating_add(1); // an earlier session's
                    continue;
                }
                let mut click = Click::Nothing;
                let mut by_tile = false;
                for at in picker::check_points(c) {
                    if spend(summary, plan)? || out_of_time() {
                        return Err(budget(summary));
                    }
                    click = click_day(actor, grounder, summary, cap, at, days.count, cap_items)?;
                    if click != Click::Nothing {
                        break;
                    }
                }
                if click == Click::Nothing {
                    // no check button: a day of one item, selected by its tile
                    if tiles.is_none() {
                        tiles = Some(visible_tiles(grounder, cap, plan.videos_only, progress)?);
                    }
                    if let Some(t) = picker::lone_tile(c, &headings, tiles.as_deref().unwrap_or_default()) {
                        if spend(summary, plan)? || out_of_time() {
                            return Err(budget(summary));
                        }
                        click = click_day(actor, grounder, summary, cap, picker::click_point(&t), days.count, cap_items)?;
                        by_tile = true;
                    }
                }
                match click {
                    Click::Took(n) => {
                        progress(&format!("screenagent: selected the day `{label}`{} ({n} selected)", if by_tile { " by its one tile" } else { "" }));
                        days.picked.push(key);
                        days.count = Some(n);
                    }
                    Click::Had if walk.earlier.contains(&key) => {
                        progress(&format!("screenagent: the day `{label}` was selected already; clicked it back on"));
                        days.had.push(key);
                    }
                    Click::Had => {
                        progress(&format!(
                            "screenagent: the click for the day `{label}` hit another day selected already; clicked it back on, and `{label}` is left for a day search"
                        ));
                        days.missed.push(key);
                    }
                    Click::Over(n) => {
                        days.passed = Some(n);
                        if days.count.unwrap_or(0) == 0 {
                            progress(&format!(
                                "screenagent: the day `{label}` alone took the count to {n}, past the session's {cap_items}-item maximum (the picker refuses Done past it); unselected it again, and going on without it"
                            ));
                            days.too_big.push(key);
                        } else {
                            progress(&format!(
                                "screenagent: the day `{label}` took the count to {n}, past the session's {cap_items}-item maximum (the picker refuses Done past it); unselected it again, and the search resumes from it in a next session"
                            ));
                            days.over = Some(key);
                            break 'pages;
                        }
                    }
                    Click::Nothing => {
                        progress(&format!("screenagent: clicking the day `{label}` selected nothing (no check button, and no tile of its own)"));
                        days.missed.push(key);
                    }
                }
            }
        }
        if spend(summary, plan)? || out_of_time() {
            return Err(budget(summary));
        }
        let shown = cap.png.clone();
        actor.scroll(SCROLL_LINES)?;
        actor.sleep(900);
        *cap = actor.capture()?;
        look = verify::changed(&shown, &cap.png)?;
        if look {
            (days.scrolled, still) = (true, 0);
        } else {
            still += 1;
            if still >= 2 {
                break; // the end of the results
            }
        }
    }
    Ok(days)
}

/// What clicking a day's check button (or its lone tile) did, by the header's count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Click {
    /// The count did not move: the click selected nothing.
    Nothing,
    /// The day is selected: the count rose to this, within the session's cap.
    Took(u32),
    /// The click lowered the count (it unselected something selected already); clicked again,
    /// it is back.
    Had,
    /// The day took the count to this, past the session's cap; clicked again, it is unselected.
    Over(u32),
}

/// Click `at` — a day's check button or its lone tile — and read the header's count after it,
/// from `before`: up within the session's cap, the day is selected; up past the cap, or down (it
/// unselected something selected already: a second click unselects a day, live), the same spot is
/// clicked again until the count is back at `before` — at most [`DAY_CLICKS`] clicks, the extra
/// ones counted but never stopped by the budget (stopping there would leave a selection the picker
/// refuses Done on, or one short of what was there). A header that shows no count after a
/// selection is an error: guessing could unselect a day that took.
fn click_day<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    at: (u32, u32),
    before: Option<u32>,
    cap_items: u32,
) -> Result<Click, ScreenError> {
    use std::cmp::Ordering;
    let b = before.unwrap_or(0);
    let (mut now, mut passed) = (b, None);
    for n in 0..DAY_CLICKS {
        if n > 0 {
            summary.actions = summary.actions.saturating_add(1);
        }
        let shown = cap.png.clone();
        actor.click_png(cap, at.0, at.1)?;
        actor.sleep(CLICK_SETTLE_MS);
        *cap = actor.capture()?;
        if verify::changed(&shown, &cap.png)? {
            now = match read_selected(grounder, cap)? {
                Some(c) => c,
                None if b == 0 => 0,
                None => {
                    return Err(ScreenError::Stalled(format!("the picker's header showed no count after a day's click ({b} were selected before it)")));
                }
            };
        }
        match now.cmp(&b) {
            Ordering::Equal if n == 0 => return Ok(Click::Nothing),
            Ordering::Equal => return Ok(passed.map_or(Click::Had, Click::Over)),
            Ordering::Greater if now <= cap_items => return Ok(Click::Took(now)),
            Ordering::Greater => passed = passed.max(Some(now)),
            Ordering::Less => {}
        }
    }
    Err(ScreenError::Stalled(format!("{DAY_CLICKS} clicks on one day left the picker's count at {now}, not back at {b}")))
}

/// What a walk over the day groups of search `query` (typed as `typed`) came to, on top of the
/// `before` already selected: its outcome, a note, and the searches it leaves for a next session.
/// It stopped at a day that did not fit: split — the days that fit are kept and the search resumes
/// from that day ([`picker::resume_from`]; one search per day would not do: Google's day search
/// leaves out items its month view groups under that day, live) — or, when none of it fit beside
/// what was selected, deferred whole. Otherwise selected (at the cap too: nothing was cut short).
/// Days no click selected, and days too big alone, are left as day searches (a day too big alone
/// is reported so by its own search); a search whose only days are too big is too big.
fn days_outcome(days: &Days, query: &str, typed: &str, before: Option<u32>, cap_items: u32) -> (Outcome, String, Vec<String>) {
    let day_searches = |keys: &[String]| -> Vec<String> {
        keys.iter().filter_map(|k| picker::day_query(k, typed)).filter(|q| !q.eq_ignore_ascii_case(typed) && !q.eq_ignore_ascii_case(query)).collect()
    };
    let b = before.unwrap_or(0);
    let now = days.count.unwrap_or(0);
    let passed = days.passed.unwrap_or(now);
    let took = !(days.picked.is_empty() && days.had.is_empty() && days.over.is_none() && days.too_big.is_empty());
    if !took && days.missed.is_empty() {
        let note = if days.resumed_past > 0 {
            format!("its {} day(s) after its resume point were picked by an earlier session, and none is left", days.resumed_past)
        } else {
            format!("its results show only {} other date(s)' days", days.foreign)
        };
        return (Outcome::NoResults, note, Vec::new());
    }
    let mut rest: Vec<String> = day_searches(&days.missed);
    rest.extend(day_searches(&days.too_big));
    let mut also = String::new();
    if !days.missed.is_empty() {
        also.push_str(&format!("; {} day(s) no click selected are left as day searches", days.missed.len()));
    }
    if !days.too_big.is_empty() {
        also.push_str(&format!("; {} day(s) alone more than the session's {cap_items}-item maximum are left as day searches", days.too_big.len()));
    }
    if let Some(from) = &days.over {
        if !days.picked.is_empty() {
            let note = format!(
                "{} of its days fit under the session's {cap_items}-item maximum ({now} selected); {from} would have taken the count to {passed}, so the search resumes from it in a next session{also}",
                days.picked.len()
            );
            rest.insert(0, picker::resume_from(query, from));
            return (Outcome::Split, note, rest);
        }
        let note = format!(
            "its first day would have taken the count to {passed}, past the session's {cap_items}-item maximum (the picker refuses Done past it); unselected again, the search waits for a next session"
        );
        return (Outcome::Deferred, note, vec![query.to_string()]);
    }
    if days.picked.is_empty() && days.had.is_empty() && days.missed.is_empty() && !days.too_big.is_empty() {
        // a day search too big, or a search whose every day is: no session holds them
        return (Outcome::TooBig, too_big_note(passed, cap_items), Vec::new());
    }
    let (mut outcome, mut note) = outcome_of(before, days.count, cap_items);
    if outcome == Outcome::AtCap {
        (outcome, note) = (Outcome::Selected, String::new()); // every day fit
    }
    if outcome == Outcome::NothingNew && b > 0 && !days.missed.is_empty() {
        note = "no day of it took".into();
    }
    let also = also.trim_start_matches("; ");
    if !also.is_empty() {
        note = if note.is_empty() { also.to_string() } else { format!("{note}; {also}") };
    }
    (outcome, note, rest)
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
        // `moved`, not `changed`: the hovered video tile plays its preview, so the screen
        // never stops changing at the top either
        still = if verify::moved(&before, &cap.png)? { 0 } else { still + 1 };
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

/// One range gesture (see [`select_everything`]). An album's is grounded on every tile, photos
/// too — the album is the whole grid, and its first and last items are its ends whatever they
/// are — and its grid must end within the budget: one that does not is more than an album holds
/// for this run (the library, or Google's ordinary results for a title that is no album's), so
/// it ends the pick without the last click, and Done is never pressed.
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
    let videos_only = plan.videos_only && plan.album.is_none();
    // 1. the first tile, in reading order (photos too, when they count: the candidates list
    // videos first)
    let band = (cap.png_h / 16).max(1);
    let mut first = None;
    let mut first_page = 0usize;
    for _ in 0..MAX_STALLED {
        let tiles = visible_tiles(grounder, cap, videos_only, progress)?;
        first_page = tiles.len();
        if let Some(t) = tiles.into_iter().min_by_key(|e| (e.y1 / band, e.x1)) {
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
            if let Some(title) = plan.album.as_deref() {
                return Err(ScreenError::Stalled(format!(
                    "the grid the search for the album `{title}` showed never ended within the budget: more than an album holds for this run (the whole library?); Done was not pressed, nothing was picked"
                )));
            }
            progress("screenagent: out of budget before the end of the grid; selecting up to here");
            break;
        }
        let before = cap.png.clone();
        actor.scroll(END_SCROLL_LINES)?;
        actor.sleep(1200);
        *cap = actor.capture()?;
        // `moved`, not `changed`: the video tile under the pointer plays its preview, which
        // kept the screen "changing" forever and spent the whole budget scrolling (a 92-item
        // album, 2026-10-19)
        if verify::moved(&before, &cap.png)? {
            still = 0;
            scrolled = true;
        } else {
            still += 1;
        }
    }
    // 3. Shift+click the last tile
    let tiles = visible_tiles(grounder, cap, videos_only, progress)?;
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
/// Day searches an item pick is budgeted for: one per UTC day its items were taken on, half as
/// many again for items found on the day next to it, at most [`MAX_ITEM_SEARCHES`].
fn item_searches(items: &[crate::items::Wanted]) -> usize {
    let days: std::collections::BTreeSet<chrono::NaiveDate> = items.iter().map(|w| w.taken.date_naive()).collect();
    days.len().saturating_add(days.len() / 2).clamp(1, MAX_ITEM_SEARCHES)
}

/// The most day searches one item pick makes (the rest of its items are [`Summary::left`]).
pub const MAX_ITEM_SEARCHES: usize = 30;
/// Actions and time per day search of an item pick, and the time it starts with.
const ITEM_ACTIONS_PER_SEARCH: u32 = 4;
const ITEM_TIME_PER_SEARCH: Duration = Duration::from_secs(75);
const ITEM_BASE_TIME: Duration = Duration::from_secs(120);
/// Seconds the page is watched for a day search's results.
const ITEM_PAGE_LOOKS: u32 = 20;

const ITEMS_NEED_PAGE: &str = "picking exact items needs the picker's page — Chrome with View ▸ Developer ▸ Allow JavaScript from Apple Events ticked — to read when each tile was taken; nothing was selected";

/// The Mac's offset from UTC at `t`, in minutes: each item's first guess at its zone.
fn home_offset(t: chrono::DateTime<chrono::Utc>) -> i32 {
    use chrono::{Offset, TimeZone};
    chrono::Local.offset_from_utc_datetime(&t.naive_utc()).fix().local_minus_utc() / 60
}

/// Search each wanted item's day and tick its tile in the page ([`crate::items::Matcher`] says
/// which tile is which item), until every item has a tile, has had its days, or the budget ends.
/// Items with no tile are [`Summary::left`] (their RFC 3339 times).
#[allow(clippy::too_many_arguments)]
fn pick_items<A: Act, G: Ground>(
    actor: &mut A,
    grounder: &G,
    plan: &Plan,
    summary: &mut Summary,
    cap: &mut crate::capture::Capture,
    page: &Page,
    started: Instant,
    progress: &mut dyn FnMut(&str),
) -> Result<(), ScreenError> {
    use crate::dom::{self, Results, Step};
    let out_of_time = || started.elapsed() >= plan.timeout;
    let mut m = crate::items::Matcher::new(plan.items.clone(), &home_offset);
    let Some(field) = find_search_field(actor, grounder, cap, progress)? else {
        return Err(ScreenError::Other("the picker's search field never appeared, so no day could be searched; nothing was selected".into()));
    };
    let Some(mut shown) = dom::settled_tiles(actor, &page.tab)? else {
        return Err(ScreenError::Other("the picker's page shows no settled grid; nothing was selected".into()));
    };
    progress(&format!("screenagent: picking {} item(s) by the days they were taken", m.len()));
    let mut typed: Option<usize> = None;
    let mut searches = 0usize;
    while let Some(day) = m.next_day() {
        let left = plan.timeout.saturating_sub(started.elapsed());
        if searches >= MAX_ITEM_SEARCHES || left < SEARCH_MIN_TIME.saturating_add(DONE_RESERVE_TIME) || spend(summary, plan)? {
            progress("screenagent: out of budget for more day searches; the items not found are left for another pick");
            break;
        }
        let text = crate::items::day_text(day);
        type_search(actor, cap, &field, &text, typed, progress)?;
        typed = Some(text.chars().count());
        summary.searched = true;
        searches += 1;
        let before = m.found();
        let (outcome, note) = match dom::wait_for_results(actor, &page.tab, &shown, ITEM_PAGE_LOOKS)? {
            Results::Shown(_) => {
                let labels = dom::grid_labels(actor, &page.tab, &out_of_time)?;
                let tick = m.choose(day, &labels);
                if !tick.is_empty() {
                    let ticked = dom::tick_labels(actor, &page.tab, &tick, &out_of_time)?;
                    for l in tick.iter().filter(|l| !ticked.contains(*l)) {
                        progress(&format!("screenagent: the tile `{l}` would not tick"));
                        m.untick(l);
                    }
                }
                let got = m.found().saturating_sub(before);
                (if got > 0 { Outcome::Selected } else { Outcome::NothingNew }, format!("{got} of its {} tile(s) ticked", labels.len()))
            }
            Results::Empty => {
                m.choose(day, &[]);
                (Outcome::NoResults, String::new())
            }
            Results::Unchanged => {
                m.choose(day, &[]);
                (Outcome::Failed, format!("the grid did not change in the {ITEM_PAGE_LOOKS} s after Return"))
            }
        };
        // the next search's results are told from the grid as it is now, at its top
        let top = dom::run(actor, &page.tab, Step::Top)?;
        shown = top.labels();
        let selected = top.selected();
        progress(&format!("screenagent: `{text}`: {}", outcome_line(outcome, selected, &note)));
        summary.searches.push(Searched { query: text, outcome, selected, note });
    }
    let look = dom::run(actor, &page.tab, Step::Look)?;
    let found = m.found();
    summary.selected = look.selected().unwrap_or(u32::try_from(found).unwrap_or(u32::MAX));
    summary.left = m.left();
    let mut note = format!("ticked {found} of {} item(s) over {searches} day search(es)", m.len());
    if !m.ambiguous.is_empty() {
        note.push_str(&format!("; {} had two tiles that fitted equally and were left", m.ambiguous.len()));
    }
    if found < m.len() {
        note.push_str(&format!("; {} not found (in `left`)", m.len().saturating_sub(found)));
    }
    summary.note = note;
    *cap = actor.capture()?;
    Ok(())
}

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

    /// Where [`Hands`] finds the search field.
    const FIELD: (u32, u32) = (100, 4);

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
        let p = plan_for(Request { items: Vec::new(), queries: vec!["beach".into()], max_items: 4, select_all: true, album: None, videos_only: true });
        assert_eq!((p.queries.as_slice(), p.max_items, p.select_all, p.videos_only, p.album), (["beach".to_string()].as_slice(), 4, true, true, None));
        let p = plan_for(Request { items: Vec::new(), queries: vec![], max_items: 0, select_all: false, album: Some("Robots".into()), videos_only: false });
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
    /// the last search is `empty`; the header's counts in turn. (Date groups: [`Dated`].)
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

    impl Ground for Eyes {
        fn inventory(&self, _png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
            self.asks.borrow_mut().push(task.to_string());
            let w = self.world.borrow();
            let last = w.typed.last().cloned().unwrap_or_default();
            if task == crate::picker::SELECTED_COUNTER {
                let label = self.counts.borrow_mut().pop_front().unwrap_or("");
                return Ok(if label.is_empty() { vec![] } else { vec![Element { label: label.into(), kind: String::new(), x1: 5, y1: 2, x2: 60, y2: 12 }] });
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
        plan_for(Request {
            items: Vec::new(),
            queries: queries.iter().map(|q| q.to_string()).collect(),
            max_items,
            select_all: true,
            album: None,
            videos_only: true,
        })
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
    fn a_non_date_search_never_asks_for_date_groups() {
        let w = world();
        let eyes = Eyes::new(&w, &["5 selected"]);
        drive(&mut Hands(Rc::clone(&w)), &eyes, &range_plan(), &mut |_| {}).unwrap();
        assert!(!eyes.asks.borrow().iter().any(|a| a == crate::picker::DAY_HEADINGS));
        assert_eq!(w.borrow().shift_clicks.len(), 1, "the range gesture");
    }

    #[test]
    fn a_range_past_the_cap_is_never_confirmed() {
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
    fn many_queries_get_a_budget_that_grows_with_them() {
        let qs = |n: usize| (0..n).map(|i| format!("q{i}")).collect::<Vec<_>>();
        let p = plan_for(Request { items: Vec::new(), queries: qs(5), max_items: MAX_PICK, select_all: true, album: None, videos_only: true });
        assert_eq!((p.max_actions, p.timeout), (ACTIONS_PER_SEARCH * 5 + DONE_RESERVE_ACTIONS, Duration::from_secs(300 + 4 * 180)));
        let p = plan_for(Request { items: Vec::new(), queries: qs(500), max_items: MAX_PICK, select_all: true, album: None, videos_only: true });
        assert_eq!(p.timeout, MAX_SESSION_TIME, "bounded");
        // one query keeps the single-search budget
        let p = plan_for(Request { items: Vec::new(), queries: qs(1), max_items: 6, select_all: false, album: None, videos_only: true });
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
        let plan = plan_for(Request {
            items: Vec::new(),
            queries: vec!["a".into(), "b".into(), "c".into()],
            max_items: 4,
            select_all: false,
            album: None,
            videos_only: true,
        });
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &plan, &mut |_| {}).unwrap();
        assert_eq!(s.selected, 4, "{s:?}");
        assert_eq!(s.left, ["c"], "{s:?}");
        assert_eq!(s.searches.iter().map(|x| x.outcome).collect::<Vec<_>>(), [Outcome::Selected, Outcome::Selected]);
        assert!(s.done_pressed);
    }

    /// Days on a page of [`Dated`]'s results (the last of one page is the first of the next).
    const ROWS: usize = 3;
    /// A [`Dated`] tile's side, and the most tiles of a day on its row.
    const TILE: u32 = 24;
    const ROW_TILES: u32 = 6;

    /// A picker whose search results come in date groups, for the date-group tests: the days a
    /// search finds, newest first, [`ROWS`] to a page — the last of one page is the first of the
    /// next, a heading still on screen after a scroll — each a heading over a row of its tiles, one
    /// per item. A day of one item has no check button (live, in compact rows: not even on hover).
    /// Clicking a check selects its day, or unselects it when all of it is selected (live); a tile
    /// toggles its item; a Shift+click selects from the last tile clicked. Every click changes the
    /// pixels (the pointer hovers). The header counts the selection, which lasts across searches;
    /// Done takes up to the session's cap and refuses past it (live).
    #[derive(Default)]
    struct Dated {
        /// (heading, items) of the library's days.
        days: Vec<(&'static str, u32)>,
        /// The days each search finds (indices into `days`).
        finds: Vec<(&'static str, Vec<usize>)>,
        cap: u32,
        shown: Vec<usize>,
        page: usize,
        selected: std::collections::BTreeSet<(usize, u32)>,
        anchor: Option<(usize, u32)>,
        tag: u8,
        typed: Vec<String>,
        shift_clicks: u32,
        done: bool,
        dones: u32,
        url: Option<String>,
        tabs_closed: u32,
        /// Headings the eyes box from their date's text (the check button left of the box).
        text_start: Vec<&'static str>,
        /// Headings the eyes box too far right, off the check button and its spot beside the box.
        off_check: Vec<&'static str>,
        /// (heading, other heading): the first is boxed on the second's row (when it is in view).
        misplaced: Vec<(&'static str, &'static str)>,
        /// (heading, sighting from 0) the eyes do not box.
        blind: Vec<(&'static str, usize)>,
        sightings: std::collections::HashMap<&'static str, usize>,
        asks: Vec<String>,
    }

    impl Dated {
        fn new(days: &[(&'static str, u32)], finds: &[(&'static str, &[usize])], cap: u32) -> Rc<RefCell<Self>> {
            Rc::new(RefCell::new(Self {
                days: days.to_vec(),
                finds: finds.iter().map(|(q, d)| (*q, d.to_vec())).collect(),
                cap,
                url: Some(PICKER_TAB.into()),
                ..Self::default()
            }))
        }
        /// The days in view, with the top of their rows.
        fn rows(&self) -> Vec<(usize, u32)> {
            self.shown.iter().copied().skip(self.page * (ROWS - 1)).take(ROWS).zip([20u32, 60, 100]).collect()
        }
        fn last_page(&self) -> usize {
            self.shown.len().saturating_sub(2) / (ROWS - 1)
        }
        fn items(&self, d: usize) -> u32 {
            self.days.get(d).map_or(0, |x| x.1)
        }
        /// How many of day `d`'s items are selected.
        fn of_day(&self, d: usize) -> usize {
            self.selected.iter().filter(|(x, _)| *x == d).count()
        }
        fn click(&mut self, px: u32, py: u32, shift: bool) {
            self.tag = self.tag.wrapping_add(10); // the pointer hovers: the pixels change whatever the click does
            if (px, py) == DONE_AT {
                self.dones += 1;
                let n = u32::try_from(self.selected.len()).unwrap();
                self.done = (1..=self.cap).contains(&n);
                return;
            }
            for (d, y) in self.rows() {
                let items = self.items(d);
                if items > 1 && (4..14).contains(&px) && (y..y + 10).contains(&py) {
                    let all = (0..items).all(|i| self.selected.contains(&(d, i)));
                    for i in 0..items {
                        if all {
                            self.selected.remove(&(d, i));
                        } else {
                            self.selected.insert((d, i));
                        }
                    }
                    return;
                }
                for i in 0..items.min(ROW_TILES) {
                    let x = 20 + 26 * i;
                    if !((x..x + TILE).contains(&px) && (y + 12..y + 12 + TILE).contains(&py)) {
                        continue;
                    }
                    match (shift, self.anchor) {
                        (true, Some(a)) => {
                            let order: Vec<(usize, u32)> = self.shown.iter().flat_map(|&d| (0..self.items(d)).map(move |i| (d, i))).collect();
                            let (p, q) = (order.iter().position(|o| *o == a).unwrap(), order.iter().position(|o| *o == (d, i)).unwrap());
                            self.selected.extend(order[p.min(q)..=p.max(q)].iter().copied());
                        }
                        _ => {
                            if !self.selected.remove(&(d, i)) {
                                self.selected.insert((d, i));
                            }
                            self.anchor = Some((d, i));
                        }
                    }
                    return;
                }
            }
        }
    }

    struct DatedHands(Rc<RefCell<Dated>>);

    impl Act for DatedHands {
        fn activate(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn click_png(&mut self, _cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            self.0.borrow_mut().click(px, py, false);
            Ok(())
        }
        fn shift_click_png(&mut self, _cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            let mut w = self.0.borrow_mut();
            w.shift_clicks += 1;
            w.click(px, py, true);
            Ok(())
        }
        fn type_text(&mut self, text: &str) -> Result<(), ScreenError> {
            let mut w = self.0.borrow_mut();
            w.typed.push(text.to_string());
            w.shown = w.finds.iter().find(|(q, _)| *q == text).map(|(_, d)| d.clone()).unwrap_or_default();
            (w.page, w.anchor) = (0, None);
            w.tag = w.tag.wrapping_add(10);
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
            let page = if n > 0 { (w.page + 1).min(w.last_page()) } else { 0 };
            if page != w.page {
                w.page = page;
                w.tag = w.tag.wrapping_add(10);
            }
            Ok(())
        }
        fn sleep(&mut self, _ms: u64) {}
        fn capture(&mut self) -> Result<Capture, ScreenError> {
            Ok(cap_of(self.0.borrow().tag))
        }
        fn clear_text(&mut self, _chars: usize) -> Result<(), ScreenError> {
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

    /// Eyes on [`Dated`] that box date headings as qwen2.5vl does: from the check button, or from
    /// the date's text (always, for a day of one item: it has no check button), on the heading's row.
    struct DatedEyes(Rc<RefCell<Dated>>);

    impl Ground for DatedEyes {
        fn inventory(&self, _png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
            let mut w = self.0.borrow_mut();
            w.asks.push(task.to_string());
            let el = |label: &str, x1: u32, y1: u32, x2: u32, y2: u32| Element { label: label.into(), kind: String::new(), x1, y1, x2, y2 };
            if task == crate::picker::SELECTED_COUNTER {
                let n = w.selected.len();
                return Ok(if n == 0 || w.done { vec![] } else { vec![el(&format!("{n} selected"), 5, 2, 60, 12)] });
            }
            let mut out = Vec::new();
            let rows = w.rows();
            for &(d, y) in &rows {
                let (label, items) = w.days[d];
                if task == crate::picker::DAY_HEADINGS {
                    let seen = w.sightings.entry(label).or_default();
                    let nth = *seen;
                    *seen += 1;
                    if w.blind.contains(&(label, nth)) {
                        continue;
                    }
                    let y = match w.misplaced.iter().find(|(l, _)| *l == label) {
                        Some((_, other)) => match rows.iter().find(|(o, _)| w.days[*o].0 == *other) {
                            Some(&(_, oy)) => oy,
                            None => continue,
                        },
                        None => y,
                    };
                    let x1 = if w.off_check.contains(&label) {
                        40
                    } else if items == 1 || w.text_start.contains(&label) {
                        20
                    } else {
                        4
                    };
                    out.push(el(label, x1, y + 1, x1 + 44, y + 9));
                } else {
                    for i in 0..items.min(ROW_TILES) {
                        let x = 20 + 26 * i;
                        out.push(el(&format!("0:{:02}", out.len() + 1), x, y + 12, x + TILE, y + 12 + TILE));
                    }
                }
            }
            Ok(out)
        }
        fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            if target == crate::picker::SEARCH_FIELD {
                return Ok(Some(Element { label: "Search".into(), kind: String::new(), x1: 40, y1: 0, x2: 160, y2: 8 }));
            }
            let done = self.0.borrow().done;
            Ok((target.contains("Done") && !done).then(|| Element { label: "Done".into(), kind: String::new(), x1: 180, y1: 0, x2: 199, y2: 20 }))
        }
    }

    fn drive_dated(w: &Rc<RefCell<Dated>>, plan: &Plan) -> (Result<Summary, ScreenError>, Vec<String>) {
        let mut lines: Vec<String> = vec![];
        let s = drive(&mut DatedHands(Rc::clone(w)), &DatedEyes(Rc::clone(w)), plan, &mut |l| lines.push(l.into()));
        (s, lines)
    }

    fn one_search(q: &str, max_items: u32) -> Plan {
        Plan { select_all: true, queries: vec![q.into()], max_items, ..Plan::default() }
    }

    #[test]
    fn a_day_of_one_item_is_selected_by_its_tile() {
        // live, Chrome 155: in a search result's compact rows a day of one item shows no check
        // button, not even on hover. The click on its heading selected nothing, yet the hover
        // changed the pixels, so the day was counted done and its item never picked.
        let w = Dated::new(&[("Fri, Oct 14, 2022", 3), ("Thu, Oct 13, 2022", 1), ("Wed, Oct 12, 2022", 2)], &[("videos October 2022", &[0, 1, 2])], 2000);
        let (s, lines) = drive_dated(&w, &one_search("videos October 2022", MAX_PICK));
        let s = s.unwrap();
        let w = w.borrow();
        assert_eq!(w.selected.len(), 6, "every item of the month: {lines:#?}");
        assert!(w.done && s.done_pressed && s.selected == 6, "{s:?}");
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(6))]);
        assert!(lines.iter().any(|l| l.contains("selected the day `Thu, Oct 13, 2022` by its one tile (4 selected)")), "{lines:#?}");
    }

    #[test]
    fn date_groups_fill_the_session_and_leave_the_days_that_did_not_fit() {
        // live: a 40-item session's month took the count past the cap; the walk that clicked its
        // days again to unselect them found 12 of 13, so 79 stayed selected, Done was refused and
        // the session lost. Here the walk back misses Nov 29 the same way.
        let w = Dated::new(
            &[
                ("Mon, Oct 31, 2022", 4),
                ("Wed, Nov 30, 2022", 6),
                ("Tue, Nov 29, 2022", 9),
                ("Mon, Nov 28, 2022", 1),
                ("Sun, Nov 27, 2022", 4),
                ("Sat, Nov 26, 2022", 7),
                ("Fri, Nov 25, 2022", 5),
            ],
            &[("videos October 2022", &[0]), ("videos November 2022", &[1, 2, 3, 4, 5, 6])],
            25,
        );
        w.borrow_mut().blind = vec![("Tue, Nov 29, 2022", 1)];
        let (s, lines) = drive_dated(&w, &many(&["videos October 2022", "videos November 2022"], 25));
        let s = s.unwrap();
        let w = w.borrow();
        // October's 4, then November's days while they fit: 6 + 9 + 1 + 4 = 24 of 25; Nov 26's 7
        // would make 31
        assert_eq!(w.selected.len(), 24, "{lines:#?}");
        assert!(w.done && w.dones == 1 && s.done_pressed && s.selected == 24, "{s:?}");
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(4)), (Outcome::Split, Some(24))]);
        // the month resumes from Nov 26 in the next session, which skips the days picked here
        assert_eq!(s.left, ["videos November 2022 from 2022-11-26"], "{s:?}");
        assert!(s.retry().is_empty() && s.too_big().is_empty(), "{s:?}");
        assert!(lines.iter().any(|l| l.contains("selected the day `Mon, Nov 28, 2022` by its one tile")), "{lines:#?}");
    }

    #[test]
    fn a_day_selected_already_is_clicked_back_on() {
        // the model never answers that a day's check is filled in, so a day already selected (an
        // earlier search of the session had it) was clicked again — which unselects it
        let w = Dated::new(
            &[("Fri, Oct 14, 2022", 2), ("Thu, Oct 13, 2022", 3), ("Wed, Oct 12, 2022", 2)],
            &[("videos October 13 2022", &[1]), ("videos October 2022", &[0, 1, 2])],
            2000,
        );
        let (s, lines) = drive_dated(&w, &many(&["videos October 13 2022", "videos October 2022"], MAX_PICK));
        let s = s.unwrap();
        let w = w.borrow();
        assert_eq!(w.selected.len(), 7, "{lines:#?}");
        assert!(w.done && s.selected == 7, "{s:?}");
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(3)), (Outcome::Selected, Some(7))]);
        assert!(lines.iter().any(|l| l.contains("`Thu, Oct 13, 2022` was selected already")), "{lines:#?}");
    }

    #[test]
    fn each_day_is_clicked_once_however_the_pages_overlap() {
        // pages of three, each starting with the last day of the page before
        let days = [("Fri, Oct 14, 2022", 2), ("Thu, Oct 13, 2022", 2), ("Wed, Oct 12, 2022", 2), ("Tue, Oct 11, 2022", 2), ("Mon, Oct 10, 2022", 2)];
        let w = Dated::new(&days, &[("videos October 2022", &[0, 1, 2, 3, 4])], 2000);
        let (s, lines) = drive_dated(&w, &one_search("videos October 2022", MAX_PICK));
        let s = s.unwrap();
        assert_eq!((w.borrow().selected.len(), s.selected), (10, 10), "{lines:#?}");
        assert_eq!(lines.iter().filter(|l| l.contains("selected the day")).count(), 5, "{lines:#?}");
        assert!(!lines.iter().any(|l| l.contains("selected already")), "no day clicked twice: {lines:#?}");
        assert_eq!(w.borrow().shift_clicks, 0, "no range gesture");
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(10))]);
        assert!(s.left.is_empty() && s.retry().is_empty() && s.done_pressed, "{s:?}");
    }

    #[test]
    fn a_split_search_ends_the_session_and_its_days_come_first_in_the_next() {
        let w = Dated::new(
            &[("Wed, Nov 30, 2022", 6), ("Tue, Nov 29, 2022", 9), ("Mon, Oct 31, 2022", 1)],
            &[("videos November 2022", &[0, 1]), ("videos October 2022", &[2])],
            10,
        );
        let (s, lines) = drive_dated(&w, &many(&["videos November 2022", "videos October 2022", "videos September 2022"], 10));
        let s = s.unwrap();
        assert_eq!(outcomes(&s), [(Outcome::Split, Some(6))], "{lines:#?}");
        // October's one item would still fit, but the session is full where the split happened
        assert_eq!(s.left, ["videos November 2022 from 2022-11-29", "videos October 2022", "videos September 2022"]);
        assert!(s.done_pressed && w.borrow().done && s.selected == 6, "{s:?}");
        assert!(s.note.contains("1 split (to resume in a next session)"), "{}", s.note);
    }

    #[test]
    fn a_search_that_does_not_fit_beside_the_others_waits_whole() {
        let w = Dated::new(
            &[("Mon, Oct 31, 2022", 4), ("Wed, Nov 30, 2022", 25), ("Tue, Nov 29, 2022", 2), ("Thu, Dec 1, 2022", 3)],
            &[("videos October 2022", &[0]), ("videos November 2022", &[1, 2]), ("videos December 2022", &[3])],
            25,
        );
        let (s, lines) = drive_dated(&w, &many(&["videos October 2022", "videos November 2022", "videos December 2022"], 25));
        let s = s.unwrap();
        // November's first day alone would make 29: unselected at once, the walk stops there, and
        // so does the session — the next session has room for November
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(4)), (Outcome::Deferred, Some(4))], "{lines:#?}");
        assert_eq!(s.left, ["videos November 2022", "videos December 2022"]);
        assert!(s.retry().is_empty() && s.too_big().is_empty(), "{s:?}");
        let w = w.borrow();
        assert_eq!((w.of_day(1), w.of_day(2), w.selected.len()), (0, 0, 4), "nothing of November stays selected");
        assert!(w.done && s.selected == 4 && w.dones == 1, "{s:?}");
        assert!(s.note.contains("1 past the session's maximum"), "{}", s.note);
    }

    #[test]
    fn date_searches_fill_a_session_past_the_biggest_search_guess() {
        // after a 20-item day the next search could pass 25 — a guess a range needs (it cannot be
        // undone), not date groups: they are taken day by day until one does not fit
        let days = [("Sat, Oct 29, 2022", 20), ("Fri, Oct 28, 2022", 3), ("Thu, Oct 27, 2022", 10), ("Wed, Oct 26, 2022", 1)];
        let finds: [(&str, &[usize]); 4] =
            [("videos October 29 2022", &[0]), ("videos October 28 2022", &[1]), ("videos October 27 2022", &[2]), ("videos October 26 2022", &[3])];
        let w = Dated::new(&days, &finds, 25);
        let plan = many(&["videos October 29 2022", "videos October 28 2022", "videos October 27 2022", "videos October 26 2022"], 25);
        let (s, lines) = drive_dated(&w, &plan);
        let s = s.unwrap();
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(20)), (Outcome::Selected, Some(23)), (Outcome::Deferred, Some(23))], "{lines:#?}");
        assert_eq!(s.left, ["videos October 27 2022", "videos October 26 2022"]);
        assert!(s.done_pressed && w.borrow().done && s.selected == 23 && w.borrow().selected.len() == 23, "{s:?}");
        // a date search whose days take nothing falls back to a range: that one still waits when
        // the range could pass the cap
        let w = Dated::new(&days, &finds, 25);
        w.borrow_mut().off_check = vec!["Fri, Oct 28, 2022"];
        let (s, lines) = drive_dated(&w, &many(&["videos October 29 2022", "videos October 28 2022"], 25));
        let s = s.unwrap();
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(20)), (Outcome::Deferred, Some(20))], "{lines:#?}");
        assert_eq!(s.left, ["videos October 28 2022"]);
        assert_eq!((w.borrow().of_day(1), w.borrow().shift_clicks, s.selected), (0, 0, 20), "no range");
        assert!(s.searches[1].note.contains("a range cannot be undone"), "{:?}", s.searches[1]);
    }

    #[test]
    fn a_day_alone_past_the_cap_is_skipped_and_the_walk_goes_on() {
        let days = [("Fri, Oct 14, 2022", 30), ("Thu, Oct 13, 2022", 5), ("Wed, Oct 12, 2022", 1)];
        let finds: [(&str, &[usize]); 2] = [("videos October 2022", &[0, 1, 2]), ("videos October 14 2022", &[0])];
        // a month: its 30-item day fits no session; the others are picked, and that day is left as
        // its own search, which reports it too big
        let w = Dated::new(&days, &finds, 25);
        let (s, lines) = drive_dated(&w, &one_search("videos October 2022", 25));
        let s = s.unwrap();
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(6))], "{lines:#?}");
        assert_eq!(s.left, ["videos October 14 2022"]);
        assert_eq!((w.borrow().of_day(0), w.borrow().selected.len()), (0, 6));
        assert!(s.done_pressed && w.borrow().done, "{s:?}");
        // that day alone: no session holds it
        let w = Dated::new(&days, &finds, 25);
        let (s, _) = drive_dated(&w, &one_search("videos October 14 2022", 25));
        let s = s.unwrap();
        assert_eq!(outcomes(&s), [(Outcome::TooBig, Some(0))]);
        assert!(s.left.is_empty() && !s.done_pressed && w.borrow().selected.is_empty() && s.stop_waiting(Duration::ZERO), "{s:?}");
        assert!(s.note.contains("more than the session's 25-item maximum"), "{}", s.note);
        // ...and in a session of many, the next search goes on
        let w = Dated::new(&[("Fri, Oct 14, 2022", 30), ("Thu, Oct 13, 2022", 5)], &[("videos October 14 2022", &[0]), ("videos October 13 2022", &[1])], 25);
        let (s, _) = drive_dated(&w, &many(&["videos October 14 2022", "videos October 13 2022"], 25));
        let s = s.unwrap();
        assert_eq!(outcomes(&s), [(Outcome::TooBig, None), (Outcome::Selected, Some(5))]);
        assert_eq!(s.too_big(), ["videos October 14 2022"]);
        assert!(s.left.is_empty() && s.retry().is_empty() && s.done_pressed && s.selected == 5, "{s:?}");
    }

    #[test]
    fn a_resumed_search_skips_the_days_an_earlier_session_picked() {
        let days = [("Wed, Nov 30, 2022", 6), ("Tue, Nov 29, 2022", 9), ("Sat, Nov 26, 2022", 7), ("Fri, Nov 25, 2022", 5)];
        let w = Dated::new(&days, &[("videos November 2022", &[0, 1, 2, 3])], 40);
        let (s, lines) = drive_dated(&w, &one_search("videos November 2022 from 2022-11-26", 40));
        let s = s.unwrap();
        assert_eq!(w.borrow().typed, ["videos November 2022"], "the resume point is not typed");
        assert_eq!((w.borrow().of_day(0), w.borrow().of_day(1), w.borrow().selected.len()), (0, 0, 12), "{lines:#?}");
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(12))]);
        assert!(s.left.is_empty() && s.done_pressed, "{s:?}");
        // resumed again where it stops again
        let w = Dated::new(&days, &[("videos November 2022", &[0, 1, 2, 3])], 10);
        let (s, _) = drive_dated(&w, &one_search("videos November 2022 from 2022-11-26", 10));
        let s = s.unwrap();
        assert_eq!(outcomes(&s), [(Outcome::Split, Some(7))]);
        assert_eq!(s.left, ["videos November 2022 from 2022-11-25"]);
        // nothing left after the resume point: no results, nothing to confirm
        let w = Dated::new(&days[..2], &[("videos November 2022", &[0, 1])], 40);
        let (s, _) = drive_dated(&w, &one_search("videos November 2022 from 2022-11-26", 40));
        let s = s.unwrap();
        assert_eq!(outcomes(&s), [(Outcome::NoResults, Some(0))]);
        assert!(!s.done_pressed && w.borrow().selected.is_empty() && w.borrow().shift_clicks == 0, "{s:?}");
    }

    #[test]
    fn a_heading_boxed_from_its_text_is_selected_by_the_button_left_of_it() {
        let w = Dated::new(&[("Mon, Oct 24, 2022", 7), ("Sat, Oct 22, 2022", 8)], &[("videos October 2022", &[0, 1])], 2000);
        w.borrow_mut().text_start = vec!["Mon, Oct 24, 2022", "Sat, Oct 22, 2022"];
        let (s, lines) = drive_dated(&w, &one_search("videos October 2022", MAX_PICK));
        let s = s.unwrap();
        assert_eq!((w.borrow().selected.len(), s.selected), (15, 15), "{lines:#?}");
        assert!(s.left.is_empty() && s.done_pressed, "{s:?}");
    }

    #[test]
    fn a_click_that_hits_another_selected_day_is_put_back_and_its_day_left() {
        // live: the box labelled Oct 18 sat on a row of Oct 22, which was selected: the click
        // unselected Oct 22 and was taken for Oct 18 being selected already
        let w = Dated::new(&[("Mon, Oct 24, 2022", 2), ("Sat, Oct 22, 2022", 3), ("Tue, Oct 18, 2022", 2)], &[("videos October 2022", &[0, 1, 2])], 2000);
        w.borrow_mut().misplaced = vec![("Tue, Oct 18, 2022", "Sat, Oct 22, 2022")];
        let (s, lines) = drive_dated(&w, &one_search("videos October 2022", MAX_PICK));
        let s = s.unwrap();
        assert_eq!((w.borrow().of_day(1), w.borrow().of_day(2), s.selected), (3, 0, 5), "Oct 22 put back: {lines:#?}");
        assert_eq!(s.left, ["videos October 18 2022"]);
        assert!(lines.iter().any(|l| l.contains("hit another day selected already")), "{lines:#?}");
    }

    #[test]
    fn a_day_whose_check_was_missed_is_never_picked_in_part() {
        // the model boxed Oct 14's date text instead of its check button: the click selects
        // nothing, and one of its three tiles would pass for the whole day — so it is left as a
        // day search instead
        let w = Dated::new(&[("Fri, Oct 14, 2022", 3), ("Thu, Oct 13, 2022", 2)], &[("videos October 2022", &[0, 1])], 2000);
        w.borrow_mut().off_check = vec!["Fri, Oct 14, 2022"];
        let (s, lines) = drive_dated(&w, &one_search("videos October 2022", MAX_PICK));
        let s = s.unwrap();
        assert_eq!(w.borrow().of_day(0), 0, "none of Oct 14 (never a part of it): {lines:#?}");
        assert_eq!((w.borrow().of_day(1), s.selected), (2, 2));
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(2))]);
        assert_eq!(s.left, ["videos October 14 2022"]);
        assert!(s.searches[0].note.contains("1 day(s) no click selected"), "{:?}", s.searches[0]);
    }

    #[test]
    fn only_the_searched_dates_days_are_selected() {
        // Google's search for a month can show a neighbouring day: another search's to pick
        let w = Dated::new(&[("Sat, Oct 1, 2022", 2), ("Fri, Sep 30, 2022", 3)], &[("videos October 2022", &[0, 1])], 2000);
        let (s, lines) = drive_dated(&w, &one_search("videos October 2022", MAX_PICK));
        let s = s.unwrap();
        assert_eq!((w.borrow().of_day(0), w.borrow().of_day(1), s.selected), (2, 0, 2), "{lines:#?}");
        assert_eq!(lines.iter().filter(|l| l.contains("`Fri, Sep 30, 2022` is not a day of")).count(), 1, "{lines:#?}");
    }

    #[test]
    fn a_date_search_that_shows_only_other_dates_has_no_results() {
        // live: `videos October 1 2022` showed Oct 26 (picked by its own search) and the range
        // fallback ran over it
        let w = Dated::new(&[("Thu, Oct 27, 2022", 2), ("Wed, Oct 26, 2022", 4)], &[("videos October 27 2022", &[0]), ("videos October 1 2022", &[1])], 2000);
        let (s, lines) = drive_dated(&w, &many(&["videos October 27 2022", "videos October 1 2022"], MAX_PICK));
        let s = s.unwrap();
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(2)), (Outcome::NoResults, Some(2))], "{lines:#?}");
        assert_eq!((w.borrow().of_day(1), w.borrow().shift_clicks), (0, 0), "no range over Oct 26");
        assert!(s.retry().is_empty() && s.left.is_empty() && s.done_pressed, "{s:?}");
        // alone: nothing selected, nothing to confirm
        let w = Dated::new(&[("Wed, Oct 26, 2022", 4)], &[("videos October 1 2022", &[0])], 2000);
        let (s, _) = drive_dated(&w, &one_search("videos October 1 2022", MAX_PICK));
        let s = s.unwrap();
        assert_eq!(outcomes(&s), [(Outcome::NoResults, Some(0))]);
        assert!(!s.done_pressed && w.borrow().selected.is_empty(), "{s:?}");
    }

    #[test]
    fn a_date_search_without_headings_is_looked_at_twice_before_the_range() {
        let w = world();
        let eyes = Eyes::new(&w, &["3 selected"]);
        let s = drive(&mut Hands(Rc::clone(&w)), &eyes, &one_search("videos October 18 2022", MAX_PICK), &mut |_| {}).unwrap();
        assert_eq!(eyes.asks.borrow().iter().filter(|a| *a == crate::picker::DAY_HEADINGS).count(), 2);
        assert_eq!((w.borrow().shift_clicks.len(), s.selected), (1, 3), "then the range: {s:?}");
    }

    #[test]
    fn a_month_that_fills_the_session_exactly_is_whole() {
        let w = Dated::new(&[("Fri, Oct 14, 2022", 3), ("Thu, Oct 13, 2022", 2)], &[("videos October 2022", &[0, 1]), ("videos November 2022", &[])], 5);
        let (s, _) = drive_dated(&w, &many(&["videos October 2022", "videos November 2022"], 5));
        let s = s.unwrap();
        // every day fit: nothing was cut short, so nothing is worth another try
        assert_eq!(outcomes(&s), [(Outcome::Selected, Some(5))]);
        assert!(s.retry().is_empty(), "{s:?}");
        assert_eq!(s.left, ["videos November 2022"], "the session holds its maximum");
        assert!(s.done_pressed && w.borrow().done);
    }

    #[test]
    fn date_groups_that_select_nothing_fall_back_to_the_range() {
        // every check boxed off its button, every day more than one tile: nothing took, so the
        // range gesture selects the result
        let w = Dated::new(&[("Fri, Oct 14, 2022", 3), ("Thu, Oct 13, 2022", 2)], &[("videos October 2022", &[0, 1])], 2000);
        w.borrow_mut().off_check = vec!["Fri, Oct 14, 2022", "Thu, Oct 13, 2022"];
        let (s, lines) = drive_dated(&w, &one_search("videos October 2022", MAX_PICK));
        let s = s.unwrap();
        assert_eq!((w.borrow().selected.len(), w.borrow().shift_clicks, s.selected), (5, 1, 5), "{lines:#?}");
        assert!(s.left.is_empty() && s.done_pressed, "{s:?}");
        assert!(lines.iter().any(|l| l.contains("selecting the range instead")), "{lines:#?}");
    }

    /// The search field of a [`Shelf`]: tall enough that the typed title, boxed whole, passes for
    /// a tile.
    fn shelf_field() -> Element {
        Element { label: "Search".into(), kind: "field".into(), x1: 30, y1: 0, x2: 170, y2: 24 }
    }
    /// Where a [`Shelf`]'s search field is clicked.
    const SHELF_FIELD_AT: (u32, u32) = (100, 12);
    /// The tops of a [`Shelf`] screen's rows, and its tiles per row and per screen.
    const SHELF_ROWS: [u32; 3] = [30, 70, 110];
    const SHELF_ROW_TILES: usize = 6;
    const SHELF_PAGE: usize = SHELF_ROWS.len() * SHELF_ROW_TILES;

    /// What Return does in a [`Shelf`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum OnReturn {
        /// It searches (Google's picker).
        Search,
        /// Nothing at all.
        Nothing,
        /// It closes the suggestions, and the library shows as it was.
        CloseOnly,
    }

    /// Google's picker as it is (live, 2026-10-19): the library grid, newest first, four items a
    /// day under date headings, below a search field. Typing opens suggestions over the grid —
    /// generic ones and people, never an album. Return searches: an album's exact title shows the
    /// album's items in the same grid, any other text Google's ordinary `results`. There is no
    /// Albums view. A click on a tile toggles it, a Shift+click selects from the last tile clicked;
    /// the header counts the selection; Done takes up to `cap`. A scroll moves the view a screen,
    /// a click only changes what is under the pointer. The eyes box the typed text in the field as
    /// a tile and as a heading, as qwen2.5vl did.
    struct Shelf {
        /// Whether each library item is a video (every third is a photo).
        library: Vec<bool>,
        albums: Vec<(&'static str, Vec<usize>)>,
        results: Vec<usize>,
        on_return: OnReturn,
        field: bool,
        cap: u32,
        typed: String,
        shown: Vec<usize>,
        page: usize,
        view: u32,
        hover: u32,
        selected: std::collections::BTreeSet<usize>,
        anchor: Option<usize>,
        done: bool,
        dones: u32,
        returns: u32,
        clicks: Vec<(u32, u32)>,
        shift_clicks: u32,
        asks: Vec<String>,
        url: Option<String>,
        tabs_closed: u32,
        /// The picker's page: `None` the hands run no scripts (another browser), `Some(false)`
        /// Chrome refuses them (Apple Events JavaScript off), `Some(true)` they run.
        js: Option<bool>,
        /// Sleeps after Return before its results show (Google answers an album after a delay).
        delay: u32,
        pending: Option<Vec<usize>>,
        /// The first item the page holds (its grid is virtual), and the items that never tick.
        dom_top: usize,
        wont_tick: std::collections::BTreeSet<usize>,
        scripts: u32,
        /// Looks at the page that find its grid still empty (the picker loading).
        loading: u32,
    }

    /// Checkboxes a [`Shelf`]'s page holds at a time (live: 29), and items a scroll moves on.
    const DOM_WINDOW: usize = 29;
    const DOM_STEP: usize = 20;

    impl Shelf {
        fn new(items: usize, albums: &[(&'static str, Vec<usize>)]) -> Rc<RefCell<Self>> {
            Rc::new(RefCell::new(Self {
                library: (0..items).map(|i| i % 3 != 0).collect(),
                albums: albums.to_vec(),
                results: Vec::new(),
                on_return: OnReturn::Search,
                field: true,
                cap: MAX_PICK,
                typed: String::new(),
                shown: (0..items).collect(),
                page: 0,
                view: 0,
                hover: 0,
                selected: Default::default(),
                anchor: None,
                done: false,
                dones: 0,
                returns: 0,
                clicks: Vec::new(),
                shift_clicks: 0,
                asks: Vec::new(),
                url: Some(PICKER_TAB.into()),
                tabs_closed: 0,
                js: None,
                delay: 0,
                pending: None,
                dom_top: 0,
                wont_tick: Default::default(),
                scripts: 0,
                loading: 0,
            }))
        }

        /// A tile's checkbox label in the page, as Google writes it (U+202F before PM).
        fn aria(&self, item: usize) -> String {
            let kind = if self.library[item] { "Video" } else { "Photo" };
            format!("{kind} - Landscape - {}, {}:{:02}:{:02}\u{202f}PM", Shelf::day_label(item), 1 + item % 12, (item / 60) % 60, item % 60)
        }

        /// The items whose checkboxes are in the page now.
        fn in_page(&self) -> Vec<usize> {
            self.shown.iter().skip(self.dom_top).take(DOM_WINDOW).copied().collect()
        }

        /// What the page answers to one of [`crate::dom::script`]'s steps.
        fn page_script(&mut self, js: &str) -> Result<String, ScreenError> {
            use crate::dom::{Step, script};
            self.scripts += 1;
            let mut clicked = 0u32;
            let mut moved = false;
            if js == script(Step::Tick) {
                for item in self.in_page() {
                    if !self.selected.contains(&item) && !self.wont_tick.contains(&item) {
                        self.selected.insert(item);
                        clicked += 1;
                    }
                }
            } else if js == script(Step::Top) {
                self.dom_top = 0;
            } else if js == script(Step::Next) {
                let end = self.shown.len().saturating_sub(DOM_WINDOW);
                let next = (self.dom_top + DOM_STEP).min(end);
                moved = next != self.dom_top;
                self.dom_top = next;
            } else if let Some(list) = js.split("var want=[").nth(1).and_then(|r| r.split(']').next()) {
                // a tick script: the tiles in the page whose labels hash to one of its numbers
                let want: Vec<u32> = list.split(',').filter_map(|n| n.parse().ok()).collect();
                for item in self.in_page() {
                    let label = crate::dom::tile_label(&self.aria(item)).unwrap_or_default();
                    if !self.selected.contains(&item) && !self.wont_tick.contains(&item) && want.contains(&crate::items::label_hash(&label)) {
                        self.selected.insert(item);
                        clicked += 1;
                    }
                }
            } else if js != script(Step::Look) {
                return Err(ScreenError::Other(format!("unexpected page script: {js}")));
            }
            // the other checkboxes are there too: "Select" and a day's
            let mut tiles = vec![
                serde_json::json!({"label": "Select", "checked": false}),
                serde_json::json!({"label": "Select all photos from Tue, Jul 29, 2025", "checked": false}),
            ];
            if self.loading > 0 {
                self.loading -= 1;
            } else {
                tiles.extend(self.in_page().into_iter().map(|i| serde_json::json!({"label": self.aria(i), "checked": self.selected.contains(&i)})));
            }
            let n = self.selected.len();
            let header = (n > 0 && !self.done).then(|| format!("{n} selected"));
            Ok(serde_json::json!({"tiles": tiles, "header": header, "top": self.dom_top, "height": 9000, "moved": moved, "clicked": clicked}).to_string())
        }

        /// The date heading over library item `item`.
        fn day_label(item: usize) -> String {
            const MONTHS: [&str; 12] = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
            let day = item / 4;
            format!("{} {}, {}", MONTHS[(day / 28) % 12], 1 + day % 28, 2026 - day / 336)
        }

        fn last_page(&self) -> usize {
            self.shown.len().saturating_sub(1) / SHELF_PAGE
        }

        /// The items on screen, with their tiles.
        fn on_screen(&self) -> Vec<(usize, Element)> {
            let tile = |k: usize, item: usize| {
                let (x, y) = (20 + 26 * (k % SHELF_ROW_TILES) as u32, SHELF_ROWS[k / SHELF_ROW_TILES] + 12);
                // each video its own duration (one label on many boxes is the model looping)
                let label = if self.library[item] { format!("0:{:02}", 10 + item % 50) } else { "photo".into() };
                Element { label, kind: String::new(), x1: x, y1: y, x2: x + TILE, y2: y + TILE }
            };
            self.shown.iter().skip(self.page * SHELF_PAGE).take(SHELF_PAGE).enumerate().map(|(k, &item)| (item, tile(k, item))).collect()
        }

        fn click(&mut self, px: u32, py: u32, shift: bool) {
            self.hover += 1; // the pointer hovers: the pixels change whatever the click does
            if shift {
                self.shift_clicks += 1;
            } else {
                self.clicks.push((px, py));
            }
            if (px, py) == DONE_AT {
                self.dones += 1;
                self.done = (1..=self.cap as usize).contains(&self.selected.len());
                if self.done {
                    self.view += 1; // the picker leaves its grid
                }
                return;
            }
            let hit = self.on_screen().into_iter().find(|(_, t)| (t.x1..t.x2).contains(&px) && (t.y1..t.y2).contains(&py)).map(|(i, _)| i);
            let Some(item) = hit else { return };
            match (shift, self.anchor) {
                (true, Some(a)) => {
                    let (p, q) = (self.shown.iter().position(|&s| s == a).unwrap(), self.shown.iter().position(|&s| s == item).unwrap());
                    let range: Vec<usize> = self.shown[p.min(q)..=p.max(q)].to_vec();
                    self.selected.extend(range);
                }
                _ => {
                    if !self.selected.remove(&item) {
                        self.selected.insert(item);
                    }
                    self.anchor = Some(item);
                }
            }
        }
    }

    struct ShelfHands(Rc<RefCell<Shelf>>);

    impl Act for ShelfHands {
        fn activate(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn click_png(&mut self, _cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            self.0.borrow_mut().click(px, py, false);
            Ok(())
        }
        fn shift_click_png(&mut self, _cap: &Capture, px: u32, py: u32) -> Result<(), ScreenError> {
            self.0.borrow_mut().click(px, py, true);
            Ok(())
        }
        fn type_text(&mut self, text: &str) -> Result<(), ScreenError> {
            let mut w = self.0.borrow_mut();
            w.typed.push_str(text);
            w.view += 1; // the suggestions open over the grid
            Ok(())
        }
        fn key_return(&mut self) -> Result<(), ScreenError> {
            let mut w = self.0.borrow_mut();
            w.returns += 1;
            match w.on_return {
                OnReturn::Search => {
                    let typed = w.typed.clone();
                    let found = w.albums.iter().find(|(t, _)| *t == typed).map(|(_, items)| items.clone()).unwrap_or_else(|| w.results.clone());
                    if w.delay > 0 {
                        w.pending = Some(found);
                    } else {
                        w.shown = found;
                        (w.page, w.anchor, w.dom_top) = (0, None, 0);
                        w.view += 1;
                    }
                }
                OnReturn::Nothing => {}
                OnReturn::CloseOnly => w.view += 1,
            }
            Ok(())
        }
        fn key_escape(&mut self) -> Result<(), ScreenError> {
            Ok(())
        }
        fn scroll(&mut self, n: i32) -> Result<(), ScreenError> {
            let mut w = self.0.borrow_mut();
            let page = if n > 0 { (w.page + 1).min(w.last_page()) } else { 0 };
            if page != w.page {
                w.page = page;
                w.view += 1;
            }
            Ok(())
        }
        fn sleep(&mut self, _ms: u64) {
            let mut w = self.0.borrow_mut();
            if w.pending.is_some() {
                w.delay = w.delay.saturating_sub(1);
                if w.delay == 0
                    && let Some(found) = w.pending.take()
                {
                    w.shown = found;
                    (w.page, w.anchor, w.dom_top) = (0, None, 0);
                    w.view += 1;
                }
            }
        }
        fn capture(&mut self) -> Result<Capture, ScreenError> {
            // a view apart from the last differs by 45 or more everywhere (moved); a hover by 5–15
            // (changed, not moved)
            let w = self.0.borrow();
            Ok(cap_of(((w.view % 4) * 60 + (w.hover % 4) * 5) as u8))
        }
        fn clear_text(&mut self, _chars: usize) -> Result<(), ScreenError> {
            self.0.borrow_mut().typed.clear();
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
        fn front_tab(&mut self) -> Result<Option<crate::tabs::Tab>, ScreenError> {
            let w = self.0.borrow();
            Ok(w.js.and(w.url.clone()).map(|url| crate::tabs::Tab { window: 1, id: 10, url }))
        }
        fn tab_url(&mut self, _tab: &crate::tabs::Tab) -> Result<Option<String>, ScreenError> {
            Ok(self.0.borrow().url.clone())
        }
        fn close_that_tab(&mut self, _tab: &crate::tabs::Tab) -> Result<bool, ScreenError> {
            self.close_tab().map(|()| true)
        }
        fn run_js(&mut self, _tab: &crate::tabs::Tab, js: &str) -> Result<String, ScreenError> {
            let mut w = self.0.borrow_mut();
            match w.js {
                Some(true) => w.page_script(js),
                _ => Err(ScreenError::Unsupported(crate::tabs::JS_OFF.into())),
            }
        }
    }

    struct ShelfEyes(Rc<RefCell<Shelf>>);

    impl Ground for ShelfEyes {
        fn inventory(&self, _png: &[u8], task: &str) -> Result<Vec<Element>, ScreenError> {
            let mut w = self.0.borrow_mut();
            w.asks.push(task.to_string());
            let el = |label: &str, x1: u32, y1: u32, x2: u32, y2: u32| Element { label: label.into(), kind: String::new(), x1, y1, x2, y2 };
            if task == crate::picker::SELECTED_COUNTER {
                let n = w.selected.len();
                return Ok(if n == 0 || w.done { vec![] } else { vec![el(&format!("{n} selected"), 2, 2, 28, 12)] });
            }
            let mut out = Vec::new();
            let screen = w.on_screen();
            if task == crate::picker::DAY_HEADINGS {
                if !w.typed.is_empty() {
                    out.push(el(&w.typed, 34, 8, 80, 16)); // the typed text, as a heading
                }
                for (r, row) in screen.chunks(SHELF_ROW_TILES).enumerate() {
                    if let Some((item, _)) = row.first() {
                        out.push(el(&Shelf::day_label(*item), 4, SHELF_ROWS[r] + 1, 48, SHELF_ROWS[r] + 9));
                    }
                }
            } else {
                if !w.typed.is_empty() {
                    out.push(el(&w.typed, 34, 2, 160, 22)); // the typed text, as a tile
                }
                let videos = task.starts_with(crate::picker::VIDEO_TILES);
                out.extend(screen.into_iter().filter(|(i, _)| !videos || w.library[*i]).map(|(_, t)| t));
            }
            Ok(out)
        }
        fn find(&self, _png: &[u8], target: &str) -> Result<Option<Element>, ScreenError> {
            let w = self.0.borrow();
            if target == crate::picker::SEARCH_FIELD {
                return Ok(w.field.then(shelf_field));
            }
            Ok((target.contains("Done") && !w.done).then(|| Element { label: "Done".into(), kind: String::new(), x1: 180, y1: 0, x2: 199, y2: 20 }))
        }
    }

    fn drive_shelf(w: &Rc<RefCell<Shelf>>, plan: &Plan) -> (Result<Summary, ScreenError>, Vec<String>) {
        let mut lines: Vec<String> = vec![];
        let s = drive(&mut ShelfHands(Rc::clone(w)), &ShelfEyes(Rc::clone(w)), plan, &mut |l| lines.push(l.into()));
        (s, lines)
    }

    fn album_plan(title: &str, videos_only: bool) -> Plan {
        plan_for(Request { items: Vec::new(), album: Some(title.into()), videos_only, queries: vec!["ignored".into()], max_items: 0, select_all: false })
    }

    const ROBOTS: &str = "Every Robot I've Met";

    /// The 92 items of [`ROBOTS`] (live: 92 and the app's own export), spread over the library:
    /// a photo first, then videos and photos.
    fn robots() -> Vec<usize> {
        (0..92).map(|k| 3 + 2 * k).collect()
    }

    #[test]
    fn an_albums_exact_title_is_searched_and_its_whole_grid_picked() {
        // live, 2026-10-19: the suggestions never name the album and the picker has no Albums
        // view; Return on the exact title shows the album's items under date headings
        let w = Shelf::new(300, &[(ROBOTS, robots())]);
        let plan = album_plan(ROBOTS, true);
        assert!(plan.select_all && plan.queries.is_empty() && plan.videos_only && plan.max_items == MAX_PICK, "{plan:?}");
        let (s, lines) = drive_shelf(&w, &plan);
        let s = s.unwrap();
        let w = w.borrow();
        assert_eq!(w.selected.iter().copied().collect::<Vec<_>>(), robots(), "exactly the album: {lines:#?}");
        assert!(s.album_shown && s.searched && s.done_pressed && w.done && w.dones == 1, "{s:?}");
        assert_eq!(s.selected, 92, "the header's count");
        assert_eq!((w.typed.as_str(), w.returns), (ROBOTS, 1), "the title as is (no ` videos`), then Return");
        // the field, the album's first tile — never the typed title boxed as a tile — the last by
        // Shift+click, then Done
        let first = Element { label: String::new(), kind: String::new(), x1: 20, y1: 42, x2: 44, y2: 66 };
        assert_eq!(w.clicks, [SHELF_FIELD_AT, crate::picker::click_point(&first), DONE_AT], "{lines:#?}");
        assert_eq!(w.shift_clicks, 1);
        // the album is selected whole, photos too (the picker cannot select by type)
        assert!(!w.asks.iter().any(|a| a.starts_with(crate::picker::VIDEO_TILES)), "{:?}", w.asks);
        assert!(s.note.contains("all 92 item(s) of the album") && s.note.contains("photos too"), "{}", s.note);
        assert!(s.tab_closed && w.tabs_closed == 1);
        // the eyes did offer the typed title as a tile, inside the field
        let typed = Shelf::new(300, &[]);
        typed.borrow_mut().typed = ROBOTS.into();
        let tiles = ShelfEyes(typed).inventory(&[], crate::picker::ALL_TILES).unwrap();
        assert!(tiles.iter().any(|t| t.label == ROBOTS && crate::picker::in_field(t, &shelf_field())), "{tiles:?}");
        // without videos only, the same pick
        let w = Shelf::new(300, &[(ROBOTS, robots())]);
        let (s, _) = drive_shelf(&w, &album_plan(ROBOTS, false));
        let s = s.unwrap();
        assert!(s.done_pressed && s.selected == 92 && w.borrow().selected.len() == 92 && !s.note.contains("photos"), "{s:?}");
    }

    #[test]
    fn a_search_that_shows_nothing_new_selects_nothing() {
        // Return did nothing: the view never moved
        let w = Shelf::new(300, &[(ROBOTS, robots())]);
        w.borrow_mut().on_return = OnReturn::Nothing;
        let (s, lines) = drive_shelf(&w, &album_plan(ROBOTS, true));
        let err = s.unwrap_err().to_string();
        assert!(err.contains(&format!("searching the picker for `{ROBOTS}` showed nothing new")) && err.contains("nothing was selected"), "{err}");
        assert!(lines.iter().any(|l| l.contains("waiting for the results")), "{lines:#?}");
        assert!(err.contains(&format!("did not move in the {} s after Return", u64::from(ALBUM_EYE_LOOKS) * SEARCH_SETTLE_MS / 1000)), "{err}");
        let w = w.borrow();
        assert_eq!(w.clicks, [SHELF_FIELD_AT], "the field only");
        assert!(w.shift_clicks == 0 && w.selected.is_empty() && w.dones == 0 && w.tabs_closed == 1);
        assert_eq!(w.asks.iter().filter(|a| *a == crate::picker::ALL_TILES).count(), 1, "the library's look only: {:?}", w.asks);
        // Return closed the suggestions and the library shows as it was. The typed title, boxed as
        // a tile and (it reads as a date) as a heading, lies in the field: the grid is the same
        for title in [ROBOTS, "Robots May 2024"] {
            let w = Shelf::new(300, &[(title, robots())]);
            w.borrow_mut().on_return = OnReturn::CloseOnly;
            let (s, lines) = drive_shelf(&w, &album_plan(title, true));
            let err = s.unwrap_err().to_string();
            assert!(err.contains("showed nothing new") && err.contains("the grid is the library's"), "{title}: {err} {lines:#?}");
            let w = w.borrow();
            assert!(w.clicks == [SHELF_FIELD_AT] && w.shift_clicks == 0 && w.selected.is_empty() && w.dones == 0, "{title}: {:?}", w.clicks);
        }
        assert_eq!(crate::picker::date_key("Robots May 2024").as_deref(), Some("2024-05"), "it would pass for a date heading");
    }

    #[test]
    fn a_runaway_grid_is_never_confirmed() {
        // the live failure's end: a grid that is no album (here Google's ordinary results for a
        // title that is no album's) scrolled until the budget ran out — the last click is not made
        // and Done is never pressed
        let w = Shelf::new(2000, &[(ROBOTS, robots())]);
        w.borrow_mut().results = (7..2000).collect();
        let (s, lines) = drive_shelf(&w, &album_plan("Every Robot I Met", true));
        let err = s.unwrap_err().to_string();
        assert!(err.contains("never ended within the budget") && err.contains("Done was not pressed"), "{err} {lines:#?}");
        let w = w.borrow();
        assert!(w.dones == 0 && w.shift_clicks == 0 && !w.done && w.tabs_closed == 1, "{:?}", w.clicks);
        // a grid that ends, with more than this run's maximum selected: not confirmed either
        let w = Shelf::new(300, &[(ROBOTS, robots())]);
        w.borrow_mut().results = (7..107).collect();
        let plan = plan_for(Request { items: Vec::new(), album: Some("robots".into()), videos_only: true, queries: vec![], max_items: 50, select_all: true });
        let (s, lines) = drive_shelf(&w, &plan);
        let err = s.unwrap_err().to_string();
        assert!(err.contains("selected 100, more than this run's 50-item maximum") && err.contains("Done was not pressed"), "{err} {lines:#?}");
        let w = w.borrow();
        assert!(w.dones == 0 && w.shift_clicks == 1 && w.selected.len() == 100 && !w.done, "{:?}", w.clicks);
    }

    #[test]
    fn a_grid_is_told_from_another_by_what_it_shows_not_only_where() {
        let look = |heads: &[&str], labels: &[&str]| GridLook {
            headings: heads.iter().map(|s| s.to_string()).collect(),
            tiles: labels.iter().zip(0u32..).map(|(l, i)| ((20 + 26 * i, 54), l.to_string())).collect(),
        };
        let library = look(&["today", "thu"], &["0 10", "photo", "0 31"]);
        assert!(library.same_as(&look(&["today", "thu"], &["0 10", "photo", "0 31"])));
        // a uniform grid has tiles at the same places whatever it shows; headings of this week
        // name no month
        assert!(!library.same_as(&look(&["today", "thu"], &["0 12", "photo", "0 07"])));
        assert!(!library.same_as(&look(&["today", "sat mar 4 2023"], &["0 10", "photo", "0 31"])));
        assert!(!library.same_as(&look(&["today", "thu"], &["0 10", "photo"])));
        // the same tile boxed a few pixels apart
        let mut jitter = library.clone();
        jitter.tiles[0].0.0 += 3;
        assert!(library.same_as(&jitter));
    }

    #[test]
    fn an_album_search_without_results_or_a_field_or_a_title_selects_nothing() {
        // no results: nothing to select, nothing to confirm, nothing to wait for
        let w = Shelf::new(300, &[(ROBOTS, robots())]);
        let (s, _) = drive_shelf(&w, &album_plan("Robot showcase", true));
        let s = s.unwrap();
        assert!(s.album_shown && !s.done_pressed && s.selected == 0 && s.stop_waiting(Duration::ZERO), "{s:?}");
        assert!(s.note.contains("showed no item to select"), "{}", s.note);
        assert!(w.borrow().clicks == [SHELF_FIELD_AT] && w.borrow().dones == 0);
        // no search field: an error after a few looks, nothing clicked or typed
        let w = Shelf::new(300, &[(ROBOTS, robots())]);
        w.borrow_mut().field = false;
        let (s, lines) = drive_shelf(&w, &album_plan(ROBOTS, true));
        let err = s.unwrap_err().to_string();
        assert!(err.contains("search field never appeared") && err.contains("nothing was selected"), "{err}");
        assert!(w.borrow().clicks.is_empty() && w.borrow().typed.is_empty());
        assert_eq!(lines.iter().filter(|l| l.contains("waiting for the picker")).count(), SEARCH_TRIES - 1);
        // a title with nothing to search for: refused before anything
        let w = Shelf::new(300, &[]);
        let (s, _) = drive_shelf(&w, &album_plan(" !! ", true));
        assert!(s.unwrap_err().to_string().contains("no letters or digits"));
        assert!(w.borrow().clicks.is_empty() && w.borrow().asks.is_empty());
    }

    /// The album of the live failure (2026-10-24): "akku tesla toy", 52 items, its results
    /// shown a few seconds after Return.
    const AKKU: &str = "akku tesla toy";
    fn akku() -> Vec<usize> {
        (0..52).map(|k| 40 + 3 * k).collect()
    }

    #[test]
    fn an_album_that_answers_late_is_waited_for_and_ticked_in_the_page() {
        // live: two looks 2.5 s apart missed the results; the page shows them after a delay
        let w = Shelf::new(300, &[(AKKU, akku())]);
        {
            let mut w = w.borrow_mut();
            w.js = Some(true);
            w.delay = 6;
        }
        let (s, lines) = drive_shelf(&w, &album_plan(AKKU, true));
        let s = s.unwrap_or_else(|e| panic!("{e} {lines:#?}"));
        let w = w.borrow();
        assert_eq!(w.selected.iter().copied().collect::<Vec<_>>(), akku(), "exactly the album: {lines:#?}");
        assert!(s.album_shown && s.done_pressed && w.done && w.dones == 1 && s.selected == 52, "{s:?}");
        // no range gesture and no tile clicked by eyes: the field, then Done
        assert_eq!((w.clicks.as_slice(), w.shift_clicks), ([SHELF_FIELD_AT, DONE_AT].as_slice(), 0), "{lines:#?}");
        assert!(lines.iter().any(|l| l.contains("ticked all 52 tile(s) of the album in the page")), "{lines:#?}");
        // neither the day's checkbox nor "Select" was clicked: only the 52 tiles are selected
        assert!(s.tab_closed && w.tabs_closed == 1);
        // by eyes (no page) the late results are waited for too
        let w = Shelf::new(300, &[(AKKU, akku())]);
        w.borrow_mut().delay = 4;
        let (s, lines) = drive_shelf(&w, &album_plan(AKKU, true));
        let s = s.unwrap_or_else(|e| panic!("{e} {lines:#?}"));
        assert!(s.done_pressed && s.selected == 52 && w.borrow().selected.len() == 52, "{s:?}");
        assert!(lines.iter().any(|l| l.contains("waiting for the results")), "{lines:#?}");
    }

    #[test]
    fn the_library_is_read_from_the_page_once_it_has_loaded() {
        // the grid still loading when the agent starts: its library is read once it has settled,
        // and the album is ticked in the page all the same
        let w = Shelf::new(300, &[(AKKU, akku())]);
        {
            let mut w = w.borrow_mut();
            w.js = Some(true);
            w.loading = 3;
            w.delay = 4;
        }
        let (s, lines) = drive_shelf(&w, &album_plan(AKKU, true));
        let s = s.unwrap_or_else(|e| panic!("{e} {lines:#?}"));
        assert!(s.done_pressed && s.selected == 52 && w.borrow().shift_clicks == 0, "in the page: {s:?} {lines:#?}");
        // a grid that never settles in the page: the eyes do it
        let w = Shelf::new(300, &[(AKKU, akku())]);
        {
            let mut w = w.borrow_mut();
            w.js = Some(true);
            w.loading = 1000;
        }
        let (s, lines) = drive_shelf(&w, &album_plan(AKKU, true));
        let s = s.unwrap_or_else(|e| panic!("{e} {lines:#?}"));
        assert!(s.done_pressed && s.selected == 52 && w.borrow().shift_clicks == 1, "by eyes: {s:?}");
        assert!(lines.iter().any(|l| l.contains("no settled grid; selecting by eyes")), "{lines:#?}");
    }

    #[test]
    fn a_page_that_refuses_scripts_is_picked_by_eyes() {
        let w = Shelf::new(300, &[(AKKU, akku())]);
        w.borrow_mut().js = Some(false);
        let (s, lines) = drive_shelf(&w, &album_plan(AKKU, true));
        let s = s.unwrap();
        assert!(s.done_pressed && s.selected == 52 && w.borrow().shift_clicks == 1, "the range gesture: {s:?}");
        assert!(lines.iter().any(|l| l.contains("Allow JavaScript from Apple Events") && l.contains("selecting by eyes")), "{lines:#?}");
    }

    #[test]
    fn the_page_never_confirms_a_runaway_a_stuck_tile_or_the_library() {
        // more tiles than this run's maximum: stopped, not confirmed
        let w = Shelf::new(300, &[]);
        {
            let mut w = w.borrow_mut();
            w.js = Some(true);
            w.results = (7..107).collect();
        }
        let plan = plan_for(Request { items: Vec::new(), album: Some("robots".into()), videos_only: true, queries: vec![], max_items: 50, select_all: true });
        let (s, lines) = drive_shelf(&w, &plan);
        let err = s.unwrap_err().to_string();
        assert!(err.contains("more than this run's 50-item maximum") && err.contains("Done was not pressed"), "{err} {lines:#?}");
        assert_eq!(w.borrow().dones, 0);
        // a tile that never ticks: said, not confirmed
        let w = Shelf::new(300, &[(AKKU, akku())]);
        {
            let mut w = w.borrow_mut();
            w.js = Some(true);
            w.wont_tick = [43].into();
        }
        let (s, lines) = drive_shelf(&w, &album_plan(AKKU, true));
        let err = s.unwrap_err().to_string();
        assert!(err.contains("1 of the album's 52 tile(s) would not tick") && err.contains("Done was not pressed"), "{err} {lines:#?}");
        assert!(w.borrow().dones == 0 && w.borrow().tabs_closed == 1);
        // Return did nothing: the library's tiles stay, nothing is ticked
        let w = Shelf::new(300, &[(AKKU, akku())]);
        {
            let mut w = w.borrow_mut();
            w.js = Some(true);
            w.on_return = OnReturn::Nothing;
        }
        let (s, _) = drive_shelf(&w, &album_plan(AKKU, true));
        let err = s.unwrap_err().to_string();
        assert!(err.contains(&format!("showed the library's tiles for {ALBUM_PAGE_LOOKS} s after Return")), "{err}");
        assert!(w.borrow().selected.is_empty() && w.borrow().dones == 0);
        // a title that is no album's and has no results: nothing to select, nothing confirmed
        let w = Shelf::new(300, &[(AKKU, akku())]);
        w.borrow_mut().js = Some(true);
        let (s, _) = drive_shelf(&w, &album_plan("akku tesla", true));
        let s = s.unwrap();
        assert!(s.album_shown && !s.done_pressed && s.selected == 0 && s.note.contains("showed no item to select"), "{s:?}");
    }

    /// What a catalogue knows of the shelf's item `item`: its UTC time (the Mac's zone taken off
    /// the local time its tile shows) and its kind.
    fn wanted_of(w: &Rc<RefCell<Shelf>>, item: usize) -> crate::items::Wanted {
        use chrono::TimeZone;
        let label = crate::dom::tile_label(&w.borrow().aria(item)).unwrap();
        let (video, local) = crate::items::tile_time(&label).unwrap();
        let off = home_offset(chrono::Utc.from_utc_datetime(&local));
        crate::items::Wanted { taken: chrono::Utc.from_utc_datetime(&(local - chrono::Duration::minutes(i64::from(off)))), video }
    }

    fn day_search(item: usize) -> &'static str {
        let label = Shelf::day_label(item);
        let day = chrono::NaiveDate::parse_from_str(&label, "%b %d, %Y").unwrap();
        Box::leak(crate::items::day_text(day).into_boxed_str())
    }

    #[test]
    fn exact_items_are_searched_by_day_and_only_their_tiles_ticked() {
        // two days' grids, as the picker answers their dates; the items wanted are a video of
        // the first, and a photo and a video of the second
        let w = Shelf::new(40, &[]);
        w.borrow_mut().albums = vec![(day_search(4), vec![4, 5, 6, 7]), (day_search(8), vec![8, 9, 10, 11])];
        w.borrow_mut().js = Some(true);
        let items = vec![wanted_of(&w, 5), wanted_of(&w, 9), wanted_of(&w, 10)];
        let plan = plan_for(Request { items, ..Request::default() });
        assert_eq!((plan.max_items, plan.select_all, plan.videos_only), (3, false, false));
        let (s, lines) = drive_shelf(&w, &plan);
        let s = s.unwrap_or_else(|e| panic!("{e}\n{}", lines.join("\n")));
        let picked: Vec<usize> = w.borrow().selected.iter().copied().collect();
        assert_eq!(picked, [5, 9, 10], "only the wanted tiles\n{}", lines.join("\n"));
        assert!(s.done_pressed && w.borrow().done);
        assert_eq!(s.searches.len(), 2, "one search per day: {:?}", s.searches);
        assert!(s.left.is_empty());
        assert!(s.note.starts_with("ticked 3 of 3 item(s) over 2 day search(es)"), "{}", s.note);
    }

    #[test]
    fn an_item_no_day_shows_is_left_and_the_rest_are_picked() {
        let w = Shelf::new(40, &[]);
        w.borrow_mut().albums = vec![(day_search(4), vec![4, 5, 6, 7])];
        w.borrow_mut().js = Some(true);
        // item 20's day is never answered (its search shows nothing): it is looked for on its
        // days, then left with its time
        let lost = wanted_of(&w, 20);
        let plan = plan_for(Request { items: vec![wanted_of(&w, 6), lost], ..Request::default() });
        let (s, lines) = drive_shelf(&w, &plan);
        let s = s.unwrap_or_else(|e| panic!("{e}\n{}", lines.join("\n")));
        assert_eq!(w.borrow().selected.iter().copied().collect::<Vec<_>>(), [6]);
        assert!(s.done_pressed);
        assert_eq!(s.left, [lost.rfc3339()]);
        assert!(s.searches.len() >= 2 && s.searches.len() <= 4, "{:?}", s.searches);
        assert!(s.searches.iter().any(|q| q.outcome == Outcome::NoResults));
    }

    #[test]
    fn exact_items_need_the_page_and_nothing_is_selected_without_it() {
        let w = Shelf::new(12, &[]);
        let plan = plan_for(Request { items: vec![wanted_of(&w, 5)], ..Request::default() });
        let (s, _) = drive_shelf(&w, &plan);
        let e = s.unwrap_err().to_string();
        assert!(e.contains("Allow JavaScript from Apple Events"), "{e}");
        assert!(w.borrow().selected.is_empty() && !w.borrow().done);
        // and an item pick's budget grows with its days, within the session's
        let many: Vec<crate::items::Wanted> = (0..400).map(|i| wanted_of(&w, i % 12)).collect();
        let p = plan_for(Request { items: many, ..Request::default() });
        assert_eq!(p.items.len(), crate::items::MAX_ITEMS);
        assert!(p.timeout <= MAX_SESSION_TIME && p.max_actions > 0);
    }

    #[test]
    fn the_browser_leaving_the_front_ends_the_whole_run() {
        assert!(fatal(&ScreenError::NotInFront("Google Chrome".into())), "no more searches once keys could go elsewhere");
    }
}
