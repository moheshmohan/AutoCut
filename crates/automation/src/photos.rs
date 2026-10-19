//! Google Photos picks a local vision agent drives — `filmcraft-cli photos pick --auto/--album`,
//! the MCP `googlephotos_pick`, and a whole album as a showcase's seed ([`album_source`]) — in
//! one place: the picking session (`filmcraft-gphotos`), the screen agent that selects in
//! Google's picker (`filmcraft-screenagent`), and a wait for the selection that ends as soon as
//! the agent is done without pressing Done, instead of waiting out the whole timeout.
//!
//! One tab, one session, many searches: before an agent-driven session opens its tab, Chrome's
//! earlier picker tabs are closed (macOS), the agent types every search of a pick into that one
//! session and closes the tab when it is done, and a session the agent gave up on is deleted.
//! Many searches — a list, or a sweep of a whole history — run in as few sessions as the item
//! cap allows ([`pick_searches`]): what one session could not take goes to the next.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use filmcraft_autocut::album::{AlbumFetched, AlbumPick, AlbumSource};
use filmcraft_gphotos::{GPhotosError, GooglePhotos, PickOptions, PickOutcome, Picked};
use filmcraft_screenagent::agent::{Request, Searched, Summary};

/// How long an album pick waits for the agent to select the album and press Done.
pub const ALBUM_WAIT: Duration = Duration::from_secs(900);

/// A pick the agent drove: how it ended, and what the agent reported (`None` without an agent,
/// or when a cancel left it running).
pub struct Driven {
    pub outcome: PickOutcome,
    pub agent: Option<Result<Summary, String>>,
    /// The session was abandoned (an agent drove it and nothing was picked) and Google confirmed
    /// its deletion.
    pub abandoned: bool,
}

impl Driven {
    /// The agent finished without pressing Done (no album or search result to select): nothing
    /// was picked, and nothing will be.
    pub fn nothing_to_pick(&self) -> bool {
        matches!(self.outcome, PickOutcome::Pending(_)) && self.agent.as_ref().is_some_and(|r| r.as_ref().is_ok_and(|s| !s.done_pressed))
    }

    /// The agent's note, or why it failed.
    pub fn agent_note(&self) -> Option<String> {
        self.agent.as_ref().map(|r| match r {
            Ok(s) => s.note.clone(),
            Err(e) => format!("the picker agent failed: {e}"),
        })
    }
}

/// The agent request that picks the whole album `title` (all of it, up to Google's limit).
pub fn album_request(title: &str, videos_only: bool) -> Request {
    Request { queries: Vec::new(), max_items: filmcraft_gphotos::api::MAX_PICK, select_all: true, album: Some(title.to_string()), videos_only }
}

fn join(h: JoinHandle<Result<Summary, String>>) -> Result<Summary, String> {
    h.join().unwrap_or_else(|_| Err("the agent thread died".into()))
}

/// Show a new session's picker link in a new tab of the default browser. With `agent` (an agent
/// will drive the picker, so nobody picks by hand in the earlier picker tabs) Chrome's earlier
/// picker tabs are closed first (macOS) — a tab holding a selection asks "Leave site?" and is
/// closed once more, Chrome's Leave — so the new session's tab is the only one; without, they
/// are left be and only counted. Says how, or `None`.
fn open_picker(uri: &str, agent: bool, notes: &mut Vec<String>) -> Option<String> {
    #[cfg(target_os = "macos")]
    {
        use filmcraft_screenagent::tabs;
        let shell = filmcraft_screenagent::capture::OsShell;
        if agent {
            let mut pause = |ms: u64| std::thread::sleep(Duration::from_millis(ms.min(10_000)));
            match tabs::close_picker_tabs(&shell, &mut pause, &mut |_| {}) {
                Ok(t) => {
                    if t.closed > 0 {
                        notes.push(format!("closed {} earlier Google Photos picker tab(s) in Chrome", t.closed));
                    }
                    if !t.left.is_empty() {
                        notes.push(format!("{} earlier picker tab(s) would not close: `filmcraft-cli photos tidy` closes picker tabs", t.left.len()));
                    }
                    notes.extend(t.notes);
                }
                Err(e) => notes.push(format!("could not close the earlier picker tabs ({e})")),
            }
        } else if let Ok(open) = tabs::picker_tabs(&shell)
            && !open.is_empty()
        {
            notes.push(format!("{} more Google Photos picker tab(s) are open in Chrome: `filmcraft-cli photos tidy` closes them", open.len()));
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (agent, notes);
    filmcraft_gphotos::open_in_browser(uri).then(|| "in your browser".to_string())
}

/// Longest an agent's pick waits for the browser to be free.
const BROWSER_WAIT: std::time::Duration = std::time::Duration::from_secs(15 * 60);

/// Open (or resume) a picking session, let the screen agent (`request`; none: the user picks)
/// drive Google's picker, and wait within `opts.wait` for the selection, then download it. The
/// wait ends early once the agent finished without pressing Done (nothing will be picked), and
/// when `cancelled()`. A session an agent drove that ends `Pending` is abandoned — the agent
/// closed its tab — and deleted; one the user picks by hand stays open to resume. An agent's
/// picker gets the only picker tab (the earlier ones are closed first). Blocking: HTTP, sleeps
/// and the agent's thread.
pub fn pick_driven(
    gp: &GooglePhotos,
    opts: &PickOptions,
    request: Option<Request>,
    progress: &mut dyn FnMut(&str),
    cancelled: &dyn Fn() -> bool,
) -> Result<Driven, GPhotosError> {
    let driven_by_agent = request.is_some();
    // an agent drives Chrome with the real mouse and keyboard: one driver at a time (another
    // pick, `photos tidy`, a Claude session holding the browser), held until the pick is over
    let _browser = if driven_by_agent {
        let mut said = false;
        let wait = opts.wait.min(BROWSER_WAIT);
        let guard = filmcraft_locks::acquire(filmcraft_locks::Resource::Chrome, "Google Photos picker agent", wait, &mut |e| {
            if !said {
                progress(&format!("waiting for {e}"));
                said = true;
            }
        })
        .map_err(|e| GPhotosError::Setup(format!("{e}; try again when it is free (`filmcraft-cli locks` shows who holds what)")))?;
        Some(guard)
    } else {
        None
    };
    let mut notes: Vec<String> = Vec::new();
    let prepared = gp.prepare_with(opts, &mut |uri| open_picker(uri, driven_by_agent, &mut notes), progress)?;
    for n in &notes {
        progress(n);
    }
    let mut agent = request.map(filmcraft_screenagent::agent::spawn);
    // the pick outcome is authoritative; the agent's own budget bounds it. Once the agent is
    // done without pressing Done (its search matched nothing, or it failed), nothing more will
    // be picked: stop waiting
    let mut report: Option<Result<Summary, String>> = None;
    let mut finished: Option<Instant> = None;
    let mut give_up = || {
        if cancelled() {
            return true;
        }
        if report.is_none() && agent.as_ref().is_some_and(JoinHandle::is_finished) {
            report = agent.take().map(join);
            finished = Some(Instant::now());
        }
        match (&report, finished) {
            (Some(Ok(s)), Some(t)) => s.stop_waiting(t.elapsed()),
            (Some(Err(_)), _) => true,
            _ => false,
        }
    };
    let outcome = gp.complete_unless(prepared, opts.wait, &mut give_up, progress)?;
    // an agent still running when the pick is done (Done pressed by hand) is waited for; one
    // still at work on a pick that is pending (out of time, or cancelled) is left be
    let report = match (report, &outcome) {
        (Some(r), _) => Some(r),
        (None, PickOutcome::Done(_)) if !cancelled() => agent.map(join),
        (None, _) => None,
    };
    // a session the agent gave up on (it closed the tab) must not linger
    let abandoned = match &outcome {
        PickOutcome::Pending(s) if driven_by_agent => {
            let deleted = gp.delete_session(&s.id).is_ok();
            if deleted {
                progress("deleted the unfinished picking session");
            }
            deleted
        }
        _ => false,
    };
    Ok(Driven { outcome, agent: report, abandoned })
}

/// Searches per picking session in a [`pick_searches`] (the agent's budget grows with each).
pub const PER_SESSION: usize = filmcraft_screenagent::agent::MAX_SEARCHES;
/// Tries of one search over the sessions of a [`pick_searches`].
const MAX_TRIES: u32 = 2;
/// The most sessions one [`pick_searches`] opens.
const MAX_SESSIONS: usize = 200;

/// The searches of a pick, in order, waiting for their sessions; how often each was tried.
#[derive(Debug, Default)]
pub struct Queue {
    todo: VecDeque<String>,
    tries: HashMap<String, u32>,
    given_up: Vec<String>,
}

impl Queue {
    pub fn new(queries: Vec<String>) -> Self {
        Self { todo: queries.into(), ..Self::default() }
    }

    pub fn is_empty(&self) -> bool {
        self.todo.is_empty()
    }

    /// The next session's searches: the first `n` (at least one) waiting.
    pub fn next_batch(&mut self, n: usize) -> Vec<String> {
        let n = n.clamp(1, self.todo.len().max(1));
        self.todo.drain(..n.min(self.todo.len())).collect()
    }

    /// After a session: `left` (not reached) go back first, as they were; `retry` (tried, worth
    /// another go) follow them, each at most [`MAX_TRIES`] times in all, then given up.
    pub fn after(&mut self, left: &[String], retry: &[String]) {
        let mut front: Vec<String> = left.to_vec();
        for q in retry {
            let tries = self.tries.entry(q.clone()).or_default();
            *tries += 1;
            if *tries >= MAX_TRIES {
                self.given_up.push(q.clone());
            } else if !front.contains(q) {
                front.push(q.clone());
            }
        }
        for q in front.into_iter().rev() {
            self.todo.push_front(q);
        }
    }

    /// Give up on `queries` (more than a session holds): they go to [`Self::remaining`] as they
    /// are, never tried again in this pick.
    pub fn give_up(&mut self, queries: &[String]) {
        for q in queries {
            self.todo.retain(|t| t != q);
            if !self.given_up.contains(q) {
                self.given_up.push(q.clone());
            }
        }
    }

    /// Searches still waiting, then those given up on: what a next run should take.
    pub fn remaining(&self) -> Vec<String> {
        self.todo.iter().chain(&self.given_up).cloned().collect()
    }
}

/// Many searches in as few picking sessions as the item cap allows.
#[derive(Debug, Clone)]
pub struct Searches {
    pub queries: Vec<String>,
    /// Select every result of each search (else tiles, up to `max_items` in all, one session).
    pub select_all: bool,
    /// The item cap of each session (Google's `maxItemCount`, 1–2000) — with `select_all` the
    /// agent leaves searches for the next session rather than pass it.
    pub max_items: u32,
    /// Searches per session (1–[`PER_SESSION`]).
    pub per_session: usize,
}

/// One session of a [`pick_searches`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct SessionReport {
    pub queries: Vec<String>,
    pub searches: Vec<Searched>,
    /// The header's count when Done was pressed.
    pub selected: u32,
    pub downloaded: usize,
    pub cached: usize,
    pub note: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// What a [`pick_searches`] brought.
#[derive(Debug, Default)]
pub struct Swept {
    /// Every session's picks together (an item picked twice is there once).
    pub picked: Picked,
    pub sessions: Vec<SessionReport>,
    /// Searches no session got to, or that failed twice, or more than a session holds: for a next
    /// run.
    pub left: Vec<String>,
    /// Of `left`, the searches whose results alone are more than a session holds (the picker
    /// selects past its cap but refuses Done): to split (by day) or to pick with a higher cap.
    pub too_big: Vec<String>,
    /// Why the sweep stopped early, if it did.
    pub stopped: Option<String>,
}

/// Run `req.queries` through the picker in sessions of at most `req.per_session` searches, each
/// driven by the agent ([`pick_driven`]): one tab, one Done and one download batch per session.
/// What a session leaves (its cap, its budget) and what failed in it (once) go to the next
/// session; a session whose agent failed outright (no grants, no model), or that got nowhere,
/// ends the sweep with the rest left over. Without `select_all` only one session runs.
pub fn pick_searches(gp: &GooglePhotos, opts: &PickOptions, req: &Searches, progress: &mut dyn FnMut(&str), cancelled: &dyn Fn() -> bool) -> Swept {
    let mut queue = Queue::new(filmcraft_screenagent::agent::clean_queries(req.queries.clone()));
    let mut out = Swept::default();
    let per = req.per_session.clamp(1, PER_SESSION);
    let cap = req.max_items.clamp(1, filmcraft_gphotos::api::MAX_PICK);
    while !queue.is_empty() {
        if cancelled() {
            out.stopped = Some("cancelled".into());
            break;
        }
        if out.sessions.len() >= MAX_SESSIONS {
            out.stopped = Some(format!("stopped after {MAX_SESSIONS} sessions"));
            break;
        }
        let batch = queue.next_batch(per);
        let k = out.sessions.len() + 1;
        progress(&format!("session {k}: {} search(es): {}", batch.len(), batch.join(" · ")));
        let request = Request { queries: batch.clone(), max_items: cap, select_all: req.select_all, album: None, videos_only: true };
        let plan_time = filmcraft_screenagent::agent::plan_for(request.clone()).timeout;
        let wait = opts.wait.max(plan_time + filmcraft_screenagent::agent::DONE_GRACE + Duration::from_secs(60)).min(filmcraft_gphotos::MAX_WAIT);
        let session_opts = PickOptions { max_items: Some(cap), wait, resume: None, open_browser: true, ..opts.clone() };
        let mut report = SessionReport { queries: batch.clone(), searches: vec![], selected: 0, downloaded: 0, cached: 0, note: String::new(), error: None };
        let driven = match pick_driven(gp, &session_opts, Some(request), progress, cancelled) {
            Ok(d) => d,
            Err(e) => {
                report.error = Some(e.to_string());
                out.sessions.push(report);
                queue.after(&batch, &[]);
                out.stopped = Some(format!("session {k}: {e}"));
                break;
            }
        };
        let summary = match driven.agent {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                // the agent itself failed (grants, model): every session would
                report.error = Some(e.clone());
                out.sessions.push(report);
                queue.after(&[], &batch);
                out.stopped = Some(format!("session {k}: the picker agent failed: {e}"));
                break;
            }
            None => {
                out.sessions.push(report);
                queue.after(&batch, &[]);
                out.stopped = Some(format!("session {k}: the agent did not report (cancelled, or out of time)"));
                break;
            }
        };
        report.searches = summary.searches.clone();
        report.selected = summary.selected;
        report.note = summary.note.clone();
        match driven.outcome {
            PickOutcome::Done(picked) => {
                report.cached = picked.fetched.iter().filter(|f| f.was_cached).count();
                report.downloaded = picked.fetched.len().saturating_sub(report.cached);
                merge(&mut out.picked, picked);
                queue.after(&summary.left, &summary.retry());
            }
            PickOutcome::Pending(_) => {
                // nothing came of it: what it selected is worth another go; what it skipped, what
                // it left (`left` goes back as it is) and what no session holds are not
                use filmcraft_screenagent::agent::Outcome;
                let retry: Vec<String> = summary
                    .searches
                    .iter()
                    .filter(|s| !matches!(s.outcome, Outcome::NoResults | Outcome::Deferred | Outcome::TooBig))
                    .map(|s| s.query.clone())
                    .collect();
                queue.after(&summary.left, &retry);
            }
        }
        let too_big = summary.too_big();
        queue.give_up(&too_big);
        for q in too_big {
            if !out.too_big.contains(&q) {
                out.too_big.push(q);
            }
        }
        let got_somewhere = !summary.searches.is_empty();
        out.sessions.push(report);
        if !got_somewhere {
            out.stopped = Some(format!("session {k} did not get through a single search"));
            break;
        }
        if !req.select_all {
            break; // tiles up to max_items in all: one session
        }
    }
    out.left = queue.remaining();
    out
}

/// Add a session's picks to the sweep's (an item already there is not added twice).
fn merge(into: &mut Picked, more: Picked) {
    for f in more.fetched {
        if !into.fetched.iter().any(|g| g.item.id == f.item.id) {
            into.fetched.push(f);
        }
    }
    into.skipped.extend(more.skipped);
    into.failed.extend(more.failed);
    into.session_deleted = more.session_deleted;
}

/// Whether picking can start, from what is on disk (no network): the OAuth client and a sign-in.
pub fn pick_readiness() -> Result<(), String> {
    filmcraft_gphotos::pick_readiness(&filmcraft_gphotos::data_dir())
}

/// What a finished album pick brought to disk. The agent selects an album whole, photos too (the
/// picker cannot select by type): the note says how many photos a videos-only pick left out.
fn fetched(picked: Picked, note: Option<String>) -> AlbumFetched {
    let cached = picked.fetched.iter().filter(|f| f.was_cached).count();
    let note = match filmcraft_gphotos::not_videos(&picked.skipped) {
        0 => note,
        n => {
            let photos = format!("{n} photo(s) of the album were selected with it and not downloaded (videos only)");
            Some(match note.filter(|s| !s.is_empty()) {
                Some(s) => format!("{s}; {photos}"),
                None => photos,
            })
        }
    };
    AlbumFetched {
        paths: picked.fetched.iter().map(|f| f.path.to_string_lossy().into_owned()).collect(),
        downloaded: picked.fetched.len().saturating_sub(cached),
        cached,
        failed: picked.failed,
        skipped: picked.skipped,
        note,
    }
}

/// Whole Google Photos albums through the picker, which the screen agent drives: the
/// [`AlbumSource`] the app, the CLI and the MCP server give the showcase jobs.
pub struct PickerAlbums {
    dir: PathBuf,
}

/// The album source over the data dir's Google Photos sign-in (`filmcraft-cli photos auth`).
pub fn album_source() -> Arc<dyn AlbumSource> {
    Arc::new(PickerAlbums { dir: filmcraft_gphotos::data_dir() })
}

impl AlbumSource for PickerAlbums {
    fn ready(&self) -> Result<(), String> {
        filmcraft_gphotos::pick_readiness(&self.dir)
    }

    fn pick(&self, req: &AlbumPick, progress: &mut dyn FnMut(&str), cancelled: &dyn Fn() -> bool) -> Result<AlbumFetched, String> {
        let title = filmcraft_gphotos::albums::clean_title(&req.title).map_err(|e| e.to_string())?;
        let since = req.since.clone().filter(|d| !d.trim().is_empty());
        if let Some(d) = since.as_deref().filter(|d| !filmcraft_gphotos::is_day(d)) {
            return Err(format!("the album's date wants YYYY-MM-DD, not `{d}`"));
        }
        let gp = GooglePhotos::configured().map_err(|e| e.to_string())?;
        let opts = PickOptions { max_items: None, wait: ALBUM_WAIT, resume: None, open_browser: true, since, videos_only: req.videos_only };
        let driven = pick_driven(&gp, &opts, Some(album_request(&title, req.videos_only)), progress, cancelled).map_err(|e| e.to_string())?;
        let note = driven.agent_note();
        if cancelled() {
            return Err("cancelled".into()); // a pending session is deleted already
        }
        let nothing = driven.nothing_to_pick();
        match driven.outcome {
            PickOutcome::Done(picked) => Ok(fetched(picked, note)),
            PickOutcome::Pending(_) => Err(match (nothing, note) {
                (true, Some(n)) => format!("the album \"{title}\" had nothing to pick: {n}"),
                (true, None) => format!("the album \"{title}\" had nothing to pick"),
                (false, Some(n)) => format!("nothing was picked from the album \"{title}\" in time ({n})"),
                (false, None) => format!("nothing was picked from the album \"{title}\" in time"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_album_request_selects_all_of_it() {
        let r = album_request("Robot showcase", true);
        assert_eq!((r.album.as_deref(), r.select_all, r.videos_only, r.max_items), (Some("Robot showcase"), true, true, filmcraft_gphotos::api::MAX_PICK));
        assert!(r.queries.is_empty());
    }

    #[test]
    fn a_pending_pick_whose_agent_pressed_nothing_has_nothing_to_pick() {
        let session = filmcraft_gphotos::api::PickingSession {
            id: "s1".into(),
            picker_uri: String::new(),
            poll_interval: None,
            timeout_in: None,
            media_items_set: false,
            expire_time: None,
        };
        let summary = |done| Summary { done_pressed: done, note: "no album titled Robots".into(), ..Summary::default() };
        let d = Driven { outcome: PickOutcome::Pending(session.clone()), agent: Some(Ok(summary(false))), abandoned: true };
        assert!(d.nothing_to_pick());
        assert_eq!(d.agent_note().as_deref(), Some("no album titled Robots"));
        let d = Driven { outcome: PickOutcome::Pending(session.clone()), agent: Some(Ok(summary(true))), abandoned: true };
        assert!(!d.nothing_to_pick(), "Done was pressed: Google may still report the selection");
        let d = Driven { outcome: PickOutcome::Pending(session), agent: Some(Err("no Screen Recording".into())), abandoned: true };
        assert!(!d.nothing_to_pick());
        assert!(d.agent_note().is_some_and(|n| n.contains("no Screen Recording")));
        let d = Driven { outcome: PickOutcome::Done(Picked::default()), agent: None, abandoned: false };
        assert!(!d.nothing_to_pick() && d.agent_note().is_none());
    }

    fn strings(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_queue_carries_what_a_session_left_and_gives_up_after_two_tries() {
        let mut queue = Queue::new(strings(&["a", "b", "c", "d", "e"]));
        assert_eq!(queue.next_batch(3), strings(&["a", "b", "c"]));
        // a and b were searched, c was left (the cap); b selected nothing new
        queue.after(&strings(&["c"]), &strings(&["b"]));
        assert_eq!(queue.next_batch(3), strings(&["c", "b", "d"]), "left first, then the retry, then the rest");
        queue.after(&[], &strings(&["b"])); // b's second try: given up
        assert_eq!(queue.next_batch(9), strings(&["e"]));
        queue.after(&[], &[]);
        assert!(queue.is_empty());
        assert_eq!(queue.remaining(), strings(&["b"]), "given up searches are for a next run");
        // a batch is never empty while searches wait, nor bigger than asked
        let mut queue = Queue::new(strings(&["x", "y"]));
        assert_eq!(queue.next_batch(0), strings(&["x"]));
        assert_eq!(queue.next_batch(5), strings(&["y"]));
        assert!(queue.next_batch(5).is_empty());
        // a session that took nothing hands everything back, in order
        let mut queue = Queue::new(strings(&["p", "q", "r"]));
        let batch = queue.next_batch(2);
        queue.after(&batch, &[]);
        assert_eq!(queue.remaining(), strings(&["p", "q", "r"]));
    }

    #[test]
    fn a_search_more_than_a_session_holds_is_given_up_at_once() {
        // live: `videos October 2022` took a 40-item session to 143; the agent unselected it
        let mut queue = Queue::new(strings(&["oct", "nov", "dec"]));
        let batch = queue.next_batch(1);
        assert_eq!(batch, strings(&["oct"]));
        queue.after(&[], &[]);
        queue.give_up(&batch);
        assert_eq!(queue.next_batch(9), strings(&["nov", "dec"]), "never tried again");
        queue.after(&[], &[]);
        assert_eq!(queue.remaining(), strings(&["oct"]), "but listed for a next run");
        // given up while still waiting (handed back as `left` by mistake): taken out, listed once
        let mut queue = Queue::new(strings(&["a", "b"]));
        queue.give_up(&strings(&["b", "b"]));
        assert_eq!(queue.remaining(), strings(&["a", "b"]));
        assert_eq!(queue.next_batch(9), strings(&["a"]));
    }

    #[test]
    fn sessions_merge_their_picks_once_each() {
        let item = |id: &str| filmcraft_gphotos::api::PickedItem {
            id: id.into(),
            filename: format!("{id}.mp4"),
            mime_type: "video/mp4".into(),
            kind: filmcraft_gphotos::api::MediaKind::Video,
            created: String::new(),
            base_url: String::new(),
            video_status: None,
        };
        let fetched = |id: &str| filmcraft_gphotos::Fetched { item: item(id), path: PathBuf::from(format!("/c/{id}.mp4")), was_cached: false };
        let mut all = Picked::default();
        merge(&mut all, Picked { fetched: vec![fetched("a"), fetched("b")], skipped: vec!["x.jpg: not a video".into()], ..Picked::default() });
        merge(&mut all, Picked { fetched: vec![fetched("b"), fetched("c")], failed: vec!["d.mp4: 404".into()], ..Picked::default() });
        assert_eq!(all.fetched.iter().map(|f| f.item.id.as_str()).collect::<Vec<_>>(), ["a", "b", "c"]);
        assert_eq!((all.skipped.len(), all.failed.len()), (1, 1));
    }

    #[test]
    fn an_album_pick_brings_back_what_was_downloaded() {
        let picked = Picked { fetched: vec![], failed: vec!["b.mp4: 404".into()], skipped: vec!["c.jpg: not a video".into()], session_deleted: true };
        let f = fetched(picked.clone(), Some("selected 2".into()));
        assert_eq!((f.paths.len(), f.downloaded, f.cached, f.failed.len(), f.skipped.len()), (0, 0, 0, 1, 1));
        // the album was selected whole, photos too: how many were left out is said
        assert_eq!(f.note.as_deref(), Some("selected 2; 1 photo(s) of the album were selected with it and not downloaded (videos only)"));
        assert_eq!(fetched(picked, None).note.as_deref(), Some("1 photo(s) of the album were selected with it and not downloaded (videos only)"));
        // no photo left out: the agent's note as it is
        let f = fetched(Picked { skipped: vec!["old.mp4: created 2019-05-01".into()], ..Picked::default() }, Some("selected 1".into()));
        assert_eq!(f.note.as_deref(), Some("selected 1"));
    }

    #[test]
    fn the_album_source_says_when_google_photos_is_not_connected() {
        let dir = std::env::temp_dir().join(format!("filmcraft-automation-albums-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let src = PickerAlbums { dir: dir.clone() };
        let e = src.ready().unwrap_err();
        assert!(e.contains("not connected") && e.contains("photos auth"), "{e}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
