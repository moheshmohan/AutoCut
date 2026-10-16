//! Google Photos picks a local vision agent drives — `filmcraft-cli photos pick --auto/--album`,
//! the MCP `googlephotos_pick`, and a whole album as a showcase's seed ([`album_source`]) — in
//! one place: the picking session (`filmcraft-gphotos`), the screen agent that selects in
//! Google's picker (`filmcraft-screenagent`), and a wait for the selection that ends as soon as
//! the agent is done without pressing Done, instead of waiting out the whole timeout.

use std::path::PathBuf;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use filmcraft_autocut::album::{AlbumFetched, AlbumPick, AlbumSource};
use filmcraft_gphotos::{GPhotosError, GooglePhotos, PickOptions, PickOutcome, Picked};
use filmcraft_screenagent::agent::{Request, Summary};

/// How long an album pick waits for the agent to select the album and press Done.
pub const ALBUM_WAIT: Duration = Duration::from_secs(900);

/// A pick the agent drove: how it ended, and what the agent reported (`None` without an agent,
/// or when a cancel left it running).
pub struct Driven {
    pub outcome: PickOutcome,
    pub agent: Option<Result<Summary, String>>,
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
    Request { query: String::new(), max_items: filmcraft_gphotos::api::MAX_PICK, select_all: true, album: Some(title.to_string()), videos_only }
}

fn join(h: JoinHandle<Result<Summary, String>>) -> Result<Summary, String> {
    h.join().unwrap_or_else(|_| Err("the agent thread died".into()))
}

/// Open (or resume) a picking session, let the screen agent (`request`; none: the user picks)
/// drive Google's picker, and wait within `opts.wait` for the selection, then download it. The
/// wait ends early once the agent finished without pressing Done (nothing will be picked: the
/// outcome is then `Pending`, the session still open for the user), and when `cancelled()`.
/// Blocking: HTTP, sleeps and the agent's thread.
pub fn pick_driven(
    gp: &GooglePhotos,
    opts: &PickOptions,
    request: Option<Request>,
    progress: &mut dyn FnMut(&str),
    cancelled: &dyn Fn() -> bool,
) -> Result<Driven, GPhotosError> {
    let prepared = gp.prepare(opts, progress)?;
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
    Ok(Driven { outcome, agent: report })
}

/// Whether picking can start, from what is on disk (no network): the OAuth client and a sign-in.
pub fn pick_readiness() -> Result<(), String> {
    filmcraft_gphotos::pick_readiness(&filmcraft_gphotos::data_dir())
}

/// What a finished album pick brought to disk.
fn fetched(picked: Picked, note: Option<String>) -> AlbumFetched {
    let cached = picked.fetched.iter().filter(|f| f.was_cached).count();
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
            if let PickOutcome::Pending(s) = &driven.outcome {
                let _ = gp.delete_session(&s.id);
            }
            return Err("cancelled".into());
        }
        let nothing = driven.nothing_to_pick();
        match driven.outcome {
            PickOutcome::Done(picked) => Ok(fetched(picked, note)),
            PickOutcome::Pending(s) => {
                let _ = gp.delete_session(&s.id);
                Err(match (nothing, note) {
                    (true, Some(n)) => format!("the album \"{title}\" had nothing to pick: {n}"),
                    (true, None) => format!("the album \"{title}\" had nothing to pick"),
                    (false, Some(n)) => format!("nothing was picked from the album \"{title}\" in time ({n})"),
                    (false, None) => format!("nothing was picked from the album \"{title}\" in time"),
                })
            }
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
        assert!(r.query.is_empty());
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
        let d = Driven { outcome: PickOutcome::Pending(session.clone()), agent: Some(Ok(summary(false))) };
        assert!(d.nothing_to_pick());
        assert_eq!(d.agent_note().as_deref(), Some("no album titled Robots"));
        let d = Driven { outcome: PickOutcome::Pending(session.clone()), agent: Some(Ok(summary(true))) };
        assert!(!d.nothing_to_pick(), "Done was pressed: Google may still report the selection");
        let d = Driven { outcome: PickOutcome::Pending(session), agent: Some(Err("no Screen Recording".into())) };
        assert!(!d.nothing_to_pick());
        assert!(d.agent_note().is_some_and(|n| n.contains("no Screen Recording")));
        let d = Driven { outcome: PickOutcome::Done(Picked::default()), agent: None };
        assert!(!d.nothing_to_pick() && d.agent_note().is_none());
    }

    #[test]
    fn an_album_pick_brings_back_what_was_downloaded() {
        let picked = Picked { fetched: vec![], failed: vec!["b.mp4: 404".into()], skipped: vec!["c.jpg: not a video".into()], session_deleted: true };
        let f = fetched(picked, Some("selected 2".into()));
        assert_eq!((f.paths.len(), f.downloaded, f.cached, f.failed.len(), f.skipped.len()), (0, 0, 0, 1, 1));
        assert_eq!(f.note.as_deref(), Some("selected 2"));
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
