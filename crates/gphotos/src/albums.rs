//! Albums FilmCraft makes in the user's Google Photos library (the Library API).
//!
//! Since 2025-03-31 an app sees only what it created itself:
//!
//! | call | scope it needs |
//! |---|---|
//! | `albums.create`, `albums.addEnrichment` | `photoslibrary.appendonly` ([`crate::auth::UPLOAD_SCOPE`]) |
//! | `mediaItems.batchCreate` with an `albumId` | `photoslibrary.appendonly` |
//! | `albums.batchAddMediaItems` | `photoslibrary.appendonly` (or `photoslibrary.edit.appcreateddata`) |
//! | `albums.list`, `albums.get`, `mediaItems.search` | `photoslibrary.readonly.appcreateddata` ([`crate::auth::ALBUM_SCOPE`]) |
//!
//! So [`Albums::list`] returns only the albums this app created (never the user's own), Google
//! adds only media items this app uploaded, and only into an album this app created. Items the
//! user adds to such an album by hand in Google Photos are *not* readable here — reading a
//! user-curated album goes through the picker (`photos pick --album`).
//!
//! Album commands need both scopes (`photos auth --albums`): finding an album FilmCraft made
//! earlier by its title is what makes "create" idempotent, and that needs the read scope.
//!
//! Every request goes through [`Transport`], so the tests drive the album calls with a fake.
//! Google's answers are hostile input: ids must be URL-safe (they go into request paths),
//! titles and links are cleaned and capped, lists are paged with a cap on pages and albums, and
//! a page token that repeats ends the listing.

use std::time::Duration;

use serde_json::{Value, json};

use crate::GPhotosError;
use crate::auth::{ALBUMS_CONSENT, truncate, urlencode};
use crate::upload::{Method, Request, Response, Transport};

/// `albums.create` and `albums.list`; an album's own calls are `{ALBUMS_URL}/{id}:…`.
pub const ALBUMS_URL: &str = "https://photoslibrary.googleapis.com/v1/albums";
/// Google's limit on an album title, in characters.
pub const MAX_TITLE_CHARS: usize = 500;
/// `albums.batchAddMediaItems` takes at most this many ids per call.
pub const MAX_ADD_PER_CALL: usize = 50;
/// Google's limit on the items in one album; more ids than this in one `add` are refused.
pub const MAX_ALBUM_ITEMS: usize = 20_000;
/// `albums.list`'s largest page.
pub const PAGE_SIZE: u32 = 50;
/// The most albums a listing reads; a listing stopped by it is [`Listed::complete`] `false`.
pub const MAX_ALBUMS: usize = 2000;
/// The most pages a listing asks for (a server sending short pages still ends).
const MAX_PAGES: usize = 80;
/// The most albums kept from one page, whatever Google sends.
const MAX_PER_PAGE: usize = 500;
/// The longest note ([`Albums::add_note`]) FilmCraft puts at the top of an album.
pub const MAX_NOTE_CHARS: usize = 1000;
/// Each album request's time limit (they are small).
pub const TIMEOUT: Duration = Duration::from_secs(60);
/// Caps on what Google sends back.
const MAX_ID_CHARS: usize = 512;
const MAX_URL_CHARS: usize = 4096;
const MAX_PAGE_TOKEN_CHARS: usize = 4096;
/// Attempts of one request (or of one album creation) before giving up.
const MAX_TRIES: u32 = 3;
/// The longest wait between attempts.
const MAX_BACKOFF: Duration = Duration::from_secs(8);
/// A string this long, of id characters only, is taken for an album id, never a title.
const ID_LIKE_CHARS: usize = 40;

// ---------------------------------------------------------------------------------------------
// what an album is
// ---------------------------------------------------------------------------------------------

/// An album this app created.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Album {
    pub id: String,
    pub title: String,
    /// The album in Google Photos (only ever an `https://photos.google.com/` link).
    pub product_url: Option<String>,
    /// How many items it holds (Google's `mediaItemsCount`), when Google says.
    pub media_items: Option<u64>,
    /// Whether this app may add to it (`isWriteable`).
    pub writeable: bool,
}

impl Album {
    /// An album known only by its id (what an upload into it reports before it is looked up).
    pub fn with_id(id: String) -> Self {
        Self { id, ..Self::default() }
    }

    pub fn to_json(&self) -> Value {
        json!({"id": self.id, "title": self.title, "productUrl": self.product_url, "mediaItemsCount": self.media_items, "isWriteable": self.writeable})
    }
}

/// A title Google takes: control characters become spaces, runs of white space one space,
/// trimmed; empty or longer than [`MAX_TITLE_CHARS`] is an error (never cut short, so the title
/// asked for is the title found next time).
pub fn clean_title(title: &str) -> Result<String, GPhotosError> {
    let spaced: String = title.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
    let t = spaced.split_whitespace().collect::<Vec<_>>().join(" ");
    if t.is_empty() {
        return Err(GPhotosError::Album("an album needs a title".into()));
    }
    let n = t.chars().count();
    if n > MAX_TITLE_CHARS {
        return Err(GPhotosError::Album(format!("the album title is {n} characters; Google Photos takes at most {MAX_TITLE_CHARS}")));
    }
    Ok(t)
}

/// Whether `c` may appear in a Google Photos id (base64url).
fn id_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '-' || c == '_'
}

/// An album or media item id as Google makes them: 1–512 URL-safe characters (an album id goes
/// into a request path, so nothing else is let through).
pub fn check_id(id: &str) -> Result<String, GPhotosError> {
    let id = id.trim();
    if id.is_empty() || id.chars().count() > MAX_ID_CHARS || !id.chars().all(id_char) {
        return Err(GPhotosError::Album(format!(
            "`{}` is not a Google Photos id (ids are letters, digits, - and _)",
            truncate(&id.chars().map(|c| if c.is_control() { '?' } else { c }).collect::<String>(), 60)
        )));
    }
    Ok(id.to_string())
}

/// Whether `s` is an id rather than a title: long, and id characters only. Such a string is
/// looked up by id and never becomes the title of a new album.
pub fn looks_like_id(s: &str) -> bool {
    let s = s.trim();
    s.chars().count() >= ID_LIKE_CHARS && s.chars().count() <= MAX_ID_CHARS && s.chars().all(id_char)
}

/// The album among `albums` that `title_or_id` names: the first with exactly that (cleaned)
/// title, else the one with that id.
pub fn find_in<'a>(albums: &'a [Album], title_or_id: &str) -> Option<&'a Album> {
    let by_title = clean_title(title_or_id).ok().and_then(|t| albums.iter().find(|a| a.title == t));
    by_title.or_else(|| {
        let id = title_or_id.trim();
        albums.iter().find(|a| !id.is_empty() && a.id == id)
    })
}

// ---------------------------------------------------------------------------------------------
// request bodies and answers
// ---------------------------------------------------------------------------------------------

/// `albums.create`'s body.
pub fn create_body(title: &str) -> Value {
    json!({"album": {"title": title}})
}

/// `albums.batchAddMediaItems`' body.
pub fn add_body(ids: &[String]) -> Value {
    json!({"mediaItemIds": ids})
}

/// `albums.addEnrichment`'s body: a text note at the top of the album.
pub fn note_body(text: &str) -> Value {
    json!({"newEnrichmentItem": {"textEnrichment": {"text": text}}, "albumPosition": {"position": "FIRST_IN_ALBUM"}})
}

/// `albums.list`'s URL for the page after `token` (the first page without one).
pub fn list_url(token: Option<&str>) -> String {
    match token {
        Some(t) => format!("{ALBUMS_URL}?pageSize={PAGE_SIZE}&pageToken={}", urlencode(t)),
        None => format!("{ALBUMS_URL}?pageSize={PAGE_SIZE}"),
    }
}

/// An album from Google's JSON, or `None` when it has no usable id.
fn album_from(v: &Value) -> Option<Album> {
    let id = check_id(v.get("id")?.as_str()?).ok()?;
    let title: String = v.get("title").and_then(Value::as_str).unwrap_or_default().chars().filter(|c| !c.is_control()).take(MAX_TITLE_CHARS).collect();
    let product_url = v
        .get("productUrl")
        .and_then(Value::as_str)
        .filter(|u| u.starts_with("https://photos.google.com/") && u.len() <= MAX_URL_CHARS && !u.contains(|c: char| c.is_whitespace() || c.is_control()))
        .map(str::to_string);
    // Google sends int64 as a string; a number is taken too
    let media_items = match v.get("mediaItemsCount") {
        Some(Value::String(s)) => s.trim().parse::<u64>().ok(),
        Some(n) => n.as_u64(),
        None => None,
    };
    let writeable = v.get("isWriteable").and_then(Value::as_bool).unwrap_or(false);
    Some(Album { id, title: title.trim().to_string(), product_url, media_items, writeable })
}

/// One page of `albums.list`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Page {
    pub albums: Vec<Album>,
    pub next: Option<String>,
    /// Albums left out for an unusable id.
    pub skipped: usize,
}

/// An `albums.list` answer. No `albums` (no albums yet) is an empty page.
pub fn parse_page(text: &str) -> Result<Page, GPhotosError> {
    let v: Value = serde_json::from_str(text).map_err(|e| GPhotosError::Album(format!("Google sent an unreadable album list: {e}")))?;
    if !v.is_object() {
        return Err(GPhotosError::Album("Google sent an album list that is not a JSON object".into()));
    }
    let raw = match v.get("albums") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(a)) => a.as_slice(),
        Some(_) => return Err(GPhotosError::Album("Google sent an album list whose `albums` is not a list".into())),
    };
    let mut page = Page::default();
    for a in raw.iter().take(MAX_PER_PAGE) {
        match album_from(a) {
            Some(album) => page.albums.push(album),
            None => page.skipped = page.skipped.saturating_add(1),
        }
    }
    page.skipped = page.skipped.saturating_add(raw.len().saturating_sub(MAX_PER_PAGE));
    page.next = v.get("nextPageToken").and_then(Value::as_str).map(str::trim).filter(|t| !t.is_empty()).map(str::to_string);
    if page.next.as_ref().is_some_and(|t| t.chars().count() > MAX_PAGE_TOKEN_CHARS || t.chars().any(char::is_control)) {
        return Err(GPhotosError::Album("Google sent an unusable album list page token".into()));
    }
    Ok(page)
}

/// An `albums.create` answer: the new album.
pub fn parse_album(text: &str) -> Result<Album, GPhotosError> {
    let v: Value = serde_json::from_str(text).map_err(|e| GPhotosError::Album(format!("Google sent an unreadable album: {e}")))?;
    album_from(&v).ok_or_else(|| GPhotosError::Album("Google answered without a usable album id".into()))
}

/// A readable error for a failed album request, with the fix for the usual causes.
pub fn album_error(status: u16, text: &str) -> GPhotosError {
    let err = serde_json::from_str::<Value>(text).ok().and_then(|v| v.get("error").cloned()).unwrap_or(Value::Null);
    let message = err.get("message").and_then(Value::as_str).unwrap_or_else(|| truncate(text.trim(), 300));
    let message = truncate(message, 300);
    let kind = err.get("status").and_then(Value::as_str).unwrap_or_default();
    let reasons: Vec<&str> =
        err.get("details").and_then(Value::as_array).into_iter().flatten().take(16).filter_map(|d| d.get("reason").and_then(Value::as_str)).collect();
    let has = |r: &str| reasons.contains(&r);
    if has("SERVICE_DISABLED") || message.contains("has not been used") || message.contains("is disabled") {
        return GPhotosError::Setup(format!(
            "the Photos Library API is not enabled in your Google Cloud project: open APIs & Services → Library, search \"Photos Library API\", press Enable, wait a few minutes and try again (Google said: {message})"
        ));
    }
    if status == 401 || kind == "UNAUTHENTICATED" || has("ACCESS_TOKEN_SCOPE_INSUFFICIENT") || message.contains("insufficient authentication scopes") {
        return GPhotosError::Consent(ALBUMS_CONSENT.into());
    }
    let label = if kind.is_empty() { status.to_string() } else { format!("{status} {kind}") };
    let app_created = "Google lets an app add only media items it uploaded itself, and only into albums it created (Library API rules since 2025-03-31)";
    match status {
        403 => GPhotosError::Album(format!("Google refused ({label}: {message}); {app_created}")),
        400 if message.to_ascii_lowercase().contains("media item") => {
            GPhotosError::Album(format!("Google refused a media item id ({label}: {message}); {app_created} — use ids from `photos upload`"))
        }
        404 => GPhotosError::Album(format!("Google knows no such album ({label}: {message}); it may have been deleted in Google Photos")),
        _ => GPhotosError::Album(format!("Google answered {label}: {message}")),
    }
}

// ---------------------------------------------------------------------------------------------
// the calls
// ---------------------------------------------------------------------------------------------

/// The outcome of one request.
enum Attempt {
    Done(Response),
    /// No answer, or one worth trying again (408, 429, 5xx).
    Retry(String),
    Fail(GPhotosError),
}

/// What a listing found.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Listed {
    pub albums: Vec<Album>,
    /// Every page was read (false: stopped at [`MAX_ALBUMS`] / the page cap, or at a page
    /// token Google repeated).
    pub complete: bool,
    /// Albums left out for an unusable id.
    pub skipped: usize,
}

impl Listed {
    pub fn to_json(&self) -> Value {
        json!({"albums": self.albums.iter().map(Album::to_json).collect::<Vec<_>>(), "complete": self.complete, "skipped": self.skipped})
    }
}

/// What [`Albums::ensure`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ensured {
    pub album: Album,
    /// It was created now (`false`: an album FilmCraft made earlier with that title, reused).
    pub created: bool,
    /// The note was put at the top of the new album.
    pub note_added: bool,
    /// Why the note could not be added (the album itself was made).
    pub note_error: Option<String>,
}

impl Ensured {
    pub fn to_json(&self) -> Value {
        let mut v = self.album.to_json();
        if let Some(o) = v.as_object_mut() {
            o.insert("created".into(), json!(self.created));
            o.insert("noteAdded".into(), json!(self.note_added));
            if let Some(e) = &self.note_error {
                o.insert("noteError".into(), json!(e));
            }
        }
        v
    }
}

/// What [`Albums::add`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Added {
    pub album: Album,
    /// Distinct ids sent (duplicates are sent once).
    pub added: usize,
    /// `batchAddMediaItems` calls made.
    pub calls: usize,
}

impl Added {
    pub fn to_json(&self) -> Value {
        json!({"album": self.album.to_json(), "added": self.added, "calls": self.calls})
    }
}

/// The album calls over a transport, with a bearer token asked for before every request.
pub struct Albums<'a> {
    http: &'a dyn Transport,
    bearer: &'a dyn Fn() -> Result<String, GPhotosError>,
    /// The first wait before another attempt, doubled each time (at most 8 s); zero in tests.
    pub backoff: Duration,
}

impl<'a> Albums<'a> {
    pub fn new(http: &'a dyn Transport, bearer: &'a dyn Fn() -> Result<String, GPhotosError>) -> Self {
        Self { http, bearer, backoff: Duration::from_secs(1) }
    }

    /// Wait before attempt `n + 1`.
    fn pause(&self, n: u32) {
        let wait = self.backoff.saturating_mul(1u32 << n.min(4)).min(MAX_BACKOFF);
        if !wait.is_zero() {
            std::thread::sleep(wait);
        }
    }

    fn send(&self, method: Method, url: &str, body: Option<&Value>) -> Attempt {
        let token = match (self.bearer)() {
            Ok(t) => t,
            Err(e) => return Attempt::Fail(e),
        };
        let text = body.map(Value::to_string).unwrap_or_default();
        let mut headers = vec![("Authorization", format!("Bearer {token}"))];
        if body.is_some() {
            headers.push(("Content-Type", "application/json".into()));
        }
        let req = Request { method, url, headers, body: text.as_bytes() };
        match self.http.send(&req) {
            Err(e) => Attempt::Retry(format!("connection: {}", truncate(&e, 200))),
            Ok(r) if (200..300).contains(&r.status) => Attempt::Done(r),
            Ok(r) if matches!(r.status, 408 | 429 | 500..=599) => Attempt::Retry(format!("Google answered {}", r.status)),
            Ok(r) => Attempt::Fail(album_error(r.status, &r.body)),
        }
    }

    /// A request that may be repeated (a listing, adding the same items again).
    fn send_retrying(&self, method: Method, url: &str, body: Option<&Value>) -> Result<Response, GPhotosError> {
        let mut why = String::new();
        for n in 0..MAX_TRIES {
            if n > 0 {
                self.pause(n - 1);
            }
            match self.send(method, url, body) {
                Attempt::Done(r) => return Ok(r),
                Attempt::Retry(w) => why = w,
                Attempt::Fail(e) => return Err(e),
            }
        }
        Err(GPhotosError::Album(format!("Google Photos did not answer ({why}); gave up after {MAX_TRIES} attempts")))
    }

    /// Every album this app created (at most [`MAX_ALBUMS`]), page by page.
    pub fn list(&self) -> Result<Listed, GPhotosError> {
        let mut out = Listed::default();
        let mut token: Option<String> = None;
        let mut seen_tokens: Vec<String> = Vec::new();
        for _ in 0..MAX_PAGES {
            let resp = self.send_retrying(Method::Get, &list_url(token.as_deref()), None)?;
            let page = parse_page(&resp.body)?;
            out.skipped = out.skipped.saturating_add(page.skipped);
            for a in page.albums {
                if out.albums.len() >= MAX_ALBUMS {
                    return Ok(out);
                }
                out.albums.push(a);
            }
            match page.next {
                None => {
                    out.complete = true;
                    return Ok(out);
                }
                // a token Google already sent: it is looping; stop with what there is
                Some(t) if seen_tokens.contains(&t) => return Ok(out),
                Some(t) => {
                    seen_tokens.push(t.clone());
                    token = Some(t);
                }
            }
        }
        Ok(out)
    }

    /// The album this app created with the title (or id) `title_or_id`, if any.
    pub fn find(&self, title_or_id: &str) -> Result<Option<Album>, GPhotosError> {
        Ok(find_in(&self.list()?.albums, title_or_id).cloned())
    }

    /// The album named `title_or_id` (an exact title or an id), created when there is none.
    /// An album FilmCraft made earlier with that exact title is reused, never duplicated: a
    /// creation whose answer is lost is followed by a fresh listing before it is tried again.
    /// `note` (at most [`MAX_NOTE_CHARS`]) goes at the top of an album created now.
    pub fn ensure(&self, title_or_id: &str, note: Option<&str>) -> Result<Ensured, GPhotosError> {
        let wanted = title_or_id.trim();
        let note = check_note(note)?;
        let title = if looks_like_id(wanted) { None } else { Some(clean_title(wanted)?) };
        let mut why = String::new();
        for n in 0..MAX_TRIES {
            if n > 0 {
                self.pause(n - 1);
            }
            let listed = self.list()?;
            if let Some(a) = find_in(&listed.albums, wanted) {
                return Ok(Ensured { album: a.clone(), created: false, note_added: false, note_error: None });
            }
            let Some(title) = title.as_deref() else {
                return Err(GPhotosError::Album(format!(
                    "no album FilmCraft made has the id `{wanted}` (Google lets FilmCraft add only to albums it created; `photos albums` lists them)"
                )));
            };
            if !listed.complete {
                return Err(GPhotosError::Album(format!(
                    "FilmCraft has more albums than it can list ({}); not creating `{title}` in case it is one of the rest — pass the album's id instead",
                    listed.albums.len()
                )));
            }
            match self.send(Method::Post, ALBUMS_URL, Some(&create_body(title))) {
                Attempt::Done(resp) => {
                    let album = parse_album(&resp.body)?;
                    let (note_added, note_error) = match note.as_deref() {
                        Some(text) => match self.add_note(&album.id, text) {
                            Ok(()) => (true, None),
                            Err(e) => (false, Some(e.to_string())),
                        },
                        None => (false, None),
                    };
                    return Ok(Ensured { album, created: true, note_added, note_error });
                }
                // the album may have been made with the answer lost: list again first
                Attempt::Retry(w) => why = w,
                Attempt::Fail(e) => return Err(e),
            }
        }
        Err(GPhotosError::Album(format!("Google Photos did not answer creating the album ({why}); gave up after {MAX_TRIES} attempts")))
    }

    /// Put a text note at the top of album `album_id` (`albums.addEnrichment`).
    pub fn add_note(&self, album_id: &str, text: &str) -> Result<(), GPhotosError> {
        let id = check_id(album_id)?;
        let Some(text) = check_note(Some(text))? else { return Ok(()) };
        // not retried: a lost answer would put the note in twice
        match self.send(Method::Post, &format!("{ALBUMS_URL}/{id}:addEnrichment"), Some(&note_body(&text))) {
            Attempt::Done(_) => Ok(()),
            Attempt::Retry(why) => Err(GPhotosError::Album(format!("the note was not added ({why})"))),
            Attempt::Fail(e) => Err(e),
        }
    }

    /// Add media items (this app's uploads) to the album named `title_or_id` (an exact title
    /// or an id; it must exist), [`MAX_ADD_PER_CALL`] per call. Duplicate ids are sent once.
    pub fn add(&self, title_or_id: &str, ids: &[String]) -> Result<Added, GPhotosError> {
        let ids = check_item_ids(ids)?;
        let album = self.find(title_or_id)?.ok_or_else(|| {
            GPhotosError::Album(format!(
                "FilmCraft has made no album `{}` (titles match exactly; `photos albums` lists them, `photos album create` makes one)",
                truncate(title_or_id.trim(), 80)
            ))
        })?;
        let url = format!("{ALBUMS_URL}/{}:batchAddMediaItems", check_id(&album.id)?);
        let mut added = 0usize;
        let mut calls = 0usize;
        for chunk in ids.chunks(MAX_ADD_PER_CALL) {
            if let Err(e) = self.send_retrying(Method::Post, &url, Some(&add_body(chunk))) {
                return Err(if added == 0 { e } else { GPhotosError::Album(format!("added {added} of {} item(s), then: {e}", ids.len())) });
            }
            calls = calls.saturating_add(1);
            added = added.saturating_add(chunk.len());
        }
        Ok(Added { album, added, calls })
    }
}

/// A note FilmCraft may put in an album: trimmed (empty is none), at most [`MAX_NOTE_CHARS`].
pub fn check_note(note: Option<&str>) -> Result<Option<String>, GPhotosError> {
    let Some(n) = note.map(str::trim).filter(|n| !n.is_empty()) else { return Ok(None) };
    let count = n.chars().count();
    if count > MAX_NOTE_CHARS {
        return Err(GPhotosError::Album(format!("the album description is {count} characters; FilmCraft adds at most {MAX_NOTE_CHARS}")));
    }
    Ok(Some(n.chars().map(|c| if c.is_control() && c != '\n' { ' ' } else { c }).collect()))
}

/// Media item ids for an `add`: each a valid id, duplicates dropped (first kept), 1 to
/// [`MAX_ALBUM_ITEMS`] of them.
pub fn check_item_ids(ids: &[String]) -> Result<Vec<String>, GPhotosError> {
    if ids.is_empty() {
        return Err(GPhotosError::Album("no media item ids given".into()));
    }
    if ids.len() > MAX_ALBUM_ITEMS {
        return Err(GPhotosError::Album(format!("{} media item ids; an album holds at most {MAX_ALBUM_ITEMS}", ids.len())));
    }
    let mut out: Vec<String> = Vec::with_capacity(ids.len());
    let mut seen = std::collections::HashSet::new();
    for id in ids {
        let id = check_id(id)?;
        if seen.insert(id.clone()) {
            out.push(id);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
