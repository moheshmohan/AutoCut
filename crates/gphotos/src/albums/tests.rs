//! The album calls against a fake Google library: request shapes, paging, idempotent
//! creation, batched adds, error mapping and hostile answers. Offline.

use std::sync::Mutex;

use super::*;

/// One request as the fake saw it.
#[derive(Debug, Clone)]
struct Seen {
    method: Method,
    url: String,
    auth: Option<String>,
    content_type: Option<String>,
    body: String,
}

/// A stand-in for the Library API's album endpoints, with an in-memory album list.
struct FakeLibrary {
    s: Mutex<Lib>,
}

struct Lib {
    albums: Vec<Value>,
    per_page: usize,
    /// Answers given before the library's own, in order, to requests whose URL contains the key.
    scripted: Vec<(&'static str, Result<Response, String>)>,
    /// The next creation makes the album but its answer is lost.
    lose_create_answer: bool,
    /// The next creation is refused with this answer.
    refuse_create: Option<Response>,
    /// After this many accepted `batchAddMediaItems` calls, the next is refused.
    refuse_add_after: Option<usize>,
    seen: Vec<Seen>,
    added: Vec<(String, Vec<String>)>,
    notes: Vec<(String, String)>,
    created: usize,
}

fn ok(body: &str) -> Response {
    Response { status: 200, headers: Vec::new(), body: body.into() }
}

fn status(code: u16, body: &str) -> Response {
    Response { status: code, headers: Vec::new(), body: body.into() }
}

fn album_id(n: usize) -> String {
    format!("AF1QipAlbum{n:0>45}")
}

fn album(n: usize, title: &str) -> Value {
    json!({"id": album_id(n), "title": title, "productUrl": format!("https://photos.google.com/lr/album/{}", album_id(n)), "mediaItemsCount": "3", "isWriteable": true})
}

impl FakeLibrary {
    fn new(albums: Vec<Value>) -> Self {
        Self {
            s: Mutex::new(Lib {
                albums,
                per_page: 2,
                scripted: Vec::new(),
                lose_create_answer: false,
                refuse_create: None,
                refuse_add_after: None,
                seen: Vec::new(),
                added: Vec::new(),
                notes: Vec::new(),
                created: 0,
            }),
        }
    }

    fn script(self, key: &'static str, answer: Result<Response, String>) -> Self {
        self.s.lock().unwrap().scripted.push((key, answer));
        self
    }

    fn seen(&self) -> Vec<Seen> {
        self.s.lock().unwrap().seen.clone()
    }

    fn creates(&self) -> usize {
        self.seen().iter().filter(|r| r.method == Method::Post && r.url == ALBUMS_URL).count()
    }
}

impl Transport for FakeLibrary {
    fn send(&self, req: &Request<'_>) -> Result<Response, String> {
        let mut s = self.s.lock().unwrap();
        s.seen.push(Seen {
            method: req.method,
            url: req.url.into(),
            auth: req.header("Authorization").map(str::to_string),
            content_type: req.header("Content-Type").map(str::to_string),
            body: String::from_utf8_lossy(req.body).into_owned(),
        });
        if let Some(i) = s.scripted.iter().position(|(k, _)| req.url.contains(k)) {
            return s.scripted.remove(i).1;
        }
        let body: Value = serde_json::from_slice(req.body).unwrap_or(Value::Null);
        match req.method {
            Method::Get => {
                let rest = req.url.strip_prefix(ALBUMS_URL).unwrap();
                assert!(rest.starts_with("?pageSize=50"), "{rest}");
                let page: usize = rest.split("pageToken=p").nth(1).map(|n| n.parse().unwrap()).unwrap_or(0);
                let from = page * s.per_page;
                let albums: Vec<Value> = s.albums.iter().skip(from).take(s.per_page).cloned().collect();
                let mut v = json!({});
                if !albums.is_empty() {
                    v["albums"] = json!(albums);
                }
                if from + s.per_page < s.albums.len() {
                    v["nextPageToken"] = json!(format!("p{}", page + 1));
                }
                Ok(ok(&v.to_string()))
            }
            Method::Post if req.url == ALBUMS_URL => {
                if let Some(r) = s.refuse_create.take() {
                    return Ok(r);
                }
                let title = body["album"]["title"].as_str().unwrap().to_string();
                let n = 900 + s.created;
                s.created += 1;
                let a = album(n, &title);
                s.albums.push(a.clone());
                if s.lose_create_answer {
                    s.lose_create_answer = false;
                    return Err("timed out reading the answer".into());
                }
                Ok(ok(&a.to_string()))
            }
            Method::Post if req.url.ends_with(":addEnrichment") => {
                let id = req.url.strip_prefix(&format!("{ALBUMS_URL}/")).unwrap().trim_end_matches(":addEnrichment").to_string();
                s.notes.push((id, body["newEnrichmentItem"]["textEnrichment"]["text"].as_str().unwrap().to_string()));
                Ok(ok(r#"{"enrichmentItem":{"id":"E1"}}"#))
            }
            Method::Post if req.url.ends_with(":batchAddMediaItems") => {
                if s.refuse_add_after.is_some_and(|n| s.added.len() >= n) {
                    return Ok(status(403, r#"{"error":{"message":"No permission to add media items to this album.","status":"PERMISSION_DENIED"}}"#));
                }
                let id = req.url.strip_prefix(&format!("{ALBUMS_URL}/")).unwrap().trim_end_matches(":batchAddMediaItems").to_string();
                let ids = body["mediaItemIds"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
                s.added.push((id, ids));
                Ok(ok("{}"))
            }
            Method::Post => panic!("unexpected POST {}", req.url),
        }
    }
}

fn bearer() -> Result<String, GPhotosError> {
    Ok("ya29.test".into())
}

fn calls(g: &FakeLibrary) -> Albums<'_> {
    let mut a = Albums::new(g, &bearer);
    a.backoff = Duration::ZERO;
    a
}

#[test]
fn titles_are_cleaned_and_capped_never_cut() {
    assert_eq!(clean_title("  Robots,\t2020–2026 \n").unwrap(), "Robots, 2020–2026");
    assert_eq!(clean_title("a\u{7}b\u{0}c").unwrap(), "a b c");
    for empty in ["", "   ", "\n\t\u{1b}"] {
        assert!(clean_title(empty).unwrap_err().to_string().contains("needs a title"), "{empty:?}");
    }
    let max = "é".repeat(MAX_TITLE_CHARS);
    assert_eq!(clean_title(&max).unwrap(), max, "characters, not bytes");
    let e = clean_title(&format!("{max}x")).unwrap_err().to_string();
    assert!(e.contains("501") && e.contains("500"), "{e}");
}

#[test]
fn ids_are_url_safe_or_refused() {
    let id = album_id(1);
    assert_eq!(check_id(&format!("  {id} ")).unwrap(), id);
    for bad in ["", " ", "../albums", "a b", "a?x=1", "a/b", "a#b", "a\u{0}b", "é", &"a".repeat(MAX_ID_CHARS + 1)] {
        assert!(check_id(bad).is_err(), "{bad:?}");
    }
    assert!(looks_like_id(&id));
    for title in ["Robots", "Robots 2020-2026", "AF1Qip", &"a b".repeat(30), &"a".repeat(MAX_ID_CHARS + 1)] {
        assert!(!looks_like_id(title), "{title}");
    }
}

#[test]
fn pages_parse_and_hostile_ones_are_errors_or_harmless() {
    let p = parse_page(r#"{}"#).unwrap();
    assert_eq!((p.albums.len(), p.next, p.skipped), (0, None, 0), "no albums yet");
    let p = parse_page(&json!({"albums": [album(1, "Robots"), {"title": "no id"}, {"id": "bad id/../x", "title": "x"}], "nextPageToken": "tok"}).to_string())
        .unwrap();
    assert_eq!(p.albums.len(), 1);
    assert_eq!(p.skipped, 2);
    assert_eq!(p.next.as_deref(), Some("tok"));
    let a = &p.albums[0];
    assert_eq!((a.title.as_str(), a.media_items, a.writeable), ("Robots", Some(3), true));
    assert!(a.product_url.as_deref().unwrap().starts_with("https://photos.google.com/"));
    // odd fields: wrong types, links elsewhere, control characters, huge counts
    let odd = json!({"albums": [{"id": album_id(2), "title": 5, "productUrl": "https://evil.example/x", "mediaItemsCount": "lots", "isWriteable": "yes"},
        {"id": album_id(3), "title": "Ro\u{7}bots", "productUrl": "https://photos.google.com/a b", "mediaItemsCount": 18446744073709551615u64}]});
    let p = parse_page(&odd.to_string()).unwrap();
    assert_eq!(p.albums[0], Album { id: album_id(2), title: String::new(), product_url: None, media_items: None, writeable: false });
    assert_eq!((p.albums[1].title.as_str(), p.albums[1].product_url.as_deref(), p.albums[1].media_items), ("Robots", None, Some(u64::MAX)));
    // not JSON, not an object, albums not a list, a page token that is not a token
    for bad in ["", "<html>502</html>", "[]", r#"{"albums": {"id": "x"}}"#, r#"{"albums": [], "nextPageToken": "a\u0000b"}"#] {
        assert!(matches!(parse_page(bad), Err(GPhotosError::Album(_))), "{bad:?}");
    }
    let long = json!({"nextPageToken": "t".repeat(MAX_PAGE_TOKEN_CHARS + 1)}).to_string();
    assert!(parse_page(&long).is_err());
    // a huge page keeps at most MAX_PER_PAGE albums
    let many: Vec<Value> = (0..MAX_PER_PAGE + 100).map(|i| album(i, "x")).collect();
    let p = parse_page(&json!({"albums": many}).to_string()).unwrap();
    assert_eq!((p.albums.len(), p.skipped), (MAX_PER_PAGE, 100));
    // the creation answer
    assert_eq!(parse_album(&album(4, "Robots").to_string()).unwrap().id, album_id(4));
    assert!(parse_album("{}").is_err() && parse_album("nope").is_err() && parse_album(r#"{"id":"a b"}"#).is_err());
}

#[test]
fn lists_every_page_with_the_bearer() {
    let g = FakeLibrary::new((0..5).map(|i| album(i, &format!("A{i}"))).collect());
    let listed = calls(&g).list().unwrap();
    assert!(listed.complete);
    assert_eq!(listed.albums.iter().map(|a| a.title.as_str()).collect::<Vec<_>>(), ["A0", "A1", "A2", "A3", "A4"]);
    let seen = g.seen();
    assert_eq!(seen.len(), 3, "pages of 2: {seen:?}");
    assert!(seen.iter().all(|r| r.method == Method::Get && r.auth.as_deref() == Some("Bearer ya29.test") && r.body.is_empty()));
    assert_eq!(seen[0].url, format!("{ALBUMS_URL}?pageSize=50"));
    assert_eq!(seen[1].url, format!("{ALBUMS_URL}?pageSize=50&pageToken=p1"));
    assert_eq!(listed.to_json()["albums"][2]["title"], "A2");
    // page tokens are URL-encoded
    assert_eq!(list_url(Some("a b&c=d/")), format!("{ALBUMS_URL}?pageSize=50&pageToken=a%20b%26c%3Dd%2F"));
}

#[test]
fn a_looping_or_endless_listing_ends() {
    // Google repeating a page token: stop, and say the list may be incomplete
    let looping = json!({"albums": [album(1, "x")], "nextPageToken": "same"}).to_string();
    let g = FakeLibrary::new(vec![]).script("pageSize", Ok(ok(&looping))).script("pageSize", Ok(ok(&looping))).script("pageSize", Ok(ok(&looping)));
    let listed = calls(&g).list().unwrap();
    assert!(!listed.complete);
    assert_eq!(g.seen().len(), 2, "the second page repeated the token");
    // an endless supply of albums stops at MAX_ALBUMS
    let g = FakeLibrary::new((0..MAX_ALBUMS + 7).map(|i| album(i, "x")).collect());
    g.s.lock().unwrap().per_page = 50;
    let listed = calls(&g).list().unwrap();
    assert_eq!(listed.albums.len(), MAX_ALBUMS);
    assert!(!listed.complete);
    assert!(g.seen().len() <= MAX_PAGES);
}

#[test]
fn listing_retries_blips_and_maps_errors() {
    let g = FakeLibrary::new(vec![album(1, "Robots")]).script("pageSize", Err("connection reset".into())).script("pageSize", Ok(status(503, "busy")));
    assert_eq!(calls(&g).list().unwrap().albums.len(), 1);
    assert_eq!(g.seen().len(), 3);
    // a dead connection gives up after MAX_TRIES
    let g = FakeLibrary::new(vec![]);
    for _ in 0..MAX_TRIES {
        g.s.lock().unwrap().scripted.push(("pageSize", Err("connection refused".into())));
    }
    let e = calls(&g).list().unwrap_err().to_string();
    assert!(e.contains("did not answer") && e.contains("connection refused"), "{e}");
    assert_eq!(g.seen().len() as u32, MAX_TRIES);
    // a sign-in without the read scope: the consent error, at once
    let scope = r#"{"error":{"code":403,"message":"Request had insufficient authentication scopes.","status":"PERMISSION_DENIED","details":[{"reason":"ACCESS_TOKEN_SCOPE_INSUFFICIENT"}]}}"#;
    let g = FakeLibrary::new(vec![]).script("pageSize", Ok(status(403, scope)));
    let e = calls(&g).list().unwrap_err();
    assert!(matches!(e, GPhotosError::Consent(_)) && e.to_string().contains("photos auth --albums"), "{e}");
    assert_eq!(g.seen().len(), 1);
}

#[test]
fn errors_name_their_fix() {
    let disabled = r#"{"error":{"code":403,"message":"Photos Library API has not been used in project 1 before or it is disabled.","status":"PERMISSION_DENIED","details":[{"reason":"SERVICE_DISABLED"}]}}"#;
    assert!(matches!(album_error(403, disabled), GPhotosError::Setup(m) if m.contains("Photos Library API")));
    assert!(matches!(album_error(401, "{}"), GPhotosError::Consent(m) if m.contains("--albums")));
    let denied = r#"{"error":{"code":403,"message":"No permission to add media items to this album.","status":"PERMISSION_DENIED"}}"#;
    let e = album_error(403, denied).to_string();
    assert!(e.contains("No permission") && e.contains("only media items it uploaded itself"), "{e}");
    let invalid = r#"{"error":{"code":400,"message":"Request contains an invalid media item id.","status":"INVALID_ARGUMENT"}}"#;
    let e = album_error(400, invalid).to_string();
    assert!(e.contains("invalid media item id") && e.contains("photos upload"), "{e}");
    assert!(album_error(404, r#"{"error":{"message":"Requested entity was not found.","status":"NOT_FOUND"}}"#).to_string().contains("no such album"));
    // not JSON, huge: cut short, no panic
    let e = album_error(418, &"x".repeat(100_000)).to_string();
    assert!(e.contains("418") && e.len() < 1000, "{}", e.len());
}

#[test]
fn ensure_reuses_an_album_with_that_exact_title() {
    let g = FakeLibrary::new(vec![album(1, "robots"), album(2, "Robots, 2020–2026"), album(3, "Robots, 2020–2026")]);
    let e = calls(&g).ensure("  Robots,   2020–2026 ", Some("never added to an existing album")).unwrap();
    assert_eq!((e.album.id.as_str(), e.created, e.note_added), (album_id(2).as_str(), false, false));
    assert_eq!(g.creates(), 0, "nothing created");
    assert!(g.s.lock().unwrap().notes.is_empty());
    // by id, too
    let e = calls(&g).ensure(&album_id(1), None).unwrap();
    assert_eq!(e.album.title, "robots");
    assert_eq!(e.to_json()["created"], false);
}

#[test]
fn ensure_creates_a_missing_album_with_its_note() {
    let g = FakeLibrary::new(vec![album(1, "Other")]);
    let e = calls(&g).ensure("Robot showcase", Some("  Every robot, 2020–2026  ")).unwrap();
    assert!(e.created && e.note_added && e.note_error.is_none(), "{e:?}");
    assert_eq!(e.album.title, "Robot showcase");
    let seen = g.seen();
    let create = seen.iter().find(|r| r.method == Method::Post && r.url == ALBUMS_URL).unwrap();
    assert_eq!(serde_json::from_str::<Value>(&create.body).unwrap(), json!({"album": {"title": "Robot showcase"}}));
    assert_eq!(create.content_type.as_deref(), Some("application/json"));
    let note = seen.iter().find(|r| r.url.ends_with(":addEnrichment")).unwrap();
    assert_eq!(note.url, format!("{ALBUMS_URL}/{}:addEnrichment", e.album.id));
    assert_eq!(
        serde_json::from_str::<Value>(&note.body).unwrap(),
        json!({"newEnrichmentItem": {"textEnrichment": {"text": "Every robot, 2020–2026"}}, "albumPosition": {"position": "FIRST_IN_ALBUM"}})
    );
    // run again: the album is found, not duplicated, and the note not repeated
    let again = calls(&g).ensure("Robot showcase", Some("Every robot, 2020–2026")).unwrap();
    assert_eq!((again.album.id, again.created, again.note_added), (e.album.id, false, false));
    assert_eq!(g.creates(), 1);
    assert_eq!(g.s.lock().unwrap().notes.len(), 1);
}

#[test]
fn a_lost_creation_answer_never_duplicates_the_album() {
    let g = FakeLibrary::new(vec![]);
    g.s.lock().unwrap().lose_create_answer = true;
    let e = calls(&g).ensure("Robots", None).unwrap();
    assert!(!e.created, "found on the second listing: {e:?}");
    assert_eq!(g.creates(), 1);
    assert_eq!(g.s.lock().unwrap().albums.len(), 1);
}

#[test]
fn ensure_refuses_what_it_must_not_create() {
    // an id nobody made: never becomes a title
    let g = FakeLibrary::new(vec![album(1, "Robots")]);
    let e = calls(&g).ensure(&album_id(77), None).unwrap_err().to_string();
    assert!(e.contains("no album FilmCraft made has the id"), "{e}");
    assert_eq!(g.creates(), 0);
    // a bad title or note: refused before anything is sent
    let g = FakeLibrary::new(vec![]);
    assert!(calls(&g).ensure(&"x ".repeat(MAX_TITLE_CHARS), None).unwrap_err().to_string().contains("at most 500"));
    assert!(calls(&g).ensure(" \n ", None).is_err());
    assert!(calls(&g).ensure("Robots", Some(&"n".repeat(MAX_NOTE_CHARS + 1))).is_err());
    assert!(g.seen().is_empty());
    // a listing that stopped early: the title may be on a page not read; do not create
    let looping = json!({"albums": [album(1, "x")], "nextPageToken": "same"}).to_string();
    let g = FakeLibrary::new(vec![]).script("pageSize", Ok(ok(&looping))).script("pageSize", Ok(ok(&looping)));
    let e = calls(&g).ensure("Robots", None).unwrap_err().to_string();
    assert!(e.contains("more albums than it can list"), "{e}");
    assert_eq!(g.creates(), 0);
    // Google refusing the creation: its reason, and no second try
    let g = FakeLibrary::new(vec![]);
    g.s.lock().unwrap().refuse_create = Some(status(400, r#"{"error":{"message":"Invalid title","status":"INVALID_ARGUMENT"}}"#));
    let e = calls(&g).ensure("Robots", None).unwrap_err().to_string();
    assert!(e.contains("Invalid title"), "{e}");
    assert_eq!(g.creates(), 1);
    assert!(g.s.lock().unwrap().albums.is_empty());
}

#[test]
fn a_failed_note_keeps_the_album() {
    let g = FakeLibrary::new(vec![]).script("addEnrichment", Ok(status(400, r#"{"error":{"message":"bad text","status":"INVALID_ARGUMENT"}}"#)));
    let e = calls(&g).ensure("Robots", Some("note")).unwrap();
    assert!(e.created && !e.note_added);
    assert!(e.note_error.as_deref().unwrap().contains("bad text"), "{e:?}");
    assert!(e.to_json()["noteError"].as_str().unwrap().contains("bad text"));
}

#[test]
fn adds_in_batches_of_fifty_once_each() {
    let g = FakeLibrary::new(vec![album(1, "Robots")]);
    let mut ids: Vec<String> = (0..120).map(|i| format!("AF1QipItem{i:03}")).collect();
    ids.push("AF1QipItem000".into()); // a duplicate
    let added = calls(&g).add("Robots", &ids).unwrap();
    assert_eq!((added.added, added.calls, added.album.id.as_str()), (120, 3, album_id(1).as_str()));
    let lib = g.s.lock().unwrap();
    assert_eq!(lib.added.iter().map(|(_, ids)| ids.len()).collect::<Vec<_>>(), [50, 50, 20]);
    assert!(lib.added.iter().all(|(id, _)| *id == album_id(1)));
    assert_eq!(lib.added[0].1[0], "AF1QipItem000");
    drop(lib);
    let post = g.seen().into_iter().find(|r| r.url.ends_with(":batchAddMediaItems")).unwrap();
    assert_eq!(post.url, format!("{ALBUMS_URL}/{}:batchAddMediaItems", album_id(1)));
    assert_eq!(serde_json::from_str::<Value>(&post.body).unwrap()["mediaItemIds"].as_array().unwrap().len(), 50);
    assert_eq!(added.to_json()["added"], 120);
}

#[test]
fn adds_refuse_bad_input_and_report_partial_progress() {
    let g = FakeLibrary::new(vec![album(1, "Robots")]);
    // hostile ids: refused before anything is sent
    for bad in [vec![], vec!["ok".to_string(), "a b".into()], vec!["../../x".into()], vec![String::new()]] {
        assert!(calls(&g).add("Robots", &bad).is_err(), "{bad:?}");
    }
    let huge: Vec<String> = (0..MAX_ALBUM_ITEMS + 1).map(|i| format!("i{i}")).collect();
    assert!(calls(&g).add("Robots", &huge).unwrap_err().to_string().contains("20000"));
    assert!(g.seen().is_empty());
    // no such album: listed, nothing added
    let e = calls(&g).add("Robotz", &["AF1QipItem".to_string()]).unwrap_err().to_string();
    assert!(e.contains("no album `Robotz`") && e.contains("photos album create"), "{e}");
    assert!(g.s.lock().unwrap().added.is_empty());
    // Google refusing items it did not see this app upload
    let denied = r#"{"error":{"code":400,"message":"Request contains an invalid media item id.","status":"INVALID_ARGUMENT"}}"#;
    let g = FakeLibrary::new(vec![album(1, "Robots")]).script("batchAddMediaItems", Ok(status(400, denied)));
    let e = calls(&g).add("Robots", &["AF1QipUserOwn".to_string()]).unwrap_err().to_string();
    assert!(e.contains("invalid media item id") && e.contains("uploaded itself"), "{e}");
    // a failure after the first batch says how far it got
    let g = FakeLibrary::new(vec![album(1, "Robots")]);
    g.s.lock().unwrap().refuse_add_after = Some(1);
    let ids: Vec<String> = (0..60).map(|i| format!("AF1QipItem{i}")).collect();
    let e = calls(&g).add("Robots", &ids).unwrap_err().to_string();
    assert!(e.contains("added 50 of 60") && e.contains("No permission"), "{e}");
    assert_eq!(g.s.lock().unwrap().added.len(), 1);
}

#[test]
fn notes_are_trimmed_capped_and_need_an_id() {
    assert_eq!(check_note(Some("  ")).unwrap(), None);
    assert_eq!(check_note(None).unwrap(), None);
    assert_eq!(check_note(Some(" two\nlines\u{7} ")).unwrap().as_deref(), Some("two\nlines "));
    assert!(check_note(Some(&"n".repeat(MAX_NOTE_CHARS + 1))).is_err());
    let g = FakeLibrary::new(vec![]);
    assert!(calls(&g).add_note("not/an id", "x").is_err());
    assert!(g.seen().is_empty());
    assert!(calls(&g).add_note(&album_id(1), "  ").is_ok(), "an empty note is no note");
    assert!(g.seen().is_empty());
}

#[test]
fn a_failing_bearer_sends_nothing() {
    let g = FakeLibrary::new(vec![]);
    let failing = || -> Result<String, GPhotosError> { Err(GPhotosError::Auth("refresh failed".into())) };
    let a = Albums::new(&g, &failing);
    assert!(a.list().unwrap_err().to_string().contains("refresh failed"));
    assert!(g.seen().is_empty());
}
