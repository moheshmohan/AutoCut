//! The upload protocol against a fake Google: start, chunks, resume, finalize and
//! `batchCreate`, error mapping, hostile answers, retry budgets and cancellation. Offline.

use std::collections::HashMap;
use std::sync::Mutex;

use super::*;

const SESSION: &str = "https://photoslibrary.googleapis.com/v1/uploads?upload_id=ABC123&upload_protocol=resumable";
const TOKEN: &str = "CAIS-upload-token_0123";

/// What happens to one chunk request (by its index among chunk requests).
#[derive(Clone)]
enum Fault {
    /// Lost on the way: nothing arrives, no answer.
    Drop,
    /// The connection breaks after Google kept this many bytes of the chunk.
    DropAfter(usize),
    /// Everything arrives (and a final chunk finalizes), but the answer is lost.
    LoseAnswer,
    /// Google answers with this status and body; nothing is kept.
    Status(u16, String),
}

/// One request as the fake saw it.
#[derive(Debug, Clone)]
struct Seen {
    url: String,
    command: String,
    offset: Option<u64>,
    len: usize,
    auth: Option<String>,
    headers: Vec<(String, String)>,
    body: String,
}

/// A stand-in for Google's upload and media item endpoints.
struct FakeGoogle {
    s: Mutex<State>,
}

struct State {
    start: Response,
    chunk_faults: HashMap<usize, Fault>,
    /// Applied to every chunk request without its own fault.
    every_chunk: Option<Fault>,
    query: Option<Response>,
    batch: Vec<Response>,
    chunks: usize,
    received: Vec<u8>,
    finalized: bool,
    seen: Vec<Seen>,
}

fn ok(headers: &[(&str, &str)], body: &str) -> Response {
    Response { status: 200, headers: headers.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(), body: body.into() }
}

fn status(code: u16, body: &str) -> Response {
    Response { status: code, headers: Vec::new(), body: body.into() }
}

const CREATED: &str = r#"{"newMediaItemResults":[{"uploadToken":"CAIS-upload-token_0123","status":{"message":"Success"},
    "mediaItem":{"id":"AF1Qip-new","description":"Robots","productUrl":"https://photos.google.com/lr/photo/AF1Qip-new","mimeType":"video/mp4","filename":"showcase.mp4"}}]}"#;

impl FakeGoogle {
    fn new(granularity: Option<&str>) -> Self {
        let mut h = vec![("X-Goog-Upload-URL", SESSION), ("X-Goog-Upload-Status", "active")];
        if let Some(g) = granularity {
            h.push(("X-Goog-Upload-Chunk-Granularity", g));
        }
        Self::with_start(ok(&h, ""))
    }

    fn with_start(start: Response) -> Self {
        Self {
            s: Mutex::new(State {
                start,
                chunk_faults: HashMap::new(),
                every_chunk: None,
                query: None,
                batch: Vec::new(),
                chunks: 0,
                received: Vec::new(),
                finalized: false,
                seen: Vec::new(),
            }),
        }
    }

    fn fault(self, chunk: usize, f: Fault) -> Self {
        self.s.lock().unwrap().chunk_faults.insert(chunk, f);
        self
    }

    fn seen(&self) -> Vec<Seen> {
        self.s.lock().unwrap().seen.clone()
    }

    fn commands(&self) -> Vec<String> {
        self.seen().iter().map(|r| if r.url == BATCH_CREATE_URL { "batchCreate".into() } else { r.command.clone() }).collect()
    }

    fn received(&self) -> Vec<u8> {
        self.s.lock().unwrap().received.clone()
    }
}

impl Transport for FakeGoogle {
    fn send(&self, req: &Request<'_>) -> Result<Response, String> {
        let mut s = self.s.lock().unwrap();
        let command = req.header("X-Goog-Upload-Command").unwrap_or_default().to_string();
        let offset = req.header("X-Goog-Upload-Offset").map(|o| o.parse().unwrap());
        s.seen.push(Seen {
            url: req.url.into(),
            command: command.clone(),
            offset,
            len: req.body.len(),
            auth: req.header("Authorization").map(str::to_string),
            headers: req.headers.iter().map(|(k, v)| (k.to_string(), v.clone())).collect(),
            body: String::from_utf8_lossy(req.body).into_owned(),
        });
        if req.url == UPLOADS_URL {
            assert_eq!(command, "start");
            return Ok(s.start.clone());
        }
        if req.url == BATCH_CREATE_URL {
            return Ok(if s.batch.is_empty() { ok(&[], CREATED) } else { s.batch.remove(0) });
        }
        assert_eq!(req.url, SESSION, "only the session URL gets the bytes");
        match command.as_str() {
            "query" => {
                if let Some(q) = s.query.clone() {
                    return Ok(q);
                }
                let n = s.received.len().to_string();
                Ok(if s.finalized {
                    ok(&[("X-Goog-Upload-Status", "final")], TOKEN)
                } else {
                    ok(&[("X-Goog-Upload-Status", "active"), ("X-Goog-Upload-Size-Received", &n)], "")
                })
            }
            "upload" | "upload, finalize" => {
                let i = s.chunks;
                s.chunks += 1;
                let at = offset.expect("every chunk names its offset");
                if at != s.received.len() as u64 {
                    return Ok(status(400, "offset mismatch"));
                }
                let fault = s.chunk_faults.get(&i).cloned().or_else(|| s.every_chunk.clone());
                let finalize = command == "upload, finalize";
                match fault {
                    Some(Fault::Drop) => Err("connection reset by peer".into()),
                    Some(Fault::DropAfter(n)) => {
                        s.received.extend_from_slice(&req.body[..n.min(req.body.len())]);
                        Err("connection reset by peer".into())
                    }
                    Some(Fault::Status(code, body)) => Ok(status(code, &body)),
                    Some(Fault::LoseAnswer) => {
                        s.received.extend_from_slice(req.body);
                        s.finalized |= finalize;
                        Err("timed out reading the answer".into())
                    }
                    None => {
                        s.received.extend_from_slice(req.body);
                        if finalize {
                            s.finalized = true;
                            Ok(ok(&[("X-Goog-Upload-Status", "final")], TOKEN))
                        } else {
                            Ok(ok(&[("X-Goog-Upload-Status", "active")], ""))
                        }
                    }
                }
            }
            other => panic!("unexpected command {other:?}"),
        }
    }
}

fn file(name: &str, bytes: &[u8]) -> PathBuf {
    let d = std::env::temp_dir().join(format!("gphotos-upload-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    std::fs::write(&p, bytes).unwrap();
    p
}

use std::path::PathBuf;

fn data(n: usize) -> Vec<u8> {
    (0..n).map(|i| (i * 7 + 3) as u8).collect()
}

fn opts(chunk: u64) -> UploadOptions {
    UploadOptions { description: Some("Robots".into()), file_name: None, chunk, max_retries: 3, backoff: Duration::ZERO, album: None }
}

fn bearer() -> Result<String, GPhotosError> {
    Ok("ya29.test".into())
}

/// Run an upload, recording every progress call.
fn run(g: &FakeGoogle, path: &Path, o: &UploadOptions) -> (Result<Uploaded, GPhotosError>, Vec<(u64, u64, String)>) {
    let mut calls = Vec::new();
    let r = upload_file(g, &bearer, path, o, &mut |sent, total, s| {
        calls.push((sent, total, s.to_string()));
        true
    });
    (r, calls)
}

#[test]
fn uploads_in_chunks_and_creates_the_item() {
    let bytes = data(10);
    let path = file("showcase.mp4", &bytes);
    let g = FakeGoogle::new(Some("4"));
    let (r, calls) = run(&g, &path, &opts(9));
    let up = r.unwrap();
    assert_eq!(g.received(), bytes);
    assert_eq!(g.commands(), ["start", "upload", "upload, finalize", "batchCreate"]);
    let seen = g.seen();
    // start: the resumable protocol, the type and the size, with the token
    let start = &seen[0];
    let h = |r: &Seen, k: &str| r.headers.iter().find(|(n, _)| n.eq_ignore_ascii_case(k)).map(|(_, v)| v.clone());
    assert_eq!(h(start, "X-Goog-Upload-Protocol").as_deref(), Some("resumable"));
    assert_eq!(h(start, "X-Goog-Upload-Content-Type").as_deref(), Some("video/mp4"));
    assert_eq!(h(start, "X-Goog-Upload-Raw-Size").as_deref(), Some("10"));
    assert_eq!(start.len, 0);
    assert!(seen.iter().all(|r| r.auth.as_deref() == Some("Bearer ya29.test")));
    // chunks: a multiple of the granularity (9 → 8), at their offsets, the last finalizes
    assert_eq!((seen[1].offset, seen[1].len), (Some(0), 8));
    assert_eq!((seen[2].offset, seen[2].len), (Some(8), 2));
    // batchCreate: the token, the file name and the description
    let body: Value = serde_json::from_str(&seen[3].body).unwrap();
    assert_eq!(body, json!({"newMediaItems":[{"description":"Robots","simpleMediaItem":{"uploadToken":TOKEN,"fileName":"showcase.mp4"}}]}));
    assert_eq!(h(&seen[3], "Content-Type").as_deref(), Some("application/json"));
    assert_eq!(
        up,
        Uploaded {
            media_item_id: "AF1Qip-new".into(),
            product_url: Some("https://photos.google.com/lr/photo/AF1Qip-new".into()),
            file_name: "showcase.mp4".into(),
            mime_type: "video/mp4".into(),
            bytes: 10,
            description: Some("Robots".into()),
            resumed: 0,
            album: None,
            album_created: false,
        }
    );
    assert_eq!(up.to_json()["mediaItemId"], "AF1Qip-new");
    // progress: by bytes, never backwards, ending complete
    assert!(calls.windows(2).all(|w| w[0].0 <= w[1].0), "{calls:?}");
    assert_eq!(calls.first().map(|c| (c.0, c.1)), Some((0, 10)));
    assert_eq!(calls.last().map(|c| (c.0, c.1)), Some((10, 10)));
}

#[test]
fn chunks_are_multiples_of_256_kib_by_default() {
    let bytes = data((1 << 20) + 3);
    let path = file("big.mov", &bytes);
    // no granularity header: Google's documented 256 KiB
    let g = FakeGoogle::new(None);
    run(&g, &path, &opts(300 << 10)).0.unwrap();
    assert_eq!(g.received(), bytes);
    let lens: Vec<usize> = g.seen().iter().filter(|r| r.command.starts_with("upload")).map(|r| r.len).collect();
    assert_eq!(lens, [256 << 10, 256 << 10, 256 << 10, 256 << 10, 3]);
    // the default 8 MiB chunk sends a small file whole
    let g = FakeGoogle::new(Some("262144"));
    run(&g, &path, &UploadOptions { backoff: Duration::ZERO, ..Default::default() }).0.unwrap();
    assert_eq!(g.commands(), ["start", "upload, finalize", "batchCreate"]);
    assert_eq!(chunk_size(CHUNK, GRANULARITY), 8 << 20);
}

#[test]
fn chunk_sizes_follow_the_granularity_within_bounds() {
    assert_eq!(chunk_size(CHUNK, 3 << 20), 6 << 20);
    assert_eq!(chunk_size(1, GRANULARITY), GRANULARITY, "at least one granule");
    assert_eq!(chunk_size(CHUNK, 0), CHUNK, "a zero granularity is one byte");
    assert_eq!(chunk_size(u64::MAX, 1), MAX_CHUNK);
    assert_eq!(chunk_size(u64::MAX, u64::MAX), MAX_CHUNK);
    assert_eq!(chunk_size(0, 0), 1);
}

#[test]
fn resumes_after_a_dropped_chunk() {
    let bytes = data(20);
    let path = file("drop.mp4", &bytes);
    // lost entirely: Google has 8, the upload resends from 8
    let g = FakeGoogle::new(Some("4")).fault(1, Fault::Drop);
    let up = run(&g, &path, &opts(8)).0.unwrap();
    assert_eq!(g.received(), bytes);
    assert_eq!(up.resumed, 1);
    assert_eq!(g.commands(), ["start", "upload", "upload", "query", "upload", "upload, finalize", "batchCreate"]);
    let offsets: Vec<Option<u64>> = g.seen().iter().filter(|r| r.command.starts_with("upload")).map(|r| r.offset).collect();
    assert_eq!(offsets, [Some(0), Some(8), Some(8), Some(16)]);

    // half kept: Google has 8 + 3, the upload goes on from 11
    let g = FakeGoogle::new(Some("4")).fault(1, Fault::DropAfter(3));
    run(&g, &path, &opts(8)).0.unwrap();
    assert_eq!(g.received(), bytes);
    let offsets: Vec<Option<u64>> = g.seen().iter().filter(|r| r.command.starts_with("upload")).map(|r| r.offset).collect();
    assert_eq!(offsets, [Some(0), Some(8), Some(11), Some(19)]);

    // a 503 is retried the same way
    let g = FakeGoogle::new(Some("4")).fault(0, Fault::Status(503, "backend error".into()));
    run(&g, &path, &opts(8)).0.unwrap();
    assert_eq!(g.received(), bytes);

    // the final chunk arrived but its answer did not: the query hands over the token
    let g = FakeGoogle::new(Some("4")).fault(2, Fault::LoseAnswer);
    let up = run(&g, &path, &opts(8)).0.unwrap();
    assert_eq!(g.commands(), ["start", "upload", "upload", "upload, finalize", "query", "batchCreate"]);
    assert_eq!((up.media_item_id.as_str(), up.resumed), ("AF1Qip-new", 1));
}

#[test]
fn retries_are_capped() {
    let path = file("flaky.mp4", &data(20));
    let g = FakeGoogle::new(Some("4"));
    g.s.lock().unwrap().every_chunk = Some(Fault::Drop);
    let e = run(&g, &path, &opts(8)).0.unwrap_err().to_string();
    assert!(e.contains("kept failing") && e.contains("connection"), "{e}");
    let chunk_tries = g.commands().iter().filter(|c| c.starts_with("upload")).count();
    assert_eq!(chunk_tries, 4, "the first try and three retries: {:?}", g.commands());
    assert!(!g.commands().contains(&"batchCreate".to_string()));

    // progress that comes and goes cannot loop forever: the total budget ends it
    let path = file("lossy.mp4", &data(64));
    let g = FakeGoogle::new(Some("4"));
    for i in (1..200).step_by(2) {
        g.s.lock().unwrap().chunk_faults.insert(i, Fault::Drop);
    }
    let e = run(&g, &path, &opts(4)).0.unwrap_err().to_string();
    assert!(e.contains("kept failing") || e.contains("did not finish"), "{e}");
    assert!(g.seen().len() < 100, "{}", g.seen().len());
}

#[test]
fn cancel_stops_the_upload() {
    let path = file("cancel.mp4", &data(20));
    // between chunks: nothing more is sent, nothing is finalized
    let g = FakeGoogle::new(Some("4"));
    let mut n = 0;
    let r = upload_file(&g, &bearer, &path, &opts(8), &mut |sent, _, _| {
        n += 1;
        sent < 8
    });
    assert!(matches!(r, Err(GPhotosError::Cancelled)), "{r:?}");
    assert_eq!(g.commands(), ["start", "upload"]);
    // before anything is sent
    let g = FakeGoogle::new(Some("4"));
    let r = upload_file(&g, &bearer, &path, &opts(8), &mut |_, _, _| false);
    assert!(matches!(r, Err(GPhotosError::Cancelled)));
    assert!(g.seen().is_empty());
    // while waiting to retry: a long backoff ends at once
    let g = FakeGoogle::new(Some("4")).fault(0, Fault::Drop);
    let o = UploadOptions { backoff: Duration::from_secs(30), ..opts(8) };
    let t0 = std::time::Instant::now();
    let r = upload_file(&g, &bearer, &path, &o, &mut |_, _, s| !s.starts_with("Retrying"));
    assert!(matches!(r, Err(GPhotosError::Cancelled)), "{r:?}");
    assert!(t0.elapsed() < Duration::from_secs(5));
    assert_eq!(n, 3, "start, chunk one, chunk two");
}

#[test]
fn errors_name_the_fix() {
    let disabled = r#"{"error":{"code":403,"message":"Photos Library API has not been used in project 1234 before or it is disabled. Enable it by visiting https://console.developers.google.com/apis/api/photoslibrary.googleapis.com/overview?project=1234 then retry.","status":"PERMISSION_DENIED","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"SERVICE_DISABLED"}]}}"#;
    let e = upload_error(403, disabled);
    assert!(matches!(e, GPhotosError::Setup(_)), "{e:?}");
    let msg = e.to_string();
    assert!(msg.contains("Photos Library API") && msg.contains("Enable"), "{msg}");

    let scopes = r#"{"error":{"code":403,"message":"Request had insufficient authentication scopes.","status":"PERMISSION_DENIED","details":[{"reason":"ACCESS_TOKEN_SCOPE_INSUFFICIENT"}]}}"#;
    for e in [upload_error(403, scopes), upload_error(401, "Unauthorized"), upload_error(400, r#"{"error":{"status":"UNAUTHENTICATED","message":"x"}}"#)] {
        assert!(matches!(e, GPhotosError::Consent(_)), "{e:?}");
        assert!(e.to_string().contains("filmcraft-cli photos auth --upload"), "{e}");
    }
    assert!(upload_error(413, "").to_string().contains("too large"));
    assert!(upload_error(404, "Not Found").to_string().contains("upload again"));
    let quota = upload_error(400, r#"{"error":{"code":400,"message":"Invalid upload token","status":"INVALID_ARGUMENT"}}"#).to_string();
    assert_eq!(quota, "upload: Google answered 400 INVALID_ARGUMENT: Invalid upload token");
    assert!(upload_error(500, &"é".repeat(5000)).to_string().chars().count() < 400, "long bodies are cut at a char boundary");

    // end to end: the API not enabled stops at `start`, nothing else is sent
    let path = file("disabled.mp4", &data(10));
    let g = FakeGoogle::with_start(status(403, disabled));
    let e = run(&g, &path, &opts(8)).0.unwrap_err();
    assert!(matches!(e, GPhotosError::Setup(_)), "{e:?}");
    assert_eq!(g.commands(), ["start"]);
    // a sign-in without the upload scope
    let g = FakeGoogle::with_start(status(403, scopes));
    assert!(matches!(run(&g, &path, &opts(8)).0, Err(GPhotosError::Consent(_))));
    // a chunk refused outright is not retried
    let g = FakeGoogle::new(Some("4")).fault(0, Fault::Status(400, "Bad chunk".into()));
    let e = run(&g, &path, &opts(8)).0.unwrap_err().to_string();
    assert!(e.contains("400") && e.contains("Bad chunk"), "{e}");
    assert_eq!(g.commands(), ["start", "upload"]);
}

#[test]
fn hostile_session_answers_are_refused() {
    let with = |h: &[(&str, &str)]| parse_session(&ok(h, ""));
    let s = with(&[("x-goog-upload-url", SESSION), ("X-GOOG-UPLOAD-CHUNK-GRANULARITY", "262144")]).unwrap();
    assert_eq!(s, UploadSession { url: SESSION.into(), granularity: GRANULARITY }, "headers match in any case");
    assert!(with(&[]).is_err(), "no session URL");
    assert!(with(&[("X-Goog-Upload-URL", "")]).is_err());
    for url in [
        "http://photoslibrary.googleapis.com/v1/uploads?upload_id=1",
        "https://evil.example/v1/uploads",
        "https://photoslibrary.googleapis.com.evil.example/x",
        "https://evil.example@photoslibrary.googleapis.com/x",
        "https://photoslibrary.googleapis.com:8443/x",
        "https://.googleapis.com/x",
        "https://a..googleapis.com/x",
        "https://photoslibrary.googleapis.com/x y",
        "https://photoslibrary.googleapis.com/\nx",
        "file:///etc/passwd",
        "-a Calculator",
    ] {
        assert!(check_upload_url(url).is_err(), "{url}");
    }
    assert!(check_upload_url(&format!("https://photoslibrary.googleapis.com/{}", "a".repeat(5000))).is_err(), "absurdly long");
    assert!(check_upload_url("https://uploads.googleapis.com/upload/x?id=1").is_ok());
    // granularity: unreadable or zero is Google's default, absurd is refused
    for g in ["0", "abc", "-4", "1.5", "", "18446744073709551616"] {
        assert_eq!(with(&[("X-Goog-Upload-URL", SESSION), ("X-Goog-Upload-Chunk-Granularity", g)]).unwrap().granularity, GRANULARITY, "{g:?}");
    }
    let e = with(&[("X-Goog-Upload-URL", SESSION), ("X-Goog-Upload-Chunk-Granularity", "1099511627776")]).unwrap_err().to_string();
    assert!(e.contains("1099511627776"), "{e}");
    assert_eq!(with(&[("X-Goog-Upload-URL", SESSION), ("X-Goog-Upload-Chunk-Granularity", "1")]).unwrap().granularity, 1);

    // end to end: a hostile session URL never gets the bytes (or the token)
    let path = file("hostile.mp4", &data(10));
    let g = FakeGoogle::with_start(ok(&[("X-Goog-Upload-URL", "https://evil.example/steal")], ""));
    let e = run(&g, &path, &opts(8)).0.unwrap_err().to_string();
    assert!(e.contains("not a Google API"), "{e}");
    assert_eq!(g.commands(), ["start"]);
    // a huge granularity allocates nothing: refused before the first chunk
    let g = FakeGoogle::with_start(ok(&[("X-Goog-Upload-URL", SESSION), ("X-Goog-Upload-Chunk-Granularity", "999999999999999")], ""));
    assert!(run(&g, &path, &opts(8)).0.is_err());
    assert_eq!(g.commands(), ["start"]);
}

#[test]
fn hostile_query_answers_are_refused() {
    let q = |h: &[(&str, &str)], body: &str| parse_query(&ok(h, body), 100, 50);
    assert_eq!(q(&[("X-Goog-Upload-Status", "active"), ("X-Goog-Upload-Size-Received", "48")], "").unwrap(), Received::Active(48));
    assert_eq!(q(&[("X-Goog-Upload-Status", "ACTIVE"), ("X-Goog-Upload-Size-Received", " 0 ")], "").unwrap(), Received::Active(0));
    assert_eq!(q(&[("X-Goog-Upload-Status", "final")], " tok \n").unwrap(), Received::Final("tok".into()));
    for (h, body) in [
        (vec![("X-Goog-Upload-Status", "active"), ("X-Goog-Upload-Size-Received", "51")], ""),
        (vec![("X-Goog-Upload-Status", "active"), ("X-Goog-Upload-Size-Received", "18446744073709551615")], ""),
        (vec![("X-Goog-Upload-Status", "active"), ("X-Goog-Upload-Size-Received", "-1")], ""),
        (vec![("X-Goog-Upload-Status", "active"), ("X-Goog-Upload-Size-Received", "lots")], ""),
        (vec![("X-Goog-Upload-Status", "active")], ""),
        (vec![("X-Goog-Upload-Status", "cancelled")], ""),
        (vec![("X-Goog-Upload-Status", "final")], ""),
        (vec![], ""),
    ] {
        assert!(q(&h, body).is_err(), "{h:?}");
    }
    // end to end: Google claiming more than was sent stops the upload
    let path = file("liar.mp4", &data(20));
    let g = FakeGoogle::new(Some("4")).fault(1, Fault::Drop);
    g.s.lock().unwrap().query = Some(ok(&[("X-Goog-Upload-Status", "active"), ("X-Goog-Upload-Size-Received", "999")], ""));
    let e = run(&g, &path, &opts(8)).0.unwrap_err().to_string();
    assert!(e.contains("999"), "{e}");
    // tokens
    assert_eq!(parse_token("  CAIS_x-1 \r\n").unwrap(), "CAIS_x-1");
    for bad in ["", "   ", "two words", "tab\there", "nul\0", "é", &"x".repeat(9000)] {
        assert!(parse_token(bad).is_err(), "{bad:?}");
    }
}

#[test]
fn batch_create_answers_are_checked() {
    let item = parse_batch_create(CREATED).unwrap();
    assert_eq!(
        item,
        CreatedItem { id: "AF1Qip-new".into(), product_url: Some("https://photos.google.com/lr/photo/AF1Qip-new".into()), filename: "showcase.mp4".into() }
    );
    let failed =
        r#"{"newMediaItemResults":[{"uploadToken":"t","status":{"code":3,"message":"Failed: There was an error while trying to create this media item."}}]}"#;
    let e = parse_batch_create(failed).unwrap_err().to_string();
    assert!(e.contains("did not add") && e.contains("error while trying"), "{e}");
    for bad in [
        "",
        "[]",
        "{}",
        r#"{"newMediaItemResults":[]}"#,
        r#"{"newMediaItemResults":[{"status":{"message":"OK"}}]}"#,
        r#"{"newMediaItemResults":[{"mediaItem":{"id":""}}]}"#,
        r#"{"newMediaItemResults":{}}"#,
    ] {
        assert!(parse_batch_create(bad).is_err(), "{bad}");
    }
    // a product link that is not Google Photos' is dropped, the item kept
    for url in ["javascript:alert(1)", "http://photos.google.com/x", "https://evil.example/x", "https://photos.google.com/a b"] {
        let t = format!(r#"{{"newMediaItemResults":[{{"status":{{"message":"OK"}},"mediaItem":{{"id":"X","productUrl":"{url}"}}}}]}}"#);
        assert_eq!(parse_batch_create(&t).unwrap().product_url, None, "{url}");
    }
    // truncations of a real answer never panic
    for n in 0..CREATED.len() {
        let _ = parse_batch_create(CREATED.get(..n).unwrap_or_default());
    }
    // a 500 on batchCreate is retried; the item still lands once
    let path = file("retry-create.mp4", &data(10));
    let g = FakeGoogle::new(Some("4"));
    g.s.lock().unwrap().batch = vec![status(500, "oops")];
    assert_eq!(run(&g, &path, &opts(8)).0.unwrap().media_item_id, "AF1Qip-new");
    assert_eq!(g.commands().iter().filter(|c| *c == "batchCreate").count(), 2);
    // the item's own failure is an error
    let g = FakeGoogle::new(Some("4"));
    g.s.lock().unwrap().batch = vec![ok(&[], failed)];
    assert!(run(&g, &path, &opts(8)).0.unwrap_err().to_string().contains("did not add"));
}

#[test]
fn files_are_checked_before_anything_is_sent() {
    let g = FakeGoogle::new(Some("4"));
    let dir = std::env::temp_dir().join(format!("gphotos-upload-{}", std::process::id()));
    for (path, want) in [
        (file("mix.wav", b"RIFF"), "does not take a .wav"),
        (file("master.mxf", b"x"), "does not take a .mxf"),
        (file("noext", b"x"), "without an extension"),
        (file("empty.mp4", b""), "is empty"),
        (dir.join("missing.mp4"), "missing.mp4"),
    ] {
        let e = run(&g, &path, &opts(8)).0.unwrap_err().to_string();
        assert!(e.contains(want), "{}: {e}", path.display());
    }
    let sub = dir.join("folder.mp4");
    std::fs::create_dir_all(&sub).unwrap();
    assert!(run(&g, &sub, &opts(8)).0.unwrap_err().to_string().contains("not a file"));
    // sizes: sparse files just over Google's limits
    for (name, limit) in [("huge.jpg", MAX_PHOTO_BYTES), ("huge.mp4", MAX_VIDEO_BYTES)] {
        let p = dir.join(name);
        std::fs::File::create(&p).unwrap().set_len(limit + 1).unwrap();
        let e = run(&g, &p, &opts(8)).0.unwrap_err().to_string();
        assert!(e.contains("too large"), "{e}");
        std::fs::File::create(&p).unwrap().set_len(limit).unwrap();
        assert!(check_file(&p).is_ok(), "{name} at the limit");
        std::fs::remove_file(&p).ok();
    }
    let ok_file = file("fine.MP4", b"0123456789");
    let long = UploadOptions { description: Some("x".repeat(MAX_DESCRIPTION_CHARS + 1)), ..opts(8) };
    assert!(run(&g, &ok_file, &long).0.unwrap_err().to_string().contains("at most 1000"));
    assert!(g.seen().is_empty(), "nothing reached Google: {:?}", g.commands());
    // types are matched by extension in any case; the MIME type follows
    assert_eq!(media_type(Path::new("a/B.MOV")).unwrap(), ("video/quicktime", Kind::Video));
    assert_eq!(media_type(Path::new("x.heic")).unwrap(), ("image/heic", Kind::Photo));
    // a file name override is cleaned; a blank one falls back to the file's own
    let g = FakeGoogle::new(Some("4"));
    let o = UploadOptions { file_name: Some("Robot\u{7}s 2026.mp4".into()), ..opts(8) };
    run(&g, &ok_file, &o).0.unwrap();
    let body: Value = serde_json::from_str(&g.seen().last().unwrap().body).unwrap();
    assert_eq!(body["newMediaItems"][0]["simpleMediaItem"]["fileName"], "Robots 2026.mp4");
    let g = FakeGoogle::new(Some("4"));
    run(&g, &ok_file, &UploadOptions { file_name: Some(" \n ".into()), description: None, ..opts(8) }).0.unwrap();
    let body: Value = serde_json::from_str(&g.seen().last().unwrap().body).unwrap();
    assert_eq!(body["newMediaItems"][0]["simpleMediaItem"]["fileName"], "fine.MP4");
    assert!(body["newMediaItems"][0].get("description").is_none(), "{body}");
}

#[test]
fn an_album_id_goes_into_batch_create() {
    let path = file("into-album.mp4", &data(10));
    let g = FakeGoogle::new(Some("4"));
    let album = "AF1QipAlbum000000000000000000000000000000000001";
    let up = run(&g, &path, &UploadOptions { album: Some(format!(" {album} ")), ..opts(8) }).0.unwrap();
    let body: Value = serde_json::from_str(&g.seen().last().unwrap().body).unwrap();
    assert_eq!(body["albumId"], album, "{body}");
    assert_eq!(body["newMediaItems"][0]["simpleMediaItem"]["uploadToken"], TOKEN);
    assert_eq!(up.album.as_ref().map(|a| a.id.as_str()), Some(album));
    assert!(!up.album_created);
    assert_eq!(up.to_json()["album"]["id"], album);
    // without one: no albumId at all
    let g = FakeGoogle::new(Some("4"));
    let up = run(&g, &path, &opts(8)).0.unwrap();
    let body: Value = serde_json::from_str(&g.seen().last().unwrap().body).unwrap();
    assert!(body.get("albumId").is_none(), "{body}");
    assert!(up.album.is_none() && up.to_json()["album"].is_null());
    // a hostile album id never reaches Google (it would end up in the request)
    for bad in ["a b", "x\"}],\"evil\":{\"", "../albums", ""] {
        let g = FakeGoogle::new(Some("4"));
        assert!(run(&g, &path, &UploadOptions { album: Some(bad.into()), ..opts(8) }).0.is_err(), "{bad:?}");
        assert!(g.seen().is_empty(), "{bad:?}");
    }
    assert_eq!(
        batch_create_body("t", "f.mp4", None, Some("A1")),
        json!({"newMediaItems":[{"simpleMediaItem":{"uploadToken":"t","fileName":"f.mp4"}}],"albumId":"A1"})
    );
}

#[test]
fn a_failing_token_source_stops_the_upload() {
    let path = file("token.mp4", &data(10));
    let g = FakeGoogle::new(Some("4"));
    let r = upload_file(&g, &|| Err(GPhotosError::Auth("the refresh failed".into())), &path, &opts(8), &mut |_, _, _| true);
    assert!(r.unwrap_err().to_string().contains("refresh failed"));
    assert!(g.seen().is_empty());
}
