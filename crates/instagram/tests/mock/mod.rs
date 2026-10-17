//! A stand-in for Instagram on 127.0.0.1: the token endpoints (code exchange, long-lived
//! exchange, refresh), `/me`, Reel containers, the resumable upload on the "rupload" path,
//! container status, `media_publish`, `content_publishing_limit` and permalinks — with scripted
//! faults (dropped uploads, error answers, odd statuses, hostile bodies). One request per
//! connection (`Connection: close`).

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::json;

pub const SECRET: &str = "0123456789abcdef0123456789abcdef";
pub const USER_ID: &str = "17841400000000001";

/// One request as the mock saw it.
#[derive(Debug, Clone)]
pub struct Seen {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    /// The body (uploads: only how many bytes came).
    pub body: Vec<u8>,
    pub body_len: usize,
}

impl Seen {
    pub fn header(&self, k: &str) -> Option<&str> {
        self.headers.iter().find(|(h, _)| h.eq_ignore_ascii_case(k)).map(|(_, v)| v.as_str())
    }
    pub fn form(&self, k: &str) -> Option<String> {
        String::from_utf8_lossy(&self.body).split('&').find_map(|p| {
            let (a, b) = p.split_once('=')?;
            (a == k).then(|| filmcraft_oauth::urldecode(b))
        })
    }
    pub fn query(&self, k: &str) -> Option<String> {
        let q = self.path.split_once('?')?.1;
        q.split('&').find_map(|p| {
            let (a, b) = p.split_once('=')?;
            (a == k).then(|| filmcraft_oauth::urldecode(b))
        })
    }
}

#[derive(Debug, Clone, Default)]
pub struct Container {
    pub resumable: bool,
    pub form: Vec<(String, String)>,
    pub received: Vec<u8>,
    pub size: u64,
    pub published: bool,
}

#[derive(Default)]
pub struct State {
    /// Tokens the mock accepts, and the next number.
    pub tokens: Vec<String>,
    pub issued: u32,
    /// The refresh is refused with this error body (status 400).
    pub refresh_refused: Option<String>,
    /// `/me` answers this instead.
    pub me_answer: Option<(u16, String)>,
    /// `/media` answers these before making a container.
    pub create_answers: VecDeque<(u16, String)>,
    /// The container answer's `uri` instead of the mock's own.
    pub uri: Option<String>,
    pub containers: HashMap<String, Container>,
    pub next_container: u64,
    /// What each upload POST does, by its index: keep this many bytes, then drop the connection.
    pub upload_drops: HashMap<usize, usize>,
    /// Upload answers instead of success (by index).
    pub upload_answers: HashMap<usize, (u16, String)>,
    pub uploads: usize,
    /// Statuses to report (one per look) once the bytes are in; then FINISHED.
    pub statuses: VecDeque<(String, String)>,
    /// Report IN_PROGRESS forever.
    pub never_finish: bool,
    /// `video_status` is an unknown field (as it may be on graph.instagram.com).
    pub no_video_status: bool,
    /// `/media_publish` answers this instead.
    pub publish_answer: Option<(u16, String)>,
    /// `content_publishing_limit`'s `(usage, total)`.
    pub limit: (u64, u64),
    pub limit_answer: Option<(u16, String)>,
    pub published: Vec<String>,
    pub seen: Vec<Seen>,
}

pub struct Mock {
    pub base: String,
    pub state: Arc<Mutex<State>>,
}

impl Mock {
    pub fn start() -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(State { next_container: 17889615691921000, limit: (0, 100), ..Default::default() }));
        let s = state.clone();
        let b = base.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let s = s.clone();
                let b = b.clone();
                std::thread::spawn(move || handle(stream, &s, &b));
            }
        });
        Self { base, state }
    }

    pub fn with<T>(&self, f: impl FnOnce(&mut State) -> T) -> T {
        f(&mut self.state.lock().unwrap())
    }

    pub fn seen(&self, method: &str, prefix: &str) -> Vec<Seen> {
        self.with(|s| s.seen.iter().filter(|r| r.method == method && r.path.starts_with(prefix)).cloned().collect())
    }

    /// A token the mock accepts (as if generated in the App Dashboard).
    pub fn dashboard_token(&self) -> String {
        self.with(|s| {
            s.issued += 1;
            let t = format!("IGAAdashboard0000000000{}", s.issued);
            s.tokens.push(t.clone());
            t
        })
    }

    /// The stand-in browser: "allows" the app and follows the redirect with a code.
    pub fn browser(&self) -> impl Fn(&str) -> bool + Send + Sync + 'static {
        move |url: &str| {
            let q = url.split_once('?').map(|(_, q)| q).unwrap_or_default();
            let get = |k: &str| q.split('&').find_map(|p| p.split_once('=').filter(|(a, _)| *a == k).map(|(_, v)| filmcraft_oauth::urldecode(v)));
            let (Some(redirect), Some(st)) = (get("redirect_uri"), get("state")) else { return false };
            if !redirect.starts_with("http://") {
                return true; // pasted back by the test
            }
            let target = format!("{redirect}?code=AQDcode123&state={}#_", filmcraft_oauth::urlencode(&st));
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(50));
                let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(Duration::from_secs(10))).build().into();
                let _ = agent.get(target.split('#').next().unwrap_or_default()).call();
            });
            true
        }
    }
}

const MAX_BODY: usize = 64 << 20;

fn read_head(reader: &mut BufReader<TcpStream>) -> Option<(String, String, Vec<(String, String)>, usize)> {
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next()?.to_string(), parts.next()?.to_string());
    let mut headers = Vec::new();
    let mut length = 0usize;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).ok()? == 0 || h == "\r\n" {
            break;
        }
        let (k, v) = h.split_once(':')?;
        let (k, v) = (k.trim().to_string(), v.trim().to_string());
        if k.eq_ignore_ascii_case("content-length") {
            length = v.parse().ok()?;
        }
        headers.push((k, v));
    }
    Some((method, path, headers, length.min(MAX_BODY)))
}

fn respond(mut stream: &TcpStream, status: u16, headers: &[(&str, &str)], body: &str) {
    let mut head = format!("HTTP/1.1 {status} X\r\nContent-Length: {}\r\nConnection: close\r\nContent-Type: application/json\r\n", body.len());
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

pub fn graph_error(code: i64, subcode: i64, message: &str) -> String {
    json!({"error": {"message": message, "type": "OAuthException", "code": code, "error_subcode": subcode, "fbtrace_id": "x"}}).to_string()
}

fn bearer_ok(s: &State, seen: &Seen) -> bool {
    seen.header("authorization").and_then(|a| a.strip_prefix("Bearer ")).is_some_and(|t| s.tokens.iter().any(|k| k == t))
}

fn handle(stream: TcpStream, state: &Arc<Mutex<State>>, base: &str) {
    let Ok(clone) = stream.try_clone() else { return };
    let mut reader = BufReader::new(clone);
    let Some((method, path, headers, length)) = read_head(&mut reader) else { return };
    let route = path.split('?').next().unwrap_or_default().to_string();
    // uploads: read as told, maybe dropping the connection part way
    if method == "POST" && route.starts_with("/ig-api-upload/") {
        let container = route.rsplit('/').next().unwrap_or_default().to_string();
        let (_index, drop_after, answer) = {
            let mut s = state.lock().unwrap();
            let i = s.uploads;
            s.uploads += 1;
            (i, s.upload_drops.get(&i).copied(), s.upload_answers.get(&i).cloned())
        };
        let want = drop_after.unwrap_or(length).min(length);
        let mut body = Vec::with_capacity(want);
        // what came before the client stopped (a cancel) is kept, as Instagram keeps it
        let _ = (&mut reader).take(want as u64).read_to_end(&mut body);
        let complete = body.len() == want;
        let seen = Seen { method: method.clone(), path: path.clone(), headers: headers.clone(), body: Vec::new(), body_len: body.len() };
        let mut s = state.lock().unwrap();
        s.seen.push(seen.clone());
        let auth_ok = seen.header("authorization").and_then(|a| a.strip_prefix("OAuth ")).is_some_and(|t| s.tokens.iter().any(|k| k == t));
        let offset: usize = seen.header("offset").and_then(|o| o.parse().ok()).unwrap_or(usize::MAX);
        let size: u64 = seen.header("file_size").and_then(|o| o.parse().ok()).unwrap_or(0);
        let Some(c) = s.containers.get_mut(&container) else {
            return respond(&stream, 400, &[], r#"{"debug_info":{"retriable":false,"type":"NotFound","message":"no such container"}}"#);
        };
        if !auth_ok {
            return respond(&stream, 400, &[], r#"{"debug_info":{"retriable":false,"type":"ProcessingFailedError","message":"unauthorized user request"}}"#);
        }
        if offset > c.received.len() {
            return respond(&stream, 400, &[], r#"{"debug_info":{"retriable":false,"type":"BadOffset","message":"offset past what was received"}}"#);
        }
        c.size = size;
        c.received.truncate(offset);
        c.received.extend_from_slice(&body);
        if drop_after.is_some() || !complete {
            return; // the connection breaks: no answer
        }
        if let Some((st, b)) = answer {
            return respond(&stream, st, &[], &b);
        }
        return respond(&stream, 200, &[], r#"{"success":true,"message":"Upload successful."}"#);
    }
    let mut body = vec![0u8; length];
    if reader.read_exact(&mut body).is_err() {
        return;
    }
    let seen = Seen { method: method.clone(), path: path.clone(), headers, body: body.clone(), body_len: body.len() };
    let mut s = state.lock().unwrap();
    s.seen.push(seen.clone());
    let v = "/v25.0/";
    match (method.as_str(), route.as_str()) {
        ("POST", "/oauth/access_token") => {
            if seen.form("client_secret").as_deref() != Some(SECRET) {
                return respond(&stream, 400, &[], r#"{"error_type":"OAuthException","code":400,"error_message":"Error validating client secret."}"#);
            }
            if seen.form("code").as_deref() != Some("AQDcode123") {
                return respond(&stream, 400, &[], r#"{"error_type":"OAuthException","code":400,"error_message":"This authorization code has been used"}"#);
            }
            s.issued += 1;
            let t = format!("IGAAshort00000000000000{}", s.issued);
            s.tokens.push(t.clone());
            let ans = json!({"data": [{"access_token": t, "user_id": USER_ID, "permissions": "instagram_business_basic,instagram_business_content_publish"}]});
            respond(&stream, 200, &[], &ans.to_string())
        }
        ("GET", "/access_token") => {
            let ok = seen.query("grant_type").as_deref() == Some("ig_exchange_token")
                && seen.query("client_secret").as_deref() == Some(SECRET)
                && seen.query("access_token").is_some_and(|t| s.tokens.contains(&t));
            if !ok {
                return respond(&stream, 400, &[], &graph_error(190, 0, "Invalid OAuth access token"));
            }
            s.issued += 1;
            let t = format!("IGAAlong000000000000000{}", s.issued);
            s.tokens.push(t.clone());
            respond(&stream, 200, &[], &json!({"access_token": t, "token_type": "bearer", "expires_in": 5183944}).to_string())
        }
        ("GET", "/refresh_access_token") => {
            if let Some(b) = s.refresh_refused.clone() {
                return respond(&stream, 400, &[], &b);
            }
            if !seen.query("access_token").is_some_and(|t| s.tokens.contains(&t)) {
                return respond(&stream, 400, &[], &graph_error(190, 0, "Error validating access token: Session has expired"));
            }
            s.issued += 1;
            let t = format!("IGAArefreshed0000000000{}", s.issued);
            s.tokens.push(t.clone());
            respond(&stream, 200, &[], &json!({"access_token": t, "token_type": "bearer", "expires_in": 5184000}).to_string())
        }
        _ if route.starts_with(v) => {
            if !bearer_ok(&s, &seen) {
                return respond(&stream, 400, &[], &graph_error(190, 463, "Error validating access token: Session has expired"));
            }
            let rest = route.strip_prefix(v).unwrap_or_default().to_string();
            graph(&stream, &mut s, &method, &rest, &seen, base);
        }
        _ => respond(&stream, 404, &[], "{}"),
    }
}

fn graph(stream: &TcpStream, s: &mut State, method: &str, rest: &str, seen: &Seen, base: &str) {
    let fields = seen.query("fields").unwrap_or_default();
    match (method, rest) {
        ("GET", "me") => match s.me_answer.clone() {
            Some((st, b)) => respond(stream, st, &[], &b),
            None => respond(
                stream,
                200,
                &[],
                &json!({"id": "999", "user_id": USER_ID, "username": "moheshmohan", "name": "Mohesh", "account_type": "MEDIA_CREATOR"}).to_string(),
            ),
        },
        ("POST", r) if r == format!("{USER_ID}/media") => {
            if let Some((st, b)) = s.create_answers.pop_front() {
                return respond(stream, st, &[], &b);
            }
            s.next_container += 1;
            let id = s.next_container.to_string();
            let form: Vec<(String, String)> = String::from_utf8_lossy(&seen.body)
                .split('&')
                .filter_map(|p| p.split_once('=').map(|(a, b)| (a.to_string(), filmcraft_oauth::urldecode(b))))
                .collect();
            let resumable = seen.form("upload_type").as_deref() == Some("resumable");
            let mut c = Container { resumable, form, ..Default::default() };
            if !resumable {
                c.received = b"fetched".to_vec();
                c.size = 7;
            }
            s.containers.insert(id.clone(), c);
            let uri = s.uri.clone().unwrap_or_else(|| format!("{base}/ig-api-upload/v25.0/{id}"));
            let ans = if resumable { json!({"id": id, "uri": uri}) } else { json!({"id": id}) };
            respond(stream, 200, &[], &ans.to_string())
        }
        ("POST", r) if r == format!("{USER_ID}/media_publish") => {
            if let Some((st, b)) = s.publish_answer.clone() {
                return respond(stream, st, &[], &b);
            }
            let id = seen.form("creation_id").unwrap_or_default();
            let Some(c) = s.containers.get_mut(&id) else { return respond(stream, 400, &[], &graph_error(24, 2207008, "container missing")) };
            if c.published {
                return respond(stream, 400, &[], &graph_error(24, 2207008, "already published"));
            }
            c.published = true;
            let media = format!("1800000000000{}", s.published.len() + 1);
            s.published.push(media.clone());
            respond(stream, 200, &[], &json!({"id": media}).to_string())
        }
        ("GET", r) if r == format!("{USER_ID}/content_publishing_limit") => {
            if let Some((st, b)) = s.limit_answer.clone() {
                return respond(stream, st, &[], &b);
            }
            let (u, t) = s.limit;
            respond(stream, 200, &[], &json!({"data": [{"quota_usage": u, "config": {"quota_total": t, "quota_duration": 86400}}]}).to_string())
        }
        ("GET", id) if s.containers.contains_key(id) => {
            if fields.contains("video_status") && s.no_video_status {
                return respond(stream, 400, &[], &graph_error(100, 0, "Tried accessing nonexisting field (video_status)"));
            }
            let (received, size, resumable, published) = {
                let c = &s.containers[id];
                (c.received.len() as u64, c.size, c.resumable, c.published)
            };
            let complete = size > 0 && received >= size;
            let (code, status) = if published {
                ("PUBLISHED".to_string(), String::new())
            } else if !complete && resumable {
                ("IN_PROGRESS".to_string(), "uploading".to_string())
            } else if s.never_finish {
                ("IN_PROGRESS".to_string(), String::new())
            } else {
                s.statuses.pop_front().unwrap_or_else(|| ("FINISHED".into(), "Finished: Media has been uploaded".into()))
            };
            let mut ans = json!({"id": id, "status_code": code, "status": status});
            if fields.contains("video_status") {
                ans["video_status"] = json!({"uploading_phase": {"status": if complete { "complete" } else { "in_progress" }, "bytes_transferred": received}});
            }
            respond(stream, 200, &[], &ans.to_string())
        }
        ("GET", media) if s.published.iter().any(|m| m == media) => {
            respond(stream, 200, &[], &json!({"id": media, "permalink": format!("https://www.instagram.com/reel/C{media}/"), "shortcode": "Cx"}).to_string())
        }
        _ => respond(stream, 400, &[], &graph_error(100, 33, "Unsupported request")),
    }
}
