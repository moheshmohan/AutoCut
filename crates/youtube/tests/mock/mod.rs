//! A stand-in for Google on 127.0.0.1: the token and revocation endpoints, `channels.list`,
//! `playlists.list`, `playlistItems.insert`, `thumbnails.set` and the resumable `videos.insert`
//! protocol, with scripted faults (dropped connections, 5xx, expired sessions, error answers,
//! hostile headers). One request per connection (`Connection: close`).

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};

/// What happens to one chunk `PUT` (by its index among chunk PUTs).
#[derive(Debug, Clone)]
pub enum Fault {
    /// The connection breaks after Google kept this many bytes of the chunk; no answer.
    Drop(usize),
    /// Google answers this; nothing is kept.
    Status(u16, String),
    /// The session is gone (404 from now on).
    Expire,
}

/// One request as the mock saw it.
#[derive(Debug, Clone)]
pub struct Seen {
    pub method: String,
    pub path: String,
    pub auth: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
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
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap_or(Value::Null)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Session {
    pub total: u64,
    pub received: Vec<u8>,
    pub channel: String,
    pub meta: Value,
    pub expired: bool,
    pub done: bool,
}

/// A channel the mock knows: its key (what the "user" picks), id, title, handle.
#[derive(Debug, Clone)]
pub struct Chan {
    pub key: String,
    pub id: String,
    pub title: String,
    pub handle: String,
}

#[derive(Default)]
pub struct State {
    pub channels: Vec<Chan>,
    /// The channel the stand-in browser picks in the account chooser.
    pub picked: String,
    /// The scope the token endpoint grants (default: what was asked).
    pub grant: Option<String>,
    /// The refresh token is refused (`invalid_grant`).
    pub refresh_refused: bool,
    pub issued: u32,
    pub sessions: HashMap<u32, Session>,
    pub next_session: u32,
    /// Answers to `videos.insert` before it opens a session.
    pub insert_answers: VecDeque<(u16, String)>,
    /// A `Location` other than the mock's own.
    pub location: Option<String>,
    pub chunk_faults: HashMap<usize, Fault>,
    pub puts: usize,
    /// Answers to status checks before the real one.
    pub status_answers: VecDeque<(u16, Vec<(String, String)>, String)>,
    /// A `Range` header replacing the real one on chunk answers.
    pub range: Option<String>,
    /// The finished upload's answer instead of the video.
    pub video_answer: Option<(u16, String)>,
    pub thumbnail_answer: Option<(u16, String)>,
    pub thumbnails: Vec<(String, usize, String)>,
    pub playlists: Vec<(String, String)>,
    pub playlist_items: Vec<(String, String)>,
    /// Answers to `channels.list` instead of the channel.
    pub channels_answer: Option<(u16, String)>,
    pub revoked: Vec<String>,
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
        let state = Arc::new(Mutex::new(State {
            channels: vec![
                Chan { key: "h4hacks".into(), id: "UCh4hacks000000000000001".into(), title: "H4hacks".into(), handle: "@h4hacks".into() },
                Chan { key: "mohesh".into(), id: "UCmohesh0000000000000002".into(), title: "Mohesh Mohan".into(), handle: "@moheshmohan".into() },
                Chan { key: "nochannel".into(), id: String::new(), title: String::new(), handle: String::new() },
            ],
            picked: "h4hacks".into(),
            next_session: 1,
            ..Default::default()
        }));
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

    /// The stand-in browser: follows the consent URL's redirect with a code for the channel
    /// `picked` (as if the user picked it in the account chooser), from another thread.
    pub fn browser(&self) -> impl Fn(&str) -> bool + Send + Sync + 'static {
        let state = self.state.clone();
        move |url: &str| {
            let q = url.split_once('?').map(|(_, q)| q).unwrap_or_default();
            let get = |k: &str| q.split('&').find_map(|p| p.split_once('=').filter(|(a, _)| *a == k).map(|(_, v)| filmcraft_oauth::urldecode(v)));
            let (Some(redirect), Some(st)) = (get("redirect_uri"), get("state")) else { return false };
            let scope = get("scope").unwrap_or_default();
            let picked = state.lock().unwrap().picked.clone();
            let code = format!("code-{picked}-{}", filmcraft_oauth::urlencode(&scope));
            let target = format!("{redirect}/?code={}&state={}", filmcraft_oauth::urlencode(&code), filmcraft_oauth::urlencode(&st));
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(30));
                let agent: ureq::Agent = ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(Duration::from_secs(10))).build().into();
                let _ = agent.get(&target).call();
            });
            true
        }
    }
}

fn read_request(stream: &TcpStream) -> Option<(String, String, Vec<(String, String)>, Vec<u8>)> {
    let mut reader = BufReader::new(stream.try_clone().ok()?);
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
    let mut body = vec![0; length];
    reader.read_exact(&mut body).ok()?;
    Some((method, path, headers, body))
}

fn respond(mut stream: &TcpStream, status: u16, headers: &[(String, String)], body: &str) {
    let reason = match status {
        200 => "OK",
        201 => "Created",
        308 => "Resume Incomplete",
        _ => "Error",
    };
    let mut head = format!("HTTP/1.1 {status} {reason}\r\nContent-Length: {}\r\nConnection: close\r\n", body.len());
    for (k, v) in headers {
        head.push_str(&format!("{k}: {v}\r\n"));
    }
    head.push_str("\r\n");
    let _ = stream.write_all(head.as_bytes());
    let _ = stream.write_all(body.as_bytes());
    let _ = stream.flush();
}

fn json_ok(stream: &TcpStream, v: Value) {
    respond(stream, 200, &[("Content-Type".into(), "application/json".into())], &v.to_string());
}

fn err(status: u16, reason: &str, message: &str) -> String {
    json!({"error": {"code": status, "message": message, "errors": [{"reason": reason, "message": message}]}}).to_string()
}

pub fn error_body(status: u16, reason: &str, message: &str) -> String {
    err(status, reason, message)
}

/// The channel a bearer token acts as (`at-<key>-<n>`).
fn channel_of(state: &State, auth: &str) -> Option<Chan> {
    let token = auth.strip_prefix("Bearer ")?;
    let key = token.strip_prefix("at-")?.rsplit_once('-')?.0;
    state.channels.iter().find(|c| c.key == key).cloned()
}

fn handle(stream: TcpStream, state: &Arc<Mutex<State>>, base: &str) {
    let Some((method, path, headers, body)) = read_request(&stream) else { return };
    let auth = headers.iter().find(|(k, _)| k.eq_ignore_ascii_case("authorization")).map(|(_, v)| v.clone()).unwrap_or_default();
    let seen = Seen { method: method.clone(), path: path.clone(), auth: auth.clone(), headers: headers.clone(), body: body.clone() };
    let mut s = state.lock().unwrap();
    s.seen.push(seen.clone());
    let route = path.split('?').next().unwrap_or_default().to_string();
    match (method.as_str(), route.as_str()) {
        ("POST", "/token") => {
            if seen.form("grant_type").as_deref() == Some("refresh_token") {
                if s.refresh_refused {
                    return respond(&stream, 400, &[], r#"{"error":"invalid_grant","error_description":"Token has been expired or revoked."}"#);
                }
                let rt = seen.form("refresh_token").unwrap_or_default();
                let key = rt.strip_prefix("rt-").unwrap_or_default().to_string();
                s.issued += 1;
                let n = s.issued;
                return json_ok(&stream, json!({"access_token": format!("at-{key}-{n}"), "expires_in": 3599, "token_type": "Bearer"}));
            }
            let code = seen.form("code").unwrap_or_default();
            let Some(rest) = code.strip_prefix("code-") else { return respond(&stream, 400, &[], r#"{"error":"invalid_grant"}"#) };
            let (key, scope) = rest.split_once('-').unwrap_or((rest, ""));
            let scope = s.grant.clone().unwrap_or_else(|| filmcraft_oauth::urldecode(scope));
            s.issued += 1;
            let n = s.issued;
            json_ok(
                &stream,
                json!({"access_token": format!("at-{key}-{n}"), "refresh_token": format!("rt-{key}"), "expires_in": 3599, "scope": scope, "token_type": "Bearer"}),
            )
        }
        ("POST", "/revoke") => {
            if let Some(t) = seen.form("token") {
                s.revoked.push(t);
            }
            respond(&stream, 200, &[], "")
        }
        ("GET", "/youtube/v3/channels") => {
            if let Some((st, b)) = s.channels_answer.clone() {
                return respond(&stream, st, &[], &b);
            }
            match channel_of(&s, &auth) {
                Some(c) if c.id.is_empty() => json_ok(&stream, json!({"kind": "youtube#channelListResponse", "items": []})),
                Some(c) => json_ok(
                    &stream,
                    json!({"items": [{"id": c.id, "snippet": {"title": c.title, "customUrl": c.handle}, "status": {"longUploadsStatus": "allowed"}}]}),
                ),
                None => respond(&stream, 401, &[], &err(401, "authError", "Invalid Credentials")),
            }
        }
        ("GET", "/youtube/v3/playlists") => {
            if channel_of(&s, &auth).is_none() {
                return respond(&stream, 401, &[], &err(401, "authError", "Invalid Credentials"));
            }
            // two pages: the first half, then the rest
            let half = s.playlists.len() / 2;
            let second = path.contains("pageToken=");
            let page: Vec<Value> = s
                .playlists
                .iter()
                .enumerate()
                .filter(|(i, _)| if second { *i >= half } else { *i < half })
                .map(|(_, (id, title))| json!({"id": id, "snippet": {"title": title}}))
                .collect();
            let mut v = json!({"items": page});
            if !second && half > 0 {
                v["nextPageToken"] = json!("PAGE2");
            }
            json_ok(&stream, v)
        }
        ("POST", "/youtube/v3/playlistItems") => {
            let v = seen.json();
            let pl = v["snippet"]["playlistId"].as_str().unwrap_or_default().to_string();
            let vid = v["snippet"]["resourceId"]["videoId"].as_str().unwrap_or_default().to_string();
            s.playlist_items.push((pl.clone(), vid));
            json_ok(&stream, json!({"id": "PLI1", "snippet": {"playlistId": pl}}))
        }
        ("POST", "/upload/youtube/v3/thumbnails/set") => {
            if let Some((st, b)) = s.thumbnail_answer.clone() {
                return respond(&stream, st, &[], &b);
            }
            let vid = path.split("videoId=").nth(1).and_then(|r| r.split('&').next()).unwrap_or_default().to_string();
            let ct = seen.header("content-type").unwrap_or_default().to_string();
            s.thumbnails.push((vid, body.len(), ct));
            json_ok(&stream, json!({"items": [{"default": {"url": "https://i.ytimg.com/vi/x/default.jpg"}}]}))
        }
        ("POST", "/upload/youtube/v3/videos") => {
            if let Some((st, b)) = s.insert_answers.pop_front() {
                return respond(&stream, st, &[], &b);
            }
            let Some(c) = channel_of(&s, &auth) else { return respond(&stream, 401, &[], &err(401, "authError", "Invalid Credentials")) };
            let total: u64 = seen.header("x-upload-content-length").and_then(|v| v.parse().ok()).unwrap_or(0);
            let id = s.next_session;
            s.next_session += 1;
            s.sessions.insert(id, Session { total, channel: c.id.clone(), meta: seen.json(), ..Default::default() });
            let loc = s.location.clone().unwrap_or_else(|| format!("{base}/upload/session/{id}"));
            respond(&stream, 200, &[("Location".into(), loc)], "")
        }
        ("PUT", r) if r.starts_with("/upload/session/") => {
            let id: u32 = r.trim_start_matches("/upload/session/").parse().unwrap_or(0);
            if channel_of(&s, &auth).is_none() {
                return respond(&stream, 401, &[], &err(401, "authError", "Invalid Credentials"));
            }
            let range = seen.header("content-range").unwrap_or_default().to_string();
            let Some(sess) = s.sessions.get(&id).cloned() else { return respond(&stream, 404, &[], "Not Found") };
            if sess.expired {
                return respond(&stream, 404, &[], "Not Found");
            }
            let received = sess.received.len() as u64;
            let range_header = |n: u64| -> Vec<(String, String)> { if n == 0 { vec![] } else { vec![("Range".into(), format!("bytes=0-{}", n - 1))] } };
            if range.starts_with("bytes */") {
                if let Some((st, h, b)) = s.status_answers.pop_front() {
                    return respond(&stream, st, &h, &b);
                }
                if sess.done {
                    return finish(&stream, &mut s, id);
                }
                return respond(&stream, 308, &range_header(received), "");
            }
            // a chunk: bytes a-b/total
            let k = s.puts;
            s.puts += 1;
            let spec = range.trim_start_matches("bytes ");
            let (span, _) = spec.split_once('/').unwrap_or((spec, ""));
            let (a, _) = span.split_once('-').unwrap_or(("0", "0"));
            let a: u64 = a.parse().unwrap_or(u64::MAX);
            if a != received {
                // overlapping or skipped bytes fail the upload, as YouTube's do
                return respond(&stream, 400, &[], &err(400, "badContent", &format!("chunk starts at {a}, {received} received")));
            }
            match s.chunk_faults.remove(&k) {
                Some(Fault::Drop(keep)) => {
                    let keep = keep.min(body.len());
                    s.sessions.get_mut(&id).unwrap().received.extend_from_slice(&body[..keep]);
                    // no answer: the connection just closes
                    drop(s);
                    let _ = stream.shutdown(std::net::Shutdown::Both);
                    return;
                }
                Some(Fault::Status(st, b)) => return respond(&stream, st, &[], &b),
                Some(Fault::Expire) => {
                    s.sessions.get_mut(&id).unwrap().expired = true;
                    return respond(&stream, 404, &[], "Not Found");
                }
                None => {}
            }
            let sess = s.sessions.get_mut(&id).unwrap();
            sess.received.extend_from_slice(&body);
            let now = sess.received.len() as u64;
            if now >= sess.total {
                sess.done = true;
                return finish(&stream, &mut s, id);
            }
            let h = s.range.clone().map(|r| vec![("Range".to_string(), r)]).unwrap_or_else(|| range_header(now));
            respond(&stream, 308, &h, "")
        }
        _ => respond(&stream, 404, &[], &format!("no route {method} {path}")),
    }
}

fn finish(stream: &TcpStream, s: &mut State, id: u32) {
    if let Some((st, b)) = s.video_answer.clone() {
        return respond(stream, st, &[], &b);
    }
    let sess = s.sessions.get(&id).cloned().unwrap_or_default();
    let v = json!({
        "kind": "youtube#video", "id": format!("vid{id:08}"),
        "snippet": {"title": sess.meta["snippet"]["title"], "channelId": sess.channel},
        "status": {"privacyStatus": "private", "uploadStatus": "uploaded"},
    });
    respond(stream, 201, &[("Content-Type".into(), "application/json".into())], &v.to_string());
}
