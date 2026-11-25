//! A small HTTP/1.1 client for servers on this computer only (a local ComfyUI): plain HTTP to a
//! loopback address, one request a connection (`Connection: close`), every wait timed out and
//! every answer capped. It refuses any other host by construction: the address it connects to
//! is a loopback IP it parsed itself, never a name looked up elsewhere.
//!
//! Std only, no dependency (the engine builds for the browser too; this module is native only).

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, TcpStream};
use std::path::Path;
use std::time::Duration;

/// The most a status line and headers may weigh.
const MAX_HEAD: usize = 64 * 1024;
const MAX_HEADERS: usize = 100;
/// The most chunks a chunked answer may have.
const MAX_CHUNKS: usize = 1 << 20;

/// A server on this computer: its address and the `Host` it is asked as.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Endpoint {
    addr: SocketAddr,
    host: String,
}

impl Endpoint {
    /// `http://127.0.0.1:<port>`, `http://localhost:<port>` or `http://[::1]:<port>` (a trailing
    /// `/` allowed); anything else (another host, https, a path, a user name) is refused.
    pub fn parse(url: &str) -> Result<Endpoint, String> {
        let shown: String = url.chars().take(120).collect();
        let refuse = |why: &str| Err(format!("`{shown}` {why}"));
        let Some(rest) = url.strip_prefix("http://") else {
            return refuse("is not an http:// address on this computer (a local server speaks plain HTTP)");
        };
        let rest = rest.strip_suffix('/').unwrap_or(rest);
        if rest.contains(['/', '?', '#', '@', ' ', '\0']) {
            return refuse("must be the server alone: http://127.0.0.1:<port>");
        }
        let (host, port) = match rest.rsplit_once(':') {
            Some((h, p)) if !h.is_empty() => (h, p),
            _ => return refuse("needs a port: http://127.0.0.1:8188"),
        };
        let port: u16 = match port.parse() {
            Ok(p) if p > 0 => p,
            _ => return refuse("has no valid port (1 to 65535)"),
        };
        let ip = match host {
            "127.0.0.1" | "localhost" => IpAddr::V4(Ipv4Addr::LOCALHOST),
            "[::1]" => IpAddr::V6(Ipv6Addr::LOCALHOST),
            _ => return refuse("is not on this computer: only 127.0.0.1, localhost and [::1] are allowed"),
        };
        Ok(Endpoint { addr: SocketAddr::new(ip, port), host: format!("{host}:{port}") })
    }

    /// The server as an address (`http://127.0.0.1:8188`).
    pub fn url(&self) -> String {
        format!("http://{}", self.host)
    }
}

/// How long to wait to connect and for each read or write, and the most an answer may weigh.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    pub connect: Duration,
    pub io: Duration,
    pub max_body: u64,
}

/// What goes in a request.
pub enum Body<'a> {
    Empty,
    /// Bytes with their content type.
    Bytes(&'a [u8], &'a str),
    /// A `multipart/form-data` body: text fields, then one file read from disk as it is sent.
    Form {
        boundary: &'a str,
        fields: &'a [(&'a str, &'a str)],
        file_field: &'a str,
        file_name: &'a str,
        file: &'a Path,
    },
}

/// An answer: its status and body.
#[derive(Debug)]
pub struct Reply {
    pub status: u16,
    pub body: Vec<u8>,
}

/// Why a request failed.
#[derive(Debug, PartialEq, Eq)]
pub enum Fail {
    /// Nothing answered (no server, or it is starting).
    Unreachable(String),
    /// It answered and then stopped mid-way (closed, reset, timed out).
    Dropped(String),
    /// The answer was larger than allowed.
    TooLarge,
    /// The answer was not HTTP that can be read.
    Bad(String),
}

impl std::fmt::Display for Fail {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Fail::Unreachable(e) => write!(f, "nothing answers ({e})"),
            Fail::Dropped(e) => write!(f, "the connection broke off ({e})"),
            Fail::TooLarge => write!(f, "the answer is larger than allowed"),
            Fail::Bad(e) => write!(f, "the answer cannot be read ({e})"),
        }
    }
}

fn dropped(e: std::io::Error) -> Fail {
    Fail::Dropped(e.to_string())
}

/// `%`-encodes a query value (everything but letters, digits and `-._~`).
pub fn encode(v: &str) -> String {
    let mut out = String::with_capacity(v.len());
    for b in v.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_' | b'~') {
            out.push(char::from(b));
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn connect(ep: &Endpoint, limits: &Limits) -> Result<TcpStream, Fail> {
    let s = TcpStream::connect_timeout(&ep.addr, limits.connect).map_err(|e| Fail::Unreachable(e.to_string()))?;
    let io = Some(limits.io.max(Duration::from_millis(10)));
    s.set_read_timeout(io).map_err(dropped)?;
    s.set_write_timeout(io).map_err(dropped)?;
    let _ = s.set_nodelay(true);
    Ok(s)
}

/// The request's head and body sent.
fn send(s: &mut TcpStream, ep: &Endpoint, method: &str, path: &str, body: &Body) -> Result<(), Fail> {
    if !path.starts_with('/') || path.contains(['\r', '\n', ' ', '\0']) {
        return Err(Fail::Bad("a request path must start with / and hold no spaces or line breaks".into()));
    }
    let (kind, length, form) = match body {
        Body::Empty => (None, 0u64, None),
        Body::Bytes(b, t) => (Some(t.to_string()), b.len() as u64, None),
        Body::Form { boundary, fields, file_field, file_name, file } => {
            if boundary.is_empty() || boundary.len() > 70 || !boundary.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-') {
                return Err(Fail::Bad("a form boundary is 1 to 70 letters, digits or -".into()));
            }
            let safe = |t: &str| !t.contains(['"', '\r', '\n', '\0']);
            if !safe(file_field) || !safe(file_name) || fields.iter().any(|(k, v)| !safe(k) || v.contains(['\r', '\n', '\0'])) {
                return Err(Fail::Bad("a form field holds a quote or a line break".into()));
            }
            let mut head = Vec::new();
            for (k, v) in fields.iter() {
                head.extend_from_slice(format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n").as_bytes());
            }
            head.extend_from_slice(
                format!("--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; filename=\"{file_name}\"\r\nContent-Type: application/octet-stream\r\n\r\n")
                    .as_bytes(),
            );
            let tail = format!("\r\n--{boundary}--\r\n").into_bytes();
            let size = std::fs::metadata(file).map_err(|e| Fail::Bad(format!("{}: {e}", file.display())))?.len();
            let length = (head.len() as u64).saturating_add(size).saturating_add(tail.len() as u64);
            (Some(format!("multipart/form-data; boundary={boundary}")), length, Some((head, size, tail)))
        }
    };
    let mut head = format!("{method} {path} HTTP/1.1\r\nHost: {}\r\nUser-Agent: AutoCut\r\nAccept: */*\r\nConnection: close\r\n", ep.host);
    if let Some(t) = &kind {
        head.push_str(&format!("Content-Type: {t}\r\n"));
    }
    if !matches!(body, Body::Empty) || method == "POST" {
        head.push_str(&format!("Content-Length: {length}\r\n"));
    }
    head.push_str("\r\n");
    s.write_all(head.as_bytes()).map_err(dropped)?;
    match (body, form) {
        (Body::Bytes(b, _), _) => s.write_all(b).map_err(dropped)?,
        (Body::Form { file, .. }, Some((head, size, tail))) => {
            s.write_all(&head).map_err(dropped)?;
            let f = std::fs::File::open(file).map_err(|e| Fail::Bad(format!("{}: {e}", file.display())))?;
            // exactly as many bytes as were announced, even if the file grew meanwhile
            let sent = std::io::copy(&mut f.take(size), s).map_err(dropped)?;
            if sent != size {
                return Err(Fail::Bad(format!("{} changed while it was being sent", file.display())));
            }
            s.write_all(&tail).map_err(dropped)?;
        }
        _ => {}
    }
    s.flush().map_err(dropped)
}

/// One line of the head (without its line break), counted against `left`.
fn head_line(r: &mut impl BufRead, left: &mut usize) -> Result<String, Fail> {
    let mut line = Vec::new();
    let n = r.by_ref().take((*left as u64).saturating_add(1)).read_until(b'\n', &mut line).map_err(dropped)?;
    if n == 0 {
        return Err(Fail::Dropped("the server closed the connection".into()));
    }
    if n > *left {
        return Err(Fail::Bad("the answer's head is too long".into()));
    }
    *left -= n;
    while matches!(line.last(), Some(b'\n' | b'\r')) {
        line.pop();
    }
    String::from_utf8(line).map_err(|_| Fail::Bad("the answer's head is not text".into()))
}

/// The status, and how the body comes: its length, or chunked, or until the connection closes.
fn read_head(r: &mut impl BufRead) -> Result<(u16, Option<u64>, bool), Fail> {
    let mut left = MAX_HEAD;
    let status_line = head_line(r, &mut left)?;
    let mut parts = status_line.split(' ');
    let (Some(version), Some(code)) = (parts.next(), parts.next()) else { return Err(Fail::Bad("no status line".into())) };
    if !version.starts_with("HTTP/1.") {
        return Err(Fail::Bad("not an HTTP/1 answer".into()));
    }
    let status: u16 = code.parse().map_err(|_| Fail::Bad("no status code".into()))?;
    let (mut length, mut chunked) = (None, false);
    for _ in 0..=MAX_HEADERS {
        let line = head_line(r, &mut left)?;
        if line.is_empty() {
            return Ok((status, length, chunked));
        }
        let Some((name, value)) = line.split_once(':') else { continue };
        let (name, value) = (name.trim().to_ascii_lowercase(), value.trim());
        if name == "content-length" {
            length = Some(value.parse::<u64>().map_err(|_| Fail::Bad("a bad Content-Length".into()))?);
        } else if name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked") {
            chunked = true;
        }
    }
    Err(Fail::Bad("too many headers".into()))
}

/// The body into `sink`, never more than `max` bytes.
fn read_body(r: &mut impl BufRead, length: Option<u64>, chunked: bool, max: u64, sink: &mut dyn Write) -> Result<u64, Fail> {
    let mut copy = |r: &mut dyn Read, n: u64, total: &mut u64| -> Result<(), Fail> {
        let next = total.checked_add(n).ok_or(Fail::TooLarge)?;
        if next > max {
            return Err(Fail::TooLarge);
        }
        let got = std::io::copy(&mut r.take(n), sink).map_err(dropped)?;
        if got != n {
            return Err(Fail::Dropped("the answer ended early".into()));
        }
        *total = next;
        Ok(())
    };
    let mut total = 0u64;
    if chunked {
        for _ in 0..MAX_CHUNKS {
            let mut left = 1024;
            let line = head_line(r, &mut left)?;
            let size_text = line.split(';').next().unwrap_or("").trim();
            let size = u64::from_str_radix(size_text, 16).map_err(|_| Fail::Bad("a bad chunk size".into()))?;
            if size == 0 {
                // trailers, up to the empty line
                let mut left = MAX_HEAD;
                for _ in 0..=MAX_HEADERS {
                    if head_line(r, &mut left)?.is_empty() {
                        return Ok(total);
                    }
                }
                return Err(Fail::Bad("too many trailers".into()));
            }
            copy(r, size, &mut total)?;
            let mut left = 2;
            if !head_line(r, &mut left)?.is_empty() {
                return Err(Fail::Bad("a chunk does not end where it said".into()));
            }
        }
        return Err(Fail::Bad("too many chunks".into()));
    }
    match length {
        Some(n) => copy(r, n, &mut total)?,
        None => {
            // until the server closes the connection
            let got = std::io::copy(&mut r.take(max.saturating_add(1)), sink).map_err(dropped)?;
            if got > max {
                return Err(Fail::TooLarge);
            }
            total = got;
        }
    }
    Ok(total)
}

/// One request; the answer's body is kept in memory (at most `limits.max_body`).
pub fn call(ep: &Endpoint, method: &str, path: &str, body: Body, limits: Limits) -> Result<Reply, Fail> {
    let mut s = connect(ep, &limits)?;
    send(&mut s, ep, method, path, &body)?;
    let mut r = BufReader::new(s);
    let (status, length, chunked) = read_head(&mut r)?;
    if length.is_some_and(|n| n > limits.max_body) {
        return Err(Fail::TooLarge);
    }
    let mut out = Vec::new();
    read_body(&mut r, length, chunked, limits.max_body, &mut out)?;
    Ok(Reply { status, body: out })
}

/// A `GET` whose answer is written to `file` (at most `limits.max_body` bytes); the status and
/// the bytes written. A status other than 200 writes nothing.
pub fn fetch_to(ep: &Endpoint, path: &str, file: &Path, limits: Limits) -> Result<(u16, u64), Fail> {
    let mut s = connect(ep, &limits)?;
    send(&mut s, ep, "GET", path, &Body::Empty)?;
    let mut r = BufReader::new(s);
    let (status, length, chunked) = read_head(&mut r)?;
    if status != 200 {
        return Ok((status, 0));
    }
    if length.is_some_and(|n| n > limits.max_body) {
        return Err(Fail::TooLarge);
    }
    let mut f = std::fs::File::create(file).map_err(|e| Fail::Bad(format!("{}: {e}", file.display())))?;
    let n = read_body(&mut r, length, chunked, limits.max_body, &mut f);
    if n.is_err() {
        drop(f);
        let _ = std::fs::remove_file(file);
    }
    Ok((status, n?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn limits() -> Limits {
        Limits { connect: Duration::from_secs(2), io: Duration::from_secs(2), max_body: 1 << 20 }
    }

    /// A server that answers one connection with `answer` (after reading the request's head).
    fn once(answer: Vec<u8>) -> (Endpoint, std::thread::JoinHandle<Vec<u8>>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = l.local_addr().unwrap().port();
        let h = std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            c.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
            let mut got = Vec::new();
            let mut buf = [0u8; 4096];
            while !got.windows(4).any(|w| w == b"\r\n\r\n") {
                let n = c.read(&mut buf).unwrap_or(0);
                if n == 0 {
                    break;
                }
                got.extend_from_slice(&buf[..n]);
            }
            let _ = c.write_all(&answer);
            got
        });
        (Endpoint::parse(&format!("http://127.0.0.1:{port}")).unwrap(), h)
    }

    #[test]
    fn only_servers_on_this_computer_are_reached() {
        for ok in ["http://127.0.0.1:8188", "http://localhost:8188/", "http://[::1]:9000"] {
            assert!(Endpoint::parse(ok).is_ok(), "{ok}");
        }
        assert_eq!(Endpoint::parse("http://localhost:8188").unwrap().url(), "http://localhost:8188");
        for bad in [
            "https://127.0.0.1:8188",
            "http://192.168.1.4:8188",
            "http://0.0.0.0:8188",
            "http://example.com:8188",
            "http://127.0.0.1.example.com:8188",
            "http://user@127.0.0.1:8188",
            "http://127.0.0.1:8188/api",
            "http://127.0.0.1",
            "http://127.0.0.1:0",
            "http://127.0.0.1:99999",
            "ftp://127.0.0.1:21",
            "",
        ] {
            assert!(Endpoint::parse(bad).is_err(), "{bad}");
        }
        assert_eq!(encode("a b/c?.png"), "a%20b%2Fc%3F.png");
    }

    #[test]
    fn answers_by_length_chunks_or_close_are_read_and_capped() {
        let (ep, h) = once(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello".to_vec());
        let r = call(&ep, "POST", "/prompt", Body::Bytes(b"{}", "application/json"), limits()).unwrap();
        assert_eq!((r.status, r.body.as_slice()), (200, &b"hello"[..]));
        let req = String::from_utf8(h.join().unwrap()).unwrap();
        assert!(req.starts_with("POST /prompt HTTP/1.1\r\n") && req.contains("Content-Length: 2") && req.contains("Host: 127.0.0.1:"), "{req}");
        let (ep, _) = once(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2;x=y\r\nde\r\n0\r\n\r\n".to_vec());
        assert_eq!(call(&ep, "GET", "/x", Body::Empty, limits()).unwrap().body, b"abcde");
        let (ep, _) = once(b"HTTP/1.0 404 Not Found\r\n\r\ngone".to_vec());
        let r = call(&ep, "GET", "/x", Body::Empty, limits()).unwrap();
        assert_eq!((r.status, r.body.as_slice()), (404, &b"gone"[..]));
        // hostile answers: an error each, never a panic or an unbounded read
        for (answer, want) in [
            (b"HTTP/1.1 200 OK\r\nContent-Length: 999999999999\r\n\r\n".to_vec(), "larger"),
            (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nffffffffffffffff\r\n".to_vec(), "larger"),
            (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\n".to_vec(), "cannot be read"),
            (b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nshort".to_vec(), "broke off"),
            (b"SSH-2.0-OpenSSH\r\n\r\n".to_vec(), "cannot be read"),
            (b"HTTP/1.1 200 OK\r\nContent-Length: x\r\n\r\n".to_vec(), "cannot be read"),
            (b"".to_vec(), "broke off"),
            ([b"HTTP/1.1 200 OK\r\nX: ".as_slice(), &vec![b'a'; MAX_HEAD + 10]].concat(), "cannot be read"),
            ([b"HTTP/1.1 200 OK\r\n\r\n".as_slice(), &vec![b'a'; (1 << 20) + 10]].concat(), "larger"),
        ] {
            let (ep, _) = once(answer);
            let e = call(&ep, "GET", "/x", Body::Empty, limits()).unwrap_err().to_string();
            assert!(e.contains(want), "{want}: {e}");
        }
        // nobody there
        let free = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
        let ep = Endpoint::parse(&format!("http://127.0.0.1:{free}")).unwrap();
        assert!(matches!(call(&ep, "GET", "/", Body::Empty, limits()), Err(Fail::Unreachable(_))));
        assert!(matches!(call(&ep, "GET", "bad path", Body::Empty, limits()), Err(Fail::Unreachable(_) | Fail::Bad(_))));
    }

    #[test]
    fn a_file_is_sent_as_a_form_and_an_answer_written_to_a_file() {
        let dir = std::env::temp_dir().join(format!("fc-loopback-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let file = dir.join("in.png");
        std::fs::write(&file, b"PNGDATA").unwrap();
        let (ep, h) = once(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}".to_vec());
        let form = Body::Form { boundary: "b0und", fields: &[("type", "input")], file_field: "image", file_name: "a.png", file: &file };
        assert_eq!(call(&ep, "POST", "/upload/image", form, limits()).unwrap().status, 200);
        let req = String::from_utf8_lossy(&h.join().unwrap()).into_owned();
        assert!(req.contains("multipart/form-data; boundary=b0und"), "{req}");
        let bad = Body::Form { boundary: "b0und", fields: &[], file_field: "image", file_name: "a\".png", file: &file };
        let (ep, _) = once(Vec::new());
        assert!(matches!(call(&ep, "POST", "/upload/image", bad, limits()), Err(Fail::Bad(_))));
        let out = dir.join("out.bin");
        let (ep, _) = once(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nDATA".to_vec());
        assert_eq!(fetch_to(&ep, "/view?filename=a.png", &out, limits()).unwrap(), (200, 4));
        assert_eq!(std::fs::read(&out).unwrap(), b"DATA");
        let (ep, _) = once(b"HTTP/1.1 200 OK\r\nContent-Length: 40\r\n\r\nDATA".to_vec());
        assert!(fetch_to(&ep, "/view", &out, limits()).is_err());
        assert!(!out.exists(), "a broken download leaves no file");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
