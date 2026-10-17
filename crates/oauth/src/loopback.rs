//! The loopback redirect of RFC 8252 §7.3: a listener on `127.0.0.1` with a port the OS picks,
//! which the provider's consent page sends the browser back to with `code` and `state`.
//!
//! The listener answers one redirect: a request whose `state` is not the one this sign-in sent
//! ends the sign-in (its code is never exchanged); requests that are not a redirect at all (a
//! favicon fetch, a speculative preconnect that never speaks) are ignored, at most
//! [`MAX_STRAY_CONNECTIONS`] of them. The browser gets a plain page saying the user can return to
//! the app.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::time::{Duration, Instant};

use crate::{OAuthError, urldecode};

/// Connections that are not the redirect before giving up.
pub const MAX_STRAY_CONNECTIONS: usize = 64;
/// A request line plus headers; a browser never sends more to a redirect.
const MAX_REQUEST: u64 = 32 * 1024;
/// How long one connection may take to send its request.
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// What the browser tab says once the redirect has arrived.
#[derive(Debug, Clone)]
pub struct DonePage {
    /// The app to return to ("FilmCraft").
    pub app: String,
    /// What was signed in to ("Google Photos").
    pub service: String,
}

impl Default for DonePage {
    fn default() -> Self {
        Self { app: "FilmCraft".into(), service: "your account".into() }
    }
}

/// A bound loopback listener.
pub struct Loopback {
    listener: TcpListener,
    port: u16,
}

impl Loopback {
    /// Listen on `127.0.0.1` on a port the OS chooses (never a fixed one, so two sign-ins, or
    /// another program, cannot collide with it).
    pub fn bind() -> Result<Self, OAuthError> {
        let listener = TcpListener::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))
            .map_err(|e| OAuthError::SignIn(format!("cannot listen on 127.0.0.1 for the sign-in redirect: {e}")))?;
        let port = listener.local_addr().map_err(|e| OAuthError::SignIn(format!("the sign-in listener has no address: {e}")))?.port();
        listener.set_nonblocking(true).map_err(|e| OAuthError::SignIn(format!("cannot set up the sign-in listener: {e}")))?;
        Ok(Self { listener, port })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// `http://127.0.0.1:<port>`, the `redirect_uri` of this sign-in. (Google's Desktop clients
    /// accept a loopback redirect on any port without registering it.)
    pub fn redirect_uri(&self) -> String {
        format!("http://127.0.0.1:{}", self.port)
    }

    /// Wait at most `wait` for the redirect carrying `state` and return its authorization code.
    pub fn wait_for_code(&self, state: &str, wait: Duration, page: &DonePage) -> Result<String, OAuthError> {
        let deadline = Instant::now().checked_add(wait).unwrap_or_else(Instant::now);
        let mut strays = 0usize;
        while Instant::now() < deadline && strays < MAX_STRAY_CONNECTIONS {
            match self.listener.accept() {
                Ok((stream, _)) => match read_redirect(stream, state, page)? {
                    Some(code) => return Ok(code),
                    None => strays += 1,
                },
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => std::thread::sleep(Duration::from_millis(50)),
                Err(_) => strays += 1,
            }
        }
        if strays >= MAX_STRAY_CONNECTIONS {
            return Err(OAuthError::SignIn(format!("{strays} connections to the sign-in listener were not the sign-in")));
        }
        let mins = wait.as_secs().div_ceil(60).max(1);
        Err(OAuthError::SignIn(format!("no sign-in came back within {mins} minute{}", if mins == 1 { "" } else { "s" })))
    }
}

/// One request to the listener. `Some(code)` for the redirect, `None` for anything else; an
/// error for a declined consent, a mismatched `state` or a redirect without a code.
pub fn read_redirect(stream: TcpStream, want_state: &str, page: &DonePage) -> Result<Option<String>, OAuthError> {
    stream.set_nonblocking(false).ok();
    stream.set_read_timeout(Some(READ_TIMEOUT)).ok();
    stream.set_write_timeout(Some(READ_TIMEOUT)).ok();
    let Ok(clone) = stream.try_clone() else { return Ok(None) };
    let mut reader = BufReader::new(clone.take(MAX_REQUEST));
    let mut request_line = String::new();
    if reader.read_line(&mut request_line).is_err() {
        return Ok(None);
    }
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default().to_string();
    let target = parts.next().unwrap_or_default().to_string();
    // drain the headers so the browser sees the response
    let mut line = String::new();
    while reader.read_line(&mut line).is_ok_and(|n| n > 0) && line != "\r\n" && line != "\n" {
        line.clear();
    }
    let query = target.split_once('?').map(|(_, q)| q).unwrap_or_default();
    let query = query.split_once('#').map(|(q, _)| q).unwrap_or(query);
    let params: Vec<(String, String)> = query
        .split('&')
        .filter(|p| !p.is_empty())
        .take(64)
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (urldecode(k), urldecode(v))
        })
        .collect();
    let get = |k: &str| params.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str());
    let respond = |status: &str, title: &str, text: &str| {
        let body = format!(
            "<!doctype html><html><head><meta charset=\"utf-8\"><title>{t}</title></head><body style=\"font-family:system-ui,sans-serif;padding:3em;max-width:40em\"><h2>{t}</h2><p>{x}</p></body></html>",
            t = html_escape(title),
            x = html_escape(text)
        );
        let mut w = &stream;
        // the browser may already have gone; the sign-in's outcome does not depend on it
        let _ = write!(
            w,
            "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let _ = w.flush();
    };
    if method != "GET" || (get("state").is_none() && get("code").is_none() && get("error").is_none()) {
        respond("404 Not Found", "Not the sign-in", "This address only takes the sign-in redirect.");
        return Ok(None);
    }
    if get("state") != Some(want_state) {
        respond(
            "400 Bad Request",
            "Sign-in not accepted",
            &format!("This sign-in did not come from the request {} made. Return to {} and sign in again.", page.app, page.app),
        );
        return Err(OAuthError::SignIn("the sign-in redirect's state did not match this request (a stale or forged redirect); nothing was signed in".into()));
    }
    if let Some(err) = get("error") {
        let err = crate::truncate(err, 80).to_string();
        respond("200 OK", "Sign-in cancelled", &format!("Nothing was signed in. You can close this tab and return to {}.", page.app));
        return Err(OAuthError::SignIn(match err.as_str() {
            "access_denied" => "the consent was declined (access_denied); nothing was signed in".to_string(),
            other => format!("the provider returned `{other}`; nothing was signed in"),
        }));
    }
    let Some(code) = get("code").filter(|c| !c.is_empty() && c.len() <= 4096) else {
        respond("400 Bad Request", "Sign-in incomplete", &format!("The answer carried no sign-in code. Return to {} and try again.", page.app));
        return Err(OAuthError::SignIn("the sign-in redirect carried no authorization code".into()));
    };
    respond("200 OK", &format!("Signed in to {}", page.service), &format!("You can close this tab and return to {}.", page.app));
    Ok(Some(code.to_string()))
}

fn html_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#39;"),
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn send(addr: SocketAddr, req: &'static str) -> std::thread::JoinHandle<String> {
        std::thread::spawn(move || {
            let mut s = TcpStream::connect(addr).unwrap();
            s.write_all(req.as_bytes()).unwrap();
            let mut out = String::new();
            let _ = s.read_to_string(&mut out);
            out
        })
    }

    #[test]
    fn strays_are_ignored_and_state_is_checked() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let page = DonePage { app: "FilmCraft".into(), service: "Google Photos".into() };
        let favicon = send(addr, "GET /favicon.ico HTTP/1.1\r\nHost: x\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert_eq!(read_redirect(stream, "s1", &page).unwrap(), None);
        assert!(favicon.join().unwrap().starts_with("HTTP/1.1 404"));

        let post = send(addr, "POST /?state=s1&code=c HTTP/1.1\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert_eq!(read_redirect(stream, "s1", &page).unwrap(), None, "only GET");
        post.join().unwrap();

        let good = send(addr, "GET /?state=s1&code=4%2F0Ax&scope=x HTTP/1.1\r\nHost: x\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert_eq!(read_redirect(stream, "s1", &page).unwrap().as_deref(), Some("4/0Ax"));
        let answer = good.join().unwrap();
        assert!(answer.starts_with("HTTP/1.1 200") && answer.contains("Signed in to Google Photos") && answer.contains("return to FilmCraft"), "{answer}");

        let forged = send(addr, "GET /?state=other&code=c HTTP/1.1\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert!(read_redirect(stream, "s1", &page).unwrap_err().to_string().contains("state did not match"));
        assert!(forged.join().unwrap().starts_with("HTTP/1.1 400"));

        let no_state = send(addr, "GET /?code=c HTTP/1.1\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert!(read_redirect(stream, "s1", &page).is_err(), "a code without state is refused");
        no_state.join().unwrap();

        let declined = send(addr, "GET /?state=s1&error=access_denied HTTP/1.1\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert!(read_redirect(stream, "s1", &page).unwrap_err().to_string().contains("declined"));
        declined.join().unwrap();

        let empty = send(addr, "GET /?state=s1&code= HTTP/1.1\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert!(read_redirect(stream, "s1", &page).unwrap_err().to_string().contains("no authorization code"));
        empty.join().unwrap();

        // a connection that never speaks, and garbage
        let silent = std::thread::spawn(move || {
            let s = TcpStream::connect(addr).unwrap();
            std::thread::sleep(Duration::from_millis(50));
            drop(s);
        });
        let (stream, _) = listener.accept().unwrap();
        assert_eq!(read_redirect(stream, "s1", &page).unwrap(), None);
        silent.join().unwrap();
        let junk = send(addr, "\u{1}\u{2}garbage\r\n\r\n");
        let (stream, _) = listener.accept().unwrap();
        assert_eq!(read_redirect(stream, "s1", &page).unwrap(), None);
        junk.join().unwrap();
    }

    #[test]
    fn the_page_is_escaped() {
        assert_eq!(html_escape("<a href=\"x\">&'"), "&lt;a href=&quot;x&quot;&gt;&amp;&#39;");
    }

    #[test]
    fn listens_on_a_random_loopback_port() {
        let a = Loopback::bind().unwrap();
        let b = Loopback::bind().unwrap();
        assert_ne!(a.port(), b.port());
        assert_ne!(a.port(), 0);
        assert_eq!(a.redirect_uri(), format!("http://127.0.0.1:{}", a.port()));
        // nothing comes: a bounded wait, then an error
        let e = a.wait_for_code("s", Duration::from_millis(120), &DonePage::default()).unwrap_err();
        assert!(e.to_string().contains("no sign-in came back"), "{e}");
    }

    #[test]
    fn waits_through_strays_for_the_redirect() {
        let lb = Loopback::bind().unwrap();
        let addr: SocketAddr = format!("127.0.0.1:{}", lb.port()).parse().unwrap();
        let browser = std::thread::spawn(move || {
            for req in ["GET /favicon.ico HTTP/1.1\r\n\r\n", "GET /?state=st&code=abc HTTP/1.1\r\n\r\n"] {
                let mut s = TcpStream::connect(addr).unwrap();
                s.write_all(req.as_bytes()).unwrap();
                let mut out = String::new();
                let _ = s.read_to_string(&mut out);
            }
        });
        assert_eq!(lb.wait_for_code("st", Duration::from_secs(20), &DonePage::default()).unwrap(), "abc");
        browser.join().unwrap();
    }
}
