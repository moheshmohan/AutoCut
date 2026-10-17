//! The whole sign-in offline: a mock token endpoint on localhost, a stand-in browser that follows
//! the consent URL's redirect, and the refresh and revocation calls against the same mock.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use filmcraft_oauth::loopback::DonePage;
use filmcraft_oauth::pkce::challenge_s256;
use filmcraft_oauth::{OAuthClient, OAuthError, Provider, SignInRequest, TokenSet, urldecode};

/// One request the mock endpoint received.
#[derive(Debug, Clone)]
struct Seen {
    method: String,
    path: String,
    content_type: String,
    form: Vec<(String, String)>,
}

impl Seen {
    fn get(&self, k: &str) -> Option<&str> {
        self.form.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str())
    }
}

/// A token endpoint on 127.0.0.1 answering each request with the next scripted (status, body).
struct MockEndpoint {
    url: String,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl MockEndpoint {
    fn start(answers: Vec<(u16, String)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for (status, body) in answers {
                let Ok((stream, _)) = listener.accept() else { return };
                log.lock().unwrap().push(read_request(&stream));
                let reason = if status == 200 { "OK" } else { "Error" };
                let mut w = &stream;
                let _ = write!(
                    w,
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
            }
        });
        Self { url, seen }
    }

    fn seen(&self) -> Vec<Seen> {
        self.seen.lock().unwrap().clone()
    }
}

fn read_request(stream: &TcpStream) -> Seen {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let mut parts = line.split_whitespace();
    let (method, path) = (parts.next().unwrap_or_default().to_string(), parts.next().unwrap_or_default().to_string());
    let (mut length, mut content_type) = (0usize, String::new());
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).unwrap() == 0 || h == "\r\n" {
            break;
        }
        let (k, v) = h.split_once(':').unwrap_or((&h, ""));
        match k.trim().to_ascii_lowercase().as_str() {
            "content-length" => length = v.trim().parse().unwrap(),
            "content-type" => content_type = v.trim().to_string(),
            _ => {}
        }
    }
    let mut body = vec![0; length];
    reader.read_exact(&mut body).unwrap();
    let body = String::from_utf8(body).unwrap();
    let form = body
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (urldecode(k), urldecode(v))
        })
        .collect();
    Seen { method, path, content_type, form }
}

/// An agent that may talk plain HTTP to the mock (the product's is https-only).
fn agent() -> ureq::Agent {
    ureq::Agent::config_builder().http_status_as_error(false).timeout_global(Some(Duration::from_secs(20))).build().into()
}

fn client(endpoint: &MockEndpoint) -> OAuthClient {
    OAuthClient {
        provider: Provider {
            token_url: format!("{}/token", endpoint.url),
            revoke_url: Some(format!("{}/revoke", endpoint.url)),
            auth_url: "https://accounts.example.test/o/oauth2/v2/auth".into(),
            ..Provider::google()
        },
        client_id: "id.apps.googleusercontent.com".into(),
        client_secret: Some("desktop-secret".into()),
        sign_in_command: "filmcraft-cli photos auth".into(),
    }
}

fn request() -> SignInRequest {
    SignInRequest {
        scopes: vec!["https://www.googleapis.com/auth/photospicker.mediaitems.readonly".into()],
        extra: vec![("access_type".into(), "offline".into()), ("include_granted_scopes".into(), "true".into()), ("prompt".into(), "consent".into())],
        wait: Duration::from_secs(30),
        page: DonePage { app: "FilmCraft".into(), service: "Google Photos".into() },
    }
}

fn query(url: &str) -> Vec<(String, String)> {
    url.split_once('?')
        .map(|(_, q)| q)
        .unwrap_or_default()
        .split('&')
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (urldecode(k), urldecode(v))
        })
        .collect()
}

fn param<'a>(q: &'a [(String, String)], k: &str) -> &'a str {
    q.iter().find(|(key, _)| key == k).map(|(_, v)| v.as_str()).unwrap_or_default()
}

/// The browser: follows the consent page's redirect with `answer(state)` as the query string, and
/// returns the page the loopback listener served.
fn browser(seen_url: Arc<Mutex<String>>, answer: impl Fn(&str) -> String + Send + 'static) -> (impl FnMut(&str) -> bool, Arc<Mutex<String>>) {
    let page = Arc::new(Mutex::new(String::new()));
    let shown = page.clone();
    let open = move |url: &str| {
        *seen_url.lock().unwrap() = url.to_string();
        let q = query(url);
        let redirect = param(&q, "redirect_uri").to_string();
        let target = format!("/?{}", answer(param(&q, "state")));
        let shown = shown.clone();
        std::thread::spawn(move || {
            let addr = redirect.strip_prefix("http://").unwrap().to_string();
            let mut s = TcpStream::connect(addr).unwrap();
            write!(s, "GET {target} HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n").unwrap();
            let mut out = String::new();
            let _ = s.read_to_string(&mut out);
            *shown.lock().unwrap() = out;
        });
        true
    };
    (open, page)
}

const ANSWER: &str = r#"{"access_token":"ya29.fresh","expires_in":3599,"refresh_token":"1//refresh","scope":"https://www.googleapis.com/auth/photospicker.mediaitems.readonly","token_type":"Bearer"}"#;

#[test]
fn sign_in_with_pkce_end_to_end() {
    let endpoint = MockEndpoint::start(vec![(200, ANSWER.into())]);
    let url = Arc::new(Mutex::new(String::new()));
    let (mut open, page) = browser(url.clone(), |state| format!("state={state}&code=4%2F0AcodeXYZ&scope=x"));
    let mut said = Vec::new();
    let tokens = client(&endpoint).sign_in(&agent(), &request(), &mut open, &mut |m| said.push(m.to_string())).unwrap();
    assert_eq!((tokens.access_token.as_str(), tokens.refresh_token.as_deref()), ("ya29.fresh", Some("1//refresh")));
    assert!(tokens.grants("https://www.googleapis.com/auth/photospicker.mediaitems.readonly"));
    assert!(tokens.expires_at > filmcraft_oauth::now() + 3000);
    assert!(said.iter().any(|m| m.contains("opened the browser")), "{said:?}");

    // the consent URL: a loopback redirect on a random port, state, an S256 challenge
    let url = url.lock().unwrap().clone();
    let q = query(&url);
    let redirect = param(&q, "redirect_uri");
    assert!(redirect.starts_with("http://127.0.0.1:") && !redirect.ends_with(":8417"), "{redirect}");
    assert_eq!(param(&q, "response_type"), "code");
    assert_eq!(param(&q, "code_challenge_method"), "S256");
    assert_eq!(param(&q, "access_type"), "offline");
    assert_eq!(param(&q, "state").len(), 32);
    assert!(!url.contains("desktop-secret"), "the client secret stays out of the browser");

    // the exchange: a form POST carrying the code, the same redirect and the verifier
    let seen = endpoint.seen();
    assert_eq!(seen.len(), 1);
    let ex = &seen[0];
    assert_eq!((ex.method.as_str(), ex.path.as_str()), ("POST", "/token"));
    assert!(ex.content_type.starts_with("application/x-www-form-urlencoded"), "{}", ex.content_type);
    assert_eq!(ex.get("grant_type"), Some("authorization_code"));
    assert_eq!(ex.get("code"), Some("4/0AcodeXYZ"));
    assert_eq!(ex.get("redirect_uri"), Some(redirect));
    assert_eq!(ex.get("client_id"), Some("id.apps.googleusercontent.com"));
    assert_eq!(ex.get("client_secret"), Some("desktop-secret"));
    let verifier = ex.get("code_verifier").unwrap();
    assert!(filmcraft_oauth::pkce::Pkce::from_verifier(verifier).is_ok());
    assert_eq!(challenge_s256(verifier), param(&q, "code_challenge"), "the verifier answers the challenge");

    // the browser was told it can go back to the app
    std::thread::sleep(Duration::from_millis(100));
    let page = page.lock().unwrap().clone();
    assert!(page.contains("Signed in to Google Photos") && page.contains("return to FilmCraft"), "{page}");
}

#[test]
fn a_mismatched_state_is_refused_and_no_code_is_exchanged() {
    let endpoint = MockEndpoint::start(vec![(200, ANSWER.into())]);
    let (mut open, page) = browser(Arc::new(Mutex::new(String::new())), |_| "state=forged&code=stolen".to_string());
    let e = client(&endpoint).sign_in(&agent(), &request(), &mut open, &mut |_| {}).unwrap_err();
    assert!(matches!(e, OAuthError::SignIn(_)), "{e:?}");
    let msg = e.to_string();
    assert!(msg.contains("state did not match") && msg.contains("filmcraft-cli photos auth"), "{msg}");
    assert!(endpoint.seen().is_empty(), "the code was never exchanged");
    std::thread::sleep(Duration::from_millis(100));
    assert!(page.lock().unwrap().starts_with("HTTP/1.1 400"));
}

#[test]
fn a_declined_consent_says_so() {
    let endpoint = MockEndpoint::start(vec![]);
    let (mut open, _) = browser(Arc::new(Mutex::new(String::new())), |state| format!("state={state}&error=access_denied"));
    let e = client(&endpoint).sign_in(&agent(), &request(), &mut open, &mut |_| {}).unwrap_err();
    assert!(e.to_string().contains("declined"), "{e}");
}

#[test]
fn without_a_browser_the_link_is_given() {
    let endpoint = MockEndpoint::start(vec![]);
    let mut said = Vec::new();
    let req = SignInRequest { wait: Duration::from_millis(200), ..request() };
    let e = client(&endpoint).sign_in(&agent(), &req, &mut |_| false, &mut |m| said.push(m.to_string())).unwrap_err();
    assert!(said.iter().any(|m| m.contains("open this link") && m.contains("https://accounts.example.test/")), "{said:?}");
    assert!(e.to_string().contains("no sign-in came back") && e.to_string().contains("photos auth"), "{e}");
}

fn saved() -> TokenSet {
    TokenSet {
        access_token: "ya29.old".into(),
        refresh_token: Some("1//refresh".into()),
        expires_at: 1,
        scope: "https://www.googleapis.com/auth/photospicker.mediaitems.readonly https://www.googleapis.com/auth/photoslibrary.appendonly".into(),
    }
}

#[test]
fn refresh_keeps_what_the_answer_leaves_out() {
    let endpoint = MockEndpoint::start(vec![(200, r#"{"access_token":"ya29.new","expires_in":3599,"token_type":"Bearer"}"#.into())]);
    let t = client(&endpoint).refresh(&agent(), &saved()).unwrap();
    assert_eq!((t.access_token.as_str(), t.refresh_token.as_deref(), t.scope.as_str()), ("ya29.new", Some("1//refresh"), saved().scope.as_str()));
    let seen = endpoint.seen();
    assert_eq!(seen[0].get("grant_type"), Some("refresh_token"));
    assert_eq!(seen[0].get("refresh_token"), Some("1//refresh"));
    assert_eq!(seen[0].get("client_secret"), Some("desktop-secret"));
    assert_eq!(seen[0].get("code_verifier"), None);
}

/// Google answers a lapsed (Testing mode, 7 days) or revoked refresh token with `invalid_grant`:
/// the user hears that they must sign in again, and how — never the raw answer.
#[test]
fn a_refused_refresh_says_to_sign_in_again() {
    let endpoint = MockEndpoint::start(vec![(400, r#"{"error":"invalid_grant","error_description":"Token has been expired or revoked."}"#.into())]);
    let e = client(&endpoint).refresh(&agent(), &saved()).unwrap_err();
    assert!(matches!(e, OAuthError::SignInAgain { .. }), "{e:?}");
    let msg = e.to_string();
    assert!(msg.starts_with("Google no longer accepts the saved sign-in"), "{msg}");
    assert!(msg.contains("7 days") && msg.ends_with("sign in again: `filmcraft-cli photos auth`"), "{msg}");
    assert!(!msg.contains("invalid_grant") && !msg.contains("{"), "no raw answer: {msg}");
}

#[test]
fn other_token_failures_are_readable() {
    let endpoint = MockEndpoint::start(vec![
        (401, r#"{"error":"invalid_client","error_description":"The OAuth client was not found."}"#.into()),
        (500, "<html>oops</html>".into()),
        (400, r#"{"error":"invalid_grant"}"#.into()),
        (200, "not json".into()),
        (200, format!(r#"{{"access_token":"{}"}}"#, "a".repeat(70_000))),
    ]);
    let c = client(&endpoint);
    let e = c.refresh(&agent(), &saved()).unwrap_err();
    assert!(matches!(e, OAuthError::Client(_)) && e.to_string().contains("client was not found") && e.to_string().contains("photos auth"), "{e}");
    let e = c.refresh(&agent(), &saved()).unwrap_err();
    assert!(matches!(e, OAuthError::Token(_)) && e.to_string().contains("500") && e.to_string().contains("oops"), "{e}");
    // a refused code (not a refresh) means: try the sign-in again
    let pkce = filmcraft_oauth::pkce::Pkce::generate().unwrap();
    let e = c.exchange_code(&agent(), "code", "http://127.0.0.1:1", &pkce).unwrap_err();
    assert!(matches!(e, OAuthError::SignIn(_)) && e.to_string().contains("refused the sign-in code"), "{e}");
    assert!(c.refresh(&agent(), &saved()).unwrap_err().to_string().contains("unreadable JSON"));
    assert!(c.refresh(&agent(), &saved()).unwrap_err().to_string().contains("too large"));
    // nobody listening
    let closed = TcpListener::bind("127.0.0.1:0").unwrap();
    let dead = format!("http://{}/token", closed.local_addr().unwrap());
    drop(closed);
    let c = OAuthClient { provider: Provider { token_url: dead, ..c.provider.clone() }, ..c };
    let e = c.refresh(&agent(), &saved()).unwrap_err();
    assert!(matches!(e, OAuthError::Token(_)) && e.to_string().contains("could not reach"), "{e}");
}

#[test]
fn revocation() {
    let endpoint = MockEndpoint::start(vec![(200, "{}".into()), (400, r#"{"error":"invalid_token"}"#.into()), (503, "{}".into())]);
    let c = client(&endpoint);
    c.revoke(&agent(), "1//refresh").unwrap();
    c.revoke(&agent(), "1//gone").unwrap();
    assert!(c.revoke(&agent(), "1//x").unwrap_err().to_string().contains("did not revoke"));
    let seen = endpoint.seen();
    assert_eq!((seen[0].path.as_str(), seen[0].get("token")), ("/revoke", Some("1//refresh")));
    let none = OAuthClient { provider: Provider { revoke_url: None, ..c.provider.clone() }, ..c };
    assert!(none.revoke(&agent(), "t").is_err());
}
