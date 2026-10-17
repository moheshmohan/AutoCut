//! The Instagram adapter against a mock Instagram on 127.0.0.1 (no network, no Keychain: a
//! memory secret store): setting the app up, the browser sign-in (loopback and pasted), a pasted
//! dashboard token, refresh, the resumable upload interrupted and resumed (within a run and
//! across runs), processing (FINISHED, ERROR, EXPIRED, still busy), publishing, the limits, and
//! hostile answers — errors, never panics.

mod mock;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use filmcraft_engine::instagram::ReelRequest;
use filmcraft_instagram::http::{UreqTransport, local_agent};
use filmcraft_instagram::{Endpoints, IgError, Instagram, PollOptions, Secrets, now};
use filmcraft_oauth::{MemoryStore, SecretStore};
use mock::{Mock, SECRET, USER_ID, graph_error};
use serde_json::json;

fn temp(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("filmcraft-ig-flow-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

struct World {
    mock: Mock,
    root: PathBuf,
    store: Arc<MemoryStore>,
}

impl World {
    fn new(tag: &str) -> Self {
        Self { mock: Mock::start(), root: temp(tag), store: Arc::new(MemoryStore::new()) }
    }

    fn ig(&self) -> Instagram {
        let mut ig = Instagram::in_data_dir(&self.root)
            .with_endpoints(Endpoints::at(&self.mock.base))
            .with_transport(Arc::new(UreqTransport::with_agent(local_agent(Duration::from_secs(20)))))
            .with_secrets(Secrets::Given(self.store.clone() as Arc<dyn SecretStore>))
            .with_opener(self.mock.browser());
        ig.sign_in_wait = Duration::from_secs(20);
        ig.backoff = Duration::from_millis(5);
        ig.upload_opts.backoff = Duration::from_millis(5);
        ig.poll = PollOptions { first: Duration::from_millis(10), every: Duration::from_millis(20), limit: Duration::from_secs(5) };
        ig
    }

    /// Connected with a dashboard token.
    fn connected(&self) -> Instagram {
        let ig = self.ig();
        let t = self.mock.dashboard_token();
        ig.connect_token(&t, &mut |_| {}).unwrap();
        ig
    }

    fn video(&self, name: &str, len: usize) -> PathBuf {
        let p = self.root.join(name);
        let bytes: Vec<u8> = (0..len).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&p, bytes).unwrap();
        p
    }
}

impl Drop for World {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.root);
    }
}

fn reel(path: &std::path::Path, caption: &str) -> ReelRequest {
    ReelRequest { path: Some(path.to_string_lossy().into_owned()), caption: Some(caption.into()), share_to_feed: true, ..Default::default() }
}

/// Containers asked for (`POST …/media`, not `…/media_publish`).
fn containers_made(w: &World) -> usize {
    w.mock.seen("POST", &format!("/v25.0/{USER_ID}/media")).iter().filter(|r| r.path == format!("/v25.0/{USER_ID}/media")).count()
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn quiet() -> impl FnMut(u64, u64, &str) -> bool {
    |_, _, _| true
}

#[test]
fn setup_and_the_browser_sign_in_over_loopback() {
    let w = World::new("login");
    let ig = w.ig();
    assert!(ig.connect_readiness().unwrap_err().contains("instagram setup"));
    let e = ig.connect(None, &mut |_| {}).unwrap_err();
    assert!(matches!(e, IgError::Setup(_)), "{e}");
    let redirect = format!("http://127.0.0.1:{}/", free_port());
    assert!(ig.setup("2133883020551415", &redirect, "short").is_err(), "not a secret");
    let place = ig.setup("2133883020551415", &redirect, SECRET).unwrap();
    assert_eq!(place, "memory");
    assert!(ig.connect_readiness().is_ok());
    let app_json = std::fs::read_to_string(w.root.join("instagram/app.json")).unwrap();
    assert!(!app_json.contains(SECRET), "the secret never lands in app.json");

    let mut said = Vec::new();
    let c = ig.connect(None, &mut |m| said.push(m.to_string())).unwrap();
    assert_eq!((c.account.username.as_str(), c.account.user_id.as_str()), ("moheshmohan", USER_ID));
    assert!(c.account.grants("instagram_business_content_publish"));
    assert!(c.account.expires_at >= now() + 59 * 86_400, "a 60-day token");
    // the code exchange: form fields with the secret and the registered redirect
    let ex = w.mock.seen("POST", "/oauth/access_token");
    assert_eq!(ex.len(), 1);
    assert_eq!((ex[0].form("redirect_uri").as_deref(), ex[0].form("grant_type").as_deref()), (Some(redirect.as_str()), Some("authorization_code")));
    assert_eq!(ex[0].form("client_id").as_deref(), Some("2133883020551415"));
    // then the long-lived exchange
    let long = w.mock.seen("GET", "/access_token?");
    assert_eq!(long.len(), 1);
    assert_eq!(long[0].query("grant_type").as_deref(), Some("ig_exchange_token"));
    // kept in the store, not in the account file
    let file = std::fs::read_to_string(w.root.join("instagram/account.json")).unwrap();
    assert!(!file.contains("IGAA"), "no token in account.json: {file}");
    assert!(!w.store.is_empty());
    assert!(ig.publish_readiness().is_ok());
    let l = ig.listing().unwrap();
    assert_eq!(l["account"]["username"], "moheshmohan");
    assert_eq!(l["signedIn"], true);
    assert!(said.iter().all(|m| !m.contains("IGAA")), "no token in progress lines: {said:?}");
}

#[test]
fn a_web_page_redirect_is_pasted_back() {
    let w = World::new("paste");
    let ig = w.ig();
    ig.setup("2133883020551415", "https://example.com/instagram", SECRET).unwrap();
    // the engine (no terminal) cannot paste: it says what to do
    assert!(ig.connect(None, &mut |_| {}).unwrap_err().to_string().contains("pasting its address"));
    let mut paste = |url: &str| {
        let state = url.split("state=").nth(1).unwrap().split('&').next().unwrap().to_string();
        Some(format!("https://example.com/instagram?code=AQDcode123&state={state}#_"))
    };
    let c = ig.connect(Some(&mut paste), &mut |_| {}).unwrap();
    assert_eq!(c.account.username, "moheshmohan");
    // a stale address (another state) connects nothing
    let mut stale = |_: &str| Some("https://example.com/instagram?code=AQDcode123&state=old#_".to_string());
    assert!(ig.connect(Some(&mut stale), &mut |_| {}).unwrap_err().to_string().contains("state does not match"));
    // a declined consent
    let mut declined = |url: &str| {
        let state = url.split("state=").nth(1).unwrap().split('&').next().unwrap().to_string();
        Some(format!("https://example.com/instagram?error=access_denied&error_reason=user_denied&state={state}"))
    };
    assert!(ig.connect(Some(&mut declined), &mut |_| {}).unwrap_err().to_string().contains("declined"));
    // a wrong secret is a setup problem
    ig.setup("2133883020551415", "https://example.com/instagram", "ffffffffffffffffffffffffffffffff").unwrap();
    let e = ig.connect(Some(&mut paste), &mut |_| {}).unwrap_err();
    assert!(matches!(e, IgError::Setup(_)) && e.to_string().contains("instagram setup"), "{e}");
}

#[test]
fn a_pasted_dashboard_token_and_its_refresh() {
    let w = World::new("token");
    let ig = w.ig();
    assert!(ig.connect_token("not a token", &mut |_| {}).is_err());
    let e = ig.connect_token("IGAAunknown000000000000000", &mut |_| {}).unwrap_err();
    assert!(matches!(e, IgError::Auth(_)), "an unknown token: {e}");
    let t = w.mock.dashboard_token();
    let c = ig.connect_token(&t, &mut |_| {}).unwrap();
    assert_eq!(c.account.via, filmcraft_instagram::Via::Token);
    assert_eq!(w.mock.seen("GET", "/refresh_access_token").len(), 1, "a refresh tells when it ends");
    // a refresh refused (too new) keeps the pasted token, 60 days from now
    w.mock.with(|s| s.refresh_refused = Some(graph_error(100, 0, "The access token is not eligible for refresh")));
    let t2 = w.mock.dashboard_token();
    let c = ig.connect_token(&t2, &mut |_| {}).unwrap();
    assert!(c.account.expires_at >= now() + 59 * 86_400);
    // a personal account cannot publish
    w.mock.with(|s| s.me_answer = Some((200, json!({"user_id": USER_ID, "username": "someone", "account_type": "PERSONAL"}).to_string())));
    let t3 = w.mock.dashboard_token();
    let e = ig.connect_token(&t3, &mut |_| {}).unwrap_err();
    assert!(matches!(e, IgError::Consent(_)) && e.to_string().contains("professional"), "{e}");
}

#[test]
fn publishing_a_reel_with_a_dropped_upload_resumed() {
    let w = World::new("publish");
    let ig = w.connected();
    let file = w.video("show.mp4", 1_200_000);
    w.mock.with(|s| {
        s.upload_drops.insert(0, 400_000);
        s.statuses.extend([("IN_PROGRESS".to_string(), String::new()), ("IN_PROGRESS".to_string(), String::new())]);
    });
    let mut last = (0, 0);
    let p = ig
        .publish(&reel(&file, "Every robot #robots @friend"), &mut |sent, total, _| {
            assert!(sent <= total);
            last = (sent, total);
            true
        })
        .unwrap();
    assert_eq!(last, (1_200_000, 1_200_000));
    assert_eq!(p.source, "upload");
    assert_eq!(p.sent.resumed, 1, "{:?}", p.sent);
    assert!(p.permalink.as_deref().unwrap().starts_with("https://www.instagram.com/reel/"));
    // every byte arrived once, in order
    let original = std::fs::read(&file).unwrap();
    let c = w.mock.with(|s| s.containers.values().next().cloned().unwrap());
    assert_eq!(c.received, original);
    assert!(c.resumable && c.published);
    let form = |k: &str| c.form.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
    assert_eq!((form("media_type").as_deref(), form("share_to_feed").as_deref()), (Some("REELS"), Some("true")));
    assert_eq!(form("caption").as_deref(), Some("Every robot #robots @friend"));
    // the upload POSTs: OAuth header, offsets
    let ups = w.mock.seen("POST", "/ig-api-upload/");
    assert_eq!(ups.len(), 2);
    assert!(ups.iter().all(|u| u.header("authorization").unwrap().starts_with("OAuth IGAA")));
    assert_eq!((ups[0].header("offset"), ups[1].header("offset")), (Some("0"), Some("400000")));
    assert_eq!(ups[1].header("file_size"), Some("1200000"));
    // graph calls carry the token in a header, never in the URL
    assert!(w.mock.seen("GET", "/v25.0/").iter().all(|r| !r.path.contains("IGAA") && r.header("authorization").unwrap().starts_with("Bearer ")));
    // counted
    let l = ig.limit_book().to_json(now());
    assert_eq!((l["posts"].as_u64(), l["containers"].as_u64()), (Some(1), Some(1)));
    assert!(w.root.join("instagram/sessions").read_dir().map(|d| d.count()).unwrap_or(0) == 0, "the kept container is gone once published");
}

#[test]
fn a_cancelled_upload_resumes_on_the_next_run_without_a_new_container() {
    let w = World::new("cancel");
    let ig = w.connected();
    let file = w.video("show.mp4", 900_000);
    let req = reel(&file, "take one");
    let e = ig.publish(&req, &mut |sent, _, _| sent < 300_000).unwrap_err();
    assert_eq!(e, IgError::Cancelled);
    assert_eq!(containers_made(&w), 1);
    // the mock keeps what came before the client stopped (its thread may still be reading)
    let mut got = 0;
    for _ in 0..200 {
        got = w.mock.with(|s| s.containers.values().next().unwrap().received.len());
        if got > 0 && w.mock.seen("POST", "/ig-api-upload/").len() == 1 {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    assert!(got > 0 && got < 900_000, "{got}");
    // the next run carries on: the kept container, from where Instagram got to
    let p = ig.publish(&req, &mut quiet()).unwrap();
    assert!(p.from_earlier && p.sent.resumed == 1 && p.sent.restarted == 0, "{p:?}");
    assert_eq!(containers_made(&w), 1, "no second container");
    let ups = w.mock.seen("POST", "/ig-api-upload/");
    assert_eq!(ups.last().unwrap().header("offset"), Some(got.to_string().as_str()), "sent from where it stopped");
    assert_eq!(w.mock.with(|s| s.containers.values().next().unwrap().received.clone()), std::fs::read(&file).unwrap());
    // once published, the kept container is forgotten: the same publish again is a new post
    let p2 = ig.publish(&req, &mut quiet()).unwrap();
    assert!(!p2.from_earlier);
    assert_eq!(containers_made(&w), 2);
}

#[test]
fn processing_errors_expiry_and_a_slow_container() {
    let w = World::new("processing");
    let ig = w.connected();
    let file = w.video("bad.mp4", 50_000);
    w.mock.with(|s| s.statuses.push_back(("ERROR".into(), "Error: 2207026".into())));
    let e = ig.publish(&reel(&file, "a"), &mut quiet()).unwrap_err();
    assert!(matches!(e, IgError::Processing(_)) && e.to_string().contains("H.264"), "{e}");
    w.mock.with(|s| s.statuses.push_back(("EXPIRED".into(), String::new())));
    let e = ig.publish(&reel(&file, "b"), &mut quiet()).unwrap_err();
    assert!(e.to_string().contains("expired"), "{e}");
    // still busy after the limit: kept, and the next run finishes without uploading again
    let mut ig = ig;
    ig.poll.limit = Duration::from_millis(60);
    w.mock.with(|s| s.never_finish = true);
    let req = reel(&file, "slow");
    let e = ig.publish(&req, &mut quiet()).unwrap_err();
    assert!(e.to_string().contains("still processing"), "{e}");
    let uploads = w.mock.seen("POST", "/ig-api-upload/").len();
    w.mock.with(|s| s.never_finish = false);
    let p = ig.publish(&req, &mut quiet()).unwrap();
    assert!(p.from_earlier);
    assert_eq!(w.mock.seen("POST", "/ig-api-upload/").len(), uploads, "not uploaded again");
}

#[test]
fn limits_stop_before_spending_and_meta_errors_read_plainly() {
    let w = World::new("limits");
    let ig = w.connected();
    let file = w.video("v.mp4", 20_000);
    // Instagram says the day's posts are used up: the container waits, nothing is published
    w.mock.with(|s| s.limit = (100, 100));
    let e = ig.publish(&reel(&file, "x"), &mut quiet()).unwrap_err();
    assert!(matches!(e, IgError::Limit(_)) && e.to_string().contains("(100)"), "{e}");
    assert!(w.mock.seen("POST", &format!("/v25.0/{USER_ID}/media_publish")).is_empty());
    // the publish refused for the limit (another app spent it)
    w.mock.with(|s| {
        s.limit = (3, 100);
        s.publish_answer = Some((400, graph_error(9, 2207042, "limit")));
    });
    let e = ig.publish(&reel(&file, "y"), &mut quiet()).unwrap_err();
    assert!(matches!(e, IgError::Limit(_)) && e.to_string().contains("tomorrow"), "{e}");
    w.mock.with(|s| s.publish_answer = None);
    // an app rate limit with its usage header
    w.mock.with(|s| s.create_answers.push_back((400, graph_error(4, 0, "Application request limit reached"))));
    let e = ig.publish(&reel(&file, "z"), &mut quiet()).unwrap_err();
    assert!(matches!(e, IgError::Limit(_)), "{e}");
    // 400 containers made here today: refused before asking Instagram
    let before = containers_made(&w);
    for _ in 0..400 {
        ig.limit_book().record_container(now());
    }
    let e = ig.publish(&reel(&file, "w"), &mut quiet()).unwrap_err();
    assert!(e.to_string().contains("400"), "{e}");
    assert_eq!(containers_made(&w), before);
    // instagram.limits: Instagram's count
    let l = ig.limits().unwrap();
    assert_eq!(l["instagram"]["quotaTotal"], 100);
}

#[test]
fn the_token_is_refreshed_once_a_day_old_and_a_refused_one_asks_to_connect_again() {
    let w = World::new("refresh");
    let ig = w.connected();
    let file = w.video("v.mp4", 10_000);
    // make the sign-in two days old
    let path = w.root.join("instagram/account.json");
    let mut a: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    a["refreshedAt"] = json!(now() - 2 * 86_400);
    std::fs::write(&path, a.to_string()).unwrap();
    let refreshes = w.mock.seen("GET", "/refresh_access_token").len();
    ig.publish(&reel(&file, "r"), &mut quiet()).unwrap();
    assert_eq!(w.mock.seen("GET", "/refresh_access_token").len(), refreshes + 1);
    let last_up = w.mock.seen("POST", "/ig-api-upload/").pop().unwrap();
    assert!(last_up.header("authorization").unwrap().starts_with("OAuth IGAArefreshed"), "the refreshed token is used");
    // a refresh Instagram refuses (expired, revoked): connect again
    let mut a: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    a["refreshedAt"] = json!(now() - 2 * 86_400);
    std::fs::write(&path, a.to_string()).unwrap();
    w.mock.with(|s| s.refresh_refused = Some(graph_error(190, 460, "Error validating access token: Session has expired")));
    let e = ig.publish(&reel(&file, "s"), &mut quiet()).unwrap_err();
    assert!(matches!(e, IgError::Auth(_)) && e.to_string().contains("instagram auth"), "{e}");
    // expired: said without the network
    let mut a: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    a["expiresAt"] = json!(now() - 1);
    std::fs::write(&path, a.to_string()).unwrap();
    assert!(ig.publish_readiness().unwrap_err().contains("ran out"));
    let seen = w.mock.with(|s| s.seen.len());
    assert!(ig.publish(&reel(&file, "t"), &mut quiet()).is_err());
    assert_eq!(w.mock.with(|s| s.seen.len()), seen, "no call with an expired sign-in");
}

#[test]
fn a_public_url_is_fetched_by_instagram() {
    let w = World::new("url");
    let ig = w.connected();
    let req = ReelRequest {
        video_url: Some("https://cdn.example.com/show.mp4".into()),
        caption: Some("from a URL".into()),
        share_to_feed: false,
        cover_ms: Some(1500),
        ..Default::default()
    };
    let p = ig.publish(&req, &mut quiet()).unwrap();
    assert_eq!(p.source, "url");
    assert!(w.mock.seen("POST", "/ig-api-upload/").is_empty());
    let c = w.mock.with(|s| s.containers.values().next().cloned().unwrap());
    let form = |k: &str| c.form.iter().find(|(a, _)| a == k).map(|(_, v)| v.clone());
    assert_eq!(
        (form("video_url").as_deref(), form("share_to_feed").as_deref(), form("thumb_offset").as_deref()),
        (Some("https://cdn.example.com/show.mp4"), Some("false"), Some("1500"))
    );
    assert_eq!(form("upload_type"), None);
}

#[test]
fn hostile_answers_are_errors_not_panics() {
    let w = World::new("hostile");
    let ig = w.connected();
    let file = w.video("v.mp4", 30_000);
    let bodies = [
        (200, r#"{"id":"../../etc"}"#.to_string()),
        (200, r#"{"id":-5}"#.to_string()),
        (200, "not json".to_string()),
        (200, format!("{{\"id\":\"{}\"}}", "9".repeat(100))),
        (500, "x".repeat(300_000)),
        (400, r#"{"error":"just a string"}"#.to_string()),
        (400, r#"{"error":{"code":"190","message":[1,2]}}"#.to_string()),
    ];
    for (status, body) in bodies {
        w.mock.with(|s| {
            s.create_answers.clear();
            for _ in 0..6 {
                s.create_answers.push_back((status, body.clone()));
            }
        });
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| ig.publish(&reel(&file, &format!("h{status}{}", body.len())), &mut quiet())));
        assert!(matches!(r, Ok(Err(_))), "{status} {:.60}: {r:?}", body);
    }
    w.mock.with(|s| s.create_answers.clear());
    // an upload URI elsewhere is not given the token
    w.mock.with(|s| s.uri = Some("https://evil.example/steal".into()));
    ig.publish(&reel(&file, "uri"), &mut quiet()).unwrap();
    assert!(w.mock.seen("POST", "/ig-api-upload/").iter().all(|u| u.path.contains("/ig-api-upload/v25.0/")));
    w.mock.with(|s| s.uri = None);
    // a refused upload says what to do
    w.mock.with(|s| {
        let i = s.uploads;
        s.upload_answers.insert(i, (400, r#"{"debug_info":{"retriable":false,"type":"ProcessingFailedError","message":"unauthorized user request"}}"#.into()));
    });
    let e = ig.publish(&reel(&file, "refused"), &mut quiet()).unwrap_err();
    assert!(e.to_string().contains("--video-url"), "{e}");
    // statuses without video_status still work; odd /me answers are errors
    w.mock.with(|s| {
        s.no_video_status = true;
        let i = s.uploads;
        s.upload_drops.insert(i, 1000);
    });
    ig.publish(&reel(&file, "no video_status"), &mut quiet()).unwrap();
    for me in [json!({"user_id": "x1", "username": "a"}), json!({"username": "<script>"}), json!([1, 2])] {
        w.mock.with(|s| s.me_answer = Some((200, me.to_string())));
        let t = w.mock.dashboard_token();
        assert!(ig.connect_token(&t, &mut |_| {}).is_err(), "{me}");
    }
}

#[test]
fn the_engine_runs_the_real_publisher() {
    use filmcraft_engine::instagram::InstagramPublisher as _;
    let w = World::new("engine");
    let ig = w.connected();
    let file = w.video("e.mp4", 40_000);
    let publisher = Arc::new(filmcraft_instagram::EnginePublisher::new(ig));
    assert!(publisher.ready().is_ok());
    let mut s = filmcraft_engine::Session::default();
    s.instagram.publisher = Some(publisher.clone());
    let v = s.execute("instagram.account", json!({})).unwrap();
    assert_eq!(v["account"]["username"], "moheshmohan");
    let v = s.execute("instagram.publish", json!({"path": file.to_string_lossy(), "caption": "engine #test", "wait": true})).unwrap();
    assert!(v["instagram"]["mediaId"].as_str().is_some(), "{v}");
    let jobs = s.execute("jobs.list", json!({})).unwrap();
    assert!(jobs.as_array().unwrap().iter().any(|j| j["instagram"]["permalink"].as_str().is_some()), "{jobs}");
    let e = s.execute("instagram.publish", json!({"path": file.to_string_lossy(), "caption": "#a ".repeat(31)})).unwrap_err();
    assert!(e.to_string().contains("31 hashtags"), "{e}");
    let v = s.execute("instagram.disconnect", json!({})).unwrap();
    assert!(v["note"].as_str().unwrap().contains("Apps and websites"));
    assert!(publisher.ready().is_err());
}

#[test]
fn upload_only_stages_the_reel_and_the_next_run_publishes_it() {
    let w = World::new("staged");
    let ig = w.connected();
    let file = w.video("staged.mp4", 70_000);
    let mut req = reel(&file, "staged");
    req.upload_only = true;
    let p = ig.publish(&req, &mut quiet()).unwrap();
    assert!(p.media_id.is_none() && p.to_json()["published"] == false);
    assert!(w.mock.with(|s| s.published.is_empty()), "nothing published");
    let uploads = w.mock.seen("POST", "/ig-api-upload/").len();
    req.upload_only = false;
    let p = ig.publish(&req, &mut quiet()).unwrap();
    assert!(p.from_earlier && p.media_id.is_some());
    assert_eq!(w.mock.seen("POST", "/ig-api-upload/").len(), uploads, "not sent again");
    assert_eq!(containers_made(&w), 1);
}
