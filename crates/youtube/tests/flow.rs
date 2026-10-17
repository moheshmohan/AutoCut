//! The YouTube adapter offline, against a mock Google on 127.0.0.1 (tests/mock): sign-ins that
//! connect channels (the account chooser picks one per sign-in), tokens kept per channel, the
//! default, disconnecting; the resumable upload — dropped chunks, a server error, an interrupted
//! run resumed by the next, an expired session started again — with the metadata, chapters,
//! thumbnail and playlist; quota counting and Google's errors in words; hostile answers.

mod mock;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use filmcraft_engine::youtube::{ConnectOptions, Privacy, UploadRequest, VideoMeta};
use filmcraft_oauth::{MemoryStore, SecretStore};
use filmcraft_youtube::http::{UreqTransport, local_agent};
use filmcraft_youtube::upload::GRANULE;
use filmcraft_youtube::{Endpoints, Secrets, YouTube, YtError};
use mock::{Fault, Mock, error_body};
use serde_json::{Value, json};

const H4: &str = "UCh4hacks000000000000001";
const MM: &str = "UCmohesh0000000000000002";

fn temp_root(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("filmcraft-yt-flow-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("google-photos")).unwrap();
    std::fs::write(
        d.join("google-photos/client_secret.json"),
        r#"{"installed":{"client_id":"id.apps.googleusercontent.com","client_secret":"desktop-secret"}}"#,
    )
    .unwrap();
    d
}

struct Rig {
    mock: Mock,
    yt: YouTube,
    store: Arc<MemoryStore>,
    root: PathBuf,
}

fn rig(tag: &str) -> Rig {
    let mock = Mock::start();
    let root = temp_root(tag);
    let store = Arc::new(MemoryStore::new());
    let mut yt = YouTube::in_data_dir(&root)
        .with_endpoints(Endpoints::at(&mock.base))
        .with_transport(Arc::new(UreqTransport::with_agent(local_agent(Duration::from_secs(20)))), local_agent(Duration::from_secs(20)))
        .with_secrets(Secrets::Store(store.clone()))
        .with_opener(mock.browser());
    yt.sign_in_wait = Duration::from_secs(20);
    yt.upload_opts.chunk = GRANULE;
    yt.upload_opts.backoff = Duration::from_millis(5);
    Rig { mock, yt, store, root }
}

fn connect(r: &Rig, pick: &str, opts: ConnectOptions) -> Result<filmcraft_youtube::Connected, YtError> {
    r.mock.with(|s| s.picked = pick.into());
    r.yt.connect(opts, &mut |_| {})
}

/// A file of `n` bytes, each its offset's low byte (so a misplaced chunk shows).
fn video(root: &Path, name: &str, n: usize) -> PathBuf {
    let p = root.join(name);
    std::fs::write(&p, (0..n).map(|i| (i % 251) as u8).collect::<Vec<_>>()).unwrap();
    p
}

fn request(path: &Path, title: &str) -> UploadRequest {
    UploadRequest {
        path: path.to_string_lossy().into_owned(),
        channel: None,
        meta: VideoMeta { title: title.into(), ..Default::default() },
        playlist: None,
        thumbnail: None,
        sidecars: true,
    }
}

fn received(r: &Rig, session: u32) -> Vec<u8> {
    r.mock.with(|s| s.sessions.get(&session).map(|x| x.received.clone()).unwrap_or_default())
}

#[test]
fn channels_are_connected_one_sign_in_each_and_kept_apart() {
    let r = rig("connect");
    let c = connect(&r, "h4hacks", ConnectOptions::default()).unwrap();
    assert_eq!((c.channel.id.as_str(), c.channel.title.as_str(), c.channel.handle.as_str()), (H4, "H4hacks", "@h4hacks"));
    assert!(c.is_default && !c.replaced, "the first channel is the default");
    assert_eq!(c.channel.long_uploads, "allowed");
    // the consent asked for YouTube's two scopes and the account chooser
    let consent = r.mock.seen("POST", "/token")[0].clone();
    assert_eq!(consent.form("grant_type").as_deref(), Some("authorization_code"));
    assert!(consent.form("code_verifier").is_some_and(|v| v.len() >= 43), "PKCE");
    let c2 = connect(&r, "mohesh", ConnectOptions::default()).unwrap();
    assert!(!c2.is_default && c2.channel.id == MM);
    // each channel its own secret-store account, holding its own refresh token
    assert_eq!(r.store.len(), 2);
    let t = |id: &str| {
        let v = r.yt.store().vault(id).unwrap();
        r.store.get(&v.account()).unwrap().map(|b| String::from_utf8(b).unwrap()).unwrap()
    };
    assert!(t(H4).contains("rt-h4hacks") && !t(H4).contains("rt-mohesh"));
    assert!(t(MM).contains("rt-mohesh") && !t(MM).contains("rt-h4hacks"));
    // the listing: no secret in it
    let l = r.yt.listing().unwrap();
    assert_eq!(l["channels"].as_array().unwrap().len(), 2, "{l}");
    assert_eq!(l["default"]["id"], H4);
    assert!(!l.to_string().contains("rt-") && !l.to_string().contains("at-"), "{l}");
    assert_eq!(l["quota"]["calls"]["channels.list"], 2, "{l}");
    // connecting a channel again replaces its sign-in; --default makes it the default
    let again = connect(&r, "mohesh", ConnectOptions { make_default: true, playlists: false }).unwrap();
    assert!(again.replaced && again.is_default);
    assert_eq!(r.yt.listing().unwrap()["channels"].as_array().unwrap().len(), 2);
    assert_eq!(r.yt.set_default("@H4HACKS").unwrap().id, H4);
    assert!(r.yt.set_default("Life Of 'S'").unwrap_err().to_string().contains("no connected YouTube channel"));
    // disconnecting revokes that channel's grant only, and the default moves on
    let d = r.yt.disconnect("H4hacks").unwrap();
    assert!(d.revoked.is_ok() && d.default.as_ref().map(|c| c.id.as_str()) == Some(MM), "{d:?}");
    assert_eq!(r.mock.with(|s| s.revoked.clone()), ["rt-h4hacks"]);
    assert!(r.yt.store().vault(H4).unwrap().load().unwrap().is_none());
    assert!(r.yt.store().vault(MM).unwrap().load().unwrap().is_some(), "the other channel keeps its sign-in");
    assert_eq!(r.store.len(), 1);
    assert!(!r.yt.store().channel_dir(H4).unwrap().exists());
    let _ = std::fs::remove_dir_all(&r.root);
}

#[test]
fn a_sign_in_without_a_channel_or_a_permission_keeps_nothing() {
    let r = rig("nochannel");
    let e = connect(&r, "nochannel", ConnectOptions::default()).unwrap_err();
    assert!(matches!(e, YtError::Channel(_)) && e.to_string().contains("create_channel"), "{e}");
    assert_eq!(r.mock.with(|s| s.revoked.len()), 1, "the unused grant is given back");
    // the user unticked the upload permission
    r.mock.with(|s| s.grant = Some("https://www.googleapis.com/auth/youtube.readonly".into()));
    let e = connect(&r, "h4hacks", ConnectOptions::default()).unwrap_err();
    assert!(matches!(e, YtError::Consent(_)) && e.to_string().contains("youtube auth"), "{e}");
    assert!(r.store.is_empty() && r.yt.listing().unwrap()["channels"].as_array().unwrap().is_empty());
    // no client file: say where it goes
    std::fs::remove_file(r.root.join("google-photos/client_secret.json")).unwrap();
    let e = r.yt.connect(ConnectOptions::default(), &mut |_| {}).unwrap_err().to_string();
    assert!(e.contains("client_secret.json") && e.contains("YouTube Data API v3"), "{e}");
    let _ = std::fs::remove_dir_all(&r.root);
}

#[test]
fn an_upload_resumes_after_dropped_chunks_and_server_errors() {
    let r = rig("resume");
    connect(&r, "h4hacks", ConnectOptions::default()).unwrap();
    connect(&r, "mohesh", ConnectOptions::default()).unwrap();
    let n = 4 * GRANULE as usize + 1000;
    let path = video(&r.root, "show.mp4", n);
    // what `showcase --export` leaves beside the video
    std::fs::write(r.root.join("show.chapters.txt"), "0:00 Intro\n0:12 #1 Ameca\n0:40 #2 Optimus\n").unwrap();
    std::fs::write(r.root.join("show.thumbnail.jpg"), [0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3]).unwrap();
    r.mock.with(|s| {
        s.chunk_faults.insert(1, Fault::Drop(100_000));
        s.chunk_faults.insert(3, Fault::Status(503, error_body(503, "backendError", "Backend Error")));
    });
    let mut req = request(&path, "Every Robot");
    req.channel = Some("@moheshmohan".into());
    req.meta.description = Some("Robots, 2020-2026".into());
    req.meta.tags = vec!["robots".into(), "expo 2020".into()];
    req.meta.notify_subscribers = false;
    let mut statuses = Vec::new();
    let up =
        r.yt.upload(&req, &mut |sent, total, status| {
            assert!(sent <= total);
            statuses.push(status.to_string());
            true
        })
        .unwrap();
    assert_eq!(received(&r, 1), std::fs::read(&path).unwrap(), "every byte, in order, once");
    assert_eq!(up.channel.id, MM);
    assert!(up.sent.resumed >= 2 && up.sent.restarted == 0 && !up.sent.from_earlier, "{:?}", up.sent);
    let v = up.to_json();
    assert_eq!(v["url"], "https://www.youtube.com/watch?v=vid00000001", "{v}");
    assert_eq!((v["privacy"].as_str(), v["chapters"].as_u64(), v["bytes"].as_u64()), (Some("private"), Some(3), Some(n as u64)), "{v}");
    assert_eq!(v["thumbnail"]["set"], true, "{v}");
    assert!(statuses.iter().any(|s| s.starts_with("Retrying")), "{statuses:?}");
    // the upload went to the channel asked for, with its own token, and the metadata
    let insert = r.mock.seen("POST", "/upload/youtube/v3/videos")[0].clone();
    assert!(insert.auth.starts_with("Bearer at-mohesh-"), "{}", insert.auth);
    assert!(insert.path.contains("uploadType=resumable") && insert.path.contains("notifySubscribers=false"), "{}", insert.path);
    assert_eq!(insert.header("x-upload-content-length"), Some(n.to_string().as_str()));
    assert_eq!(insert.header("x-upload-content-type"), Some("video/mp4"));
    let meta = insert.json();
    assert_eq!(meta["snippet"]["title"], "Every Robot");
    assert_eq!(meta["snippet"]["description"], "Robots, 2020-2026\n\n0:00 Intro\n0:12 #1 Ameca\n0:40 #2 Optimus");
    assert_eq!(meta["snippet"]["tags"], json!(["robots", "expo 2020"]));
    assert_eq!((meta["status"]["privacyStatus"].as_str(), meta["status"]["selfDeclaredMadeForKids"].as_bool()), (Some("private"), Some(false)));
    assert_eq!(r.mock.with(|s| s.thumbnails.clone()), [("vid00000001".to_string(), 7, "image/jpeg".to_string())]);
    // every chunk but the last is a multiple of 256 KiB
    for put in r.mock.seen("PUT", "/upload/session/") {
        let len = put.body.len() as u64;
        let range = put.header("content-range").unwrap().to_string();
        let last = range.contains(&format!("-{}/", n - 1));
        assert!(range.starts_with("bytes */") || last || len.is_multiple_of(GRANULE), "{range} {len}");
    }
    // the quota: one upload, a thumbnail, the channel lookups
    let q = r.yt.quota();
    assert_eq!((q.uploads, q.units), (1, 2 + 50), "{q:?}");
    // nothing kept once it is done
    assert!(std::fs::read_dir(r.yt.dir().join("sessions")).map(|d| d.count()).unwrap_or(0) == 0);
    let _ = std::fs::remove_dir_all(&r.root);
}

#[test]
fn an_interrupted_upload_resumes_on_the_next_run_without_a_new_session() {
    let r = rig("rerun");
    connect(&r, "h4hacks", ConnectOptions::default()).unwrap();
    let n = 5 * GRANULE as usize;
    let path = video(&r.root, "long.mov", n);
    let req = request(&path, "Long");
    // the user cancels after two chunks
    let e = r.yt.upload(&req, &mut |sent, _, _| sent < 2 * GRANULE).unwrap_err();
    assert_eq!(e, YtError::Cancelled);
    assert_eq!(received(&r, 1).len() as u64, 2 * GRANULE);
    assert_eq!(std::fs::read_dir(r.yt.dir().join("sessions")).unwrap().count(), 1, "the session is kept");
    // the next run asks where it got to and sends the rest
    let up = r.yt.upload(&req, &mut |_, _, _| true).unwrap();
    assert!(up.sent.from_earlier, "{:?}", up.sent);
    assert_eq!(received(&r, 1), std::fs::read(&path).unwrap());
    assert_eq!(r.mock.seen("POST", "/upload/youtube/v3/videos").len(), 1, "no second upload spent");
    assert_eq!(r.yt.quota().uploads, 1);
    let first_put = r.mock.seen("PUT", "/upload/session/1").into_iter().find(|p| p.header("content-range").is_some_and(|c| c.starts_with("bytes */")));
    assert!(first_put.is_some(), "a status check first");
    // a changed title is another upload
    let other = request(&path, "Long, take two");
    r.yt.upload(&other, &mut |_, _, _| true).unwrap();
    assert_eq!(r.mock.seen("POST", "/upload/youtube/v3/videos").len(), 2);
    let _ = std::fs::remove_dir_all(&r.root);
}

#[test]
fn an_expired_session_starts_again_once() {
    let r = rig("expire");
    connect(&r, "h4hacks", ConnectOptions::default()).unwrap();
    let path = video(&r.root, "clip.mp4", 3 * GRANULE as usize + 7);
    r.mock.with(|s| {
        s.chunk_faults.insert(1, Fault::Expire);
    });
    let up = r.yt.upload(&request(&path, "Clip"), &mut |_, _, _| true).unwrap();
    assert_eq!(up.sent.restarted, 1);
    assert_eq!(received(&r, 2), std::fs::read(&path).unwrap(), "the new session has the whole file");
    assert_eq!(up.video.id, "vid00000002");
    // twice is an error, and nothing is kept for a re-run
    r.mock.with(|s| {
        s.chunk_faults.insert(s.puts + 1, Fault::Expire);
        s.chunk_faults.insert(s.puts + 2, Fault::Expire);
    });
    let e = r.yt.upload(&request(&path, "Clip again"), &mut |_, _, _| true).unwrap_err();
    assert!(e.to_string().contains("dropped the upload session again"), "{e}");
    assert_eq!(std::fs::read_dir(r.yt.dir().join("sessions")).map(|d| d.count()).unwrap_or(0), 0);
    let _ = std::fs::remove_dir_all(&r.root);
}

#[test]
fn quota_and_errors_are_said_plainly() {
    let r = rig("errors");
    connect(&r, "h4hacks", ConnectOptions::default()).unwrap();
    let path = video(&r.root, "e.mp4", 1000);
    let up = |r: &Rig, answer: (u16, String)| {
        r.mock.with(|s| s.insert_answers.push_back(answer));
        r.yt.upload(&request(&path, "E"), &mut |_, _, _| true).unwrap_err()
    };
    let e = up(&r, (403, error_body(403, "quotaExceeded", "The request cannot be completed because you have exceeded your quota.")));
    assert!(matches!(e, YtError::Quota(_)) && e.to_string().contains("midnight Pacific time"), "{e}");
    assert_eq!(r.yt.quota().exceeded.unwrap().0, "quotaExceeded");
    let e = up(&r, (400, error_body(400, "uploadLimitExceeded", "The user has exceeded the number of videos they may upload.")));
    assert!(matches!(e, YtError::Quota(_)) && e.to_string().contains("H4hacks"), "{e}");
    let disabled = json!({"error": {"code": 403, "message": "YouTube Data API v3 has not been used in project 1 before or it is disabled.",
        "details": [{"reason": "SERVICE_DISABLED"}]}})
    .to_string();
    let e = up(&r, (403, disabled));
    assert!(matches!(e, YtError::Setup(_)) && e.to_string().contains("Enable"), "{e}");
    let e = up(&r, (400, error_body(400, "invalidTitle", "The request metadata specifies an invalid or empty video title.")));
    assert!(matches!(e, YtError::Invalid(_)) && e.to_string().contains("title"), "{e}");
    // a refused token twice: sign in again, naming the channel
    r.mock.with(|s| {
        s.insert_answers.push_back((401, error_body(401, "authError", "Invalid Credentials")));
        s.insert_answers.push_back((401, error_body(401, "authError", "Invalid Credentials")));
    });
    let e = r.yt.upload(&request(&path, "E"), &mut |_, _, _| true).unwrap_err();
    assert!(matches!(e, YtError::Auth(_)) && e.to_string().contains("youtube auth") && e.to_string().contains("H4hacks"), "{e}");
    // a lapsed refresh token (Testing mode: 7 days), found when YouTube refuses the access token
    r.mock.with(|s| {
        s.refresh_refused = true;
        s.insert_answers.push_back((401, error_body(401, "authError", "Invalid Credentials")));
    });
    let e = r.yt.upload(&request(&path, "E"), &mut |_, _, _| true).unwrap_err();
    assert!(matches!(e, YtError::Auth(_)) && e.to_string().contains("sign in again") && e.to_string().contains("pick H4hacks"), "{e}");
    let _ = std::fs::remove_dir_all(&r.root);
}

#[test]
fn a_playlist_needs_its_permission_and_is_found_by_title() {
    let r = rig("playlist");
    connect(&r, "h4hacks", ConnectOptions::default()).unwrap();
    let path = video(&r.root, "p.mp4", 2000);
    let mut req = request(&path, "P");
    req.playlist = Some("Robot showcases".into());
    let e = r.yt.upload(&req, &mut |_, _, _| true).unwrap_err();
    assert!(matches!(e, YtError::Consent(_)) && e.to_string().contains("--playlists"), "{e}");
    assert!(r.mock.seen("POST", "/upload/youtube/v3/videos").is_empty(), "refused before any byte");
    connect(&r, "h4hacks", ConnectOptions { make_default: false, playlists: true }).unwrap();
    r.mock.with(|s| s.playlists = vec![("PLa".into(), "Shorts".into()), ("PLb".into(), "Talks".into()), ("PLc".into(), "Robot Showcases".into())]);
    // an unknown playlist: refused before the upload
    req.playlist = Some("Holidays".into());
    let e = r.yt.upload(&req, &mut |_, _, _| true).unwrap_err().to_string();
    assert!(e.contains("no playlist `Holidays`") && e.contains("Talks"), "{e}");
    assert!(r.mock.seen("POST", "/upload/youtube/v3/videos").is_empty());
    req.playlist = Some("robot showcases".into());
    let up = r.yt.upload(&req, &mut |_, _, _| true).unwrap();
    assert_eq!(up.to_json()["playlist"], json!({"id": "PLc", "title": "Robot Showcases", "added": true, "error": null}));
    assert_eq!(r.mock.with(|s| s.playlist_items.clone()), [("PLc".to_string(), up.video.id.clone())]);
    assert!(r.mock.seen("GET", "/youtube/v3/playlists").iter().any(|g| g.path.contains("pageToken=PAGE2")), "the second page was read");
    // a channel without custom thumbnails: the video is up, the thumbnail says why not
    r.mock.with(|s| {
        s.thumbnail_answer = Some((
            403,
            error_body(403, "forbidden", "The authenticated user doesnt have permissions for uploading and setting custom thumbnail images for videos."),
        ))
    });
    let mut req = request(&path, "T");
    req.thumbnail = Some(r.root.join("t.png").to_string_lossy().into_owned());
    std::fs::write(r.root.join("t.png"), [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A, 0]).unwrap();
    let up = r.yt.upload(&req, &mut |_, _, _| true).unwrap();
    let t = up.to_json()["thumbnail"].clone();
    assert_eq!(t["set"], false, "{t}");
    assert!(t["error"].as_str().unwrap().contains("youtube.com/verify"), "{t}");
    let _ = std::fs::remove_dir_all(&r.root);
}

#[test]
fn hostile_answers_are_errors_never_panics() {
    let r = rig("hostile");
    connect(&r, "h4hacks", ConnectOptions::default()).unwrap();
    let path = video(&r.root, "h.mp4", 2 * GRANULE as usize + 3);
    let run = |r: &Rig, f: &dyn Fn(&mut mock::State)| {
        r.mock.with(f);
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            r.yt.upload(&request(&path, &format!("H{}", r.mock.with(|s| s.puts))), &mut |_, _, _| true)
        }));
        r.mock.with(|s| {
            s.location = None;
            s.range = None;
            s.video_answer = None;
            s.status_answers.clear();
        });
        out.expect("no panic")
    };
    // the session URI elsewhere: the file (and the token) never go there
    let e = run(&r, &|s| s.location = Some("http://evil.example/upload".into())).unwrap_err();
    assert!(e.to_string().contains("not YouTube's"), "{e}");
    let e = run(&r, &|s| s.location = Some(String::new())).unwrap_err();
    assert!(e.to_string().contains("named no address"), "{e}");
    // more received than sent
    let e = run(&r, &|s| s.range = Some("bytes=0-99999999".into())).unwrap_err();
    assert!(e.to_string().contains("bytes received"), "{e}");
    let e = run(&r, &|s| s.range = Some("bytes=7-9".into())).unwrap_err();
    assert!(matches!(e, YtError::Upload(_)), "{e}");
    let e = run(&r, &|s| s.range = Some("bytes=0-x".into())).unwrap_err();
    assert!(e.to_string().contains("unreadable upload range"), "{e}");
    // a finished upload without a usable video
    for answer in ["", "not json", r#"{"id":"../../x"}"#, r#"{"id":"x","status":{"uploadStatus":"rejected","rejectionReason":"duplicate"}}"#] {
        let e = run(&r, &|s| s.video_answer = Some((201, answer.into()))).unwrap_err();
        assert!(matches!(e, YtError::Upload(_)), "{answer}: {e}");
    }
    let e = run(&r, &|s| s.video_answer = Some((201, "x".repeat(3_000_000)))).unwrap_err();
    assert!(matches!(e, YtError::Upload(_)), "{e}");
    // an odd channel answer at sign-in
    for answer in [(200, r#"{"items":[{"id":"bad id!"}]}"#), (200, "<html>"), (500, "oops"), (403, "{}")] {
        r.mock.with(|s| s.channels_answer = Some((answer.0, answer.1.into())));
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| connect(&r, "mohesh", ConnectOptions::default())));
        assert!(out.expect("no panic").is_err(), "{answer:?}");
    }
    let _ = std::fs::remove_dir_all(&r.root);
}

#[test]
fn files_are_checked_before_anything_is_sent() {
    let r = rig("files");
    connect(&r, "h4hacks", ConnectOptions::default()).unwrap();
    let audio = video(&r.root, "mix.wav", 10);
    let e = r.yt.upload(&request(&audio, "A"), &mut |_, _, _| true).unwrap_err();
    assert!(e.to_string().contains("does not take a .wav"), "{e}");
    let ok = video(&r.root, "ok.mp4", 10);
    let mut req = request(&ok, "A");
    req.thumbnail = Some(r.root.join("nope.jpg").to_string_lossy().into_owned());
    assert!(r.yt.upload(&req, &mut |_, _, _| true).unwrap_err().to_string().contains("thumbnail"));
    req.thumbnail = None;
    req.channel = Some("S4Saranya".into());
    assert!(matches!(r.yt.upload(&req, &mut |_, _, _| true), Err(YtError::Channel(_))));
    assert!(r.mock.seen("POST", "/upload/youtube/v3/videos").is_empty());
    // a broken chapters file or thumbnail beside the video is a note, not a failure
    std::fs::write(r.root.join("ok.chapters.txt"), "not chapters").unwrap();
    std::fs::write(r.root.join("ok.thumbnail.jpg"), "GIF89a").unwrap();
    req.channel = None;
    req.meta.privacy = Privacy::Unlisted;
    let up = r.yt.upload(&req, &mut |_, _, _| true).unwrap();
    let v = up.to_json();
    let notes: Vec<String> = v["notes"].as_array().unwrap().iter().map(|n| n.as_str().unwrap().to_string()).collect();
    assert!(notes.iter().any(|n| n.contains("chapters not added")), "{notes:?}");
    assert!(notes.iter().any(|n| n.contains("thumbnail beside the video was not used")), "{notes:?}");
    // the mock keeps uploads private, as an unaudited project's may be
    assert!(notes.iter().any(|n| n.contains("kept the video private, not unlisted")), "{notes:?}");
    assert_eq!(v["thumbnail"], Value::Null);
    let _ = std::fs::remove_dir_all(&r.root);
}

#[test]
fn the_engine_runs_it_as_commands() {
    use filmcraft_engine::Session;
    let r = rig("engine");
    connect(&r, "h4hacks", ConnectOptions::default()).unwrap();
    let path = video(&r.root, "e.mp4", 3000);
    let Rig { mock, yt, root, .. } = r;
    let mut s = Session::default();
    s.youtube.publisher = Some(Arc::new(filmcraft_youtube::EnginePublisher::new(yt)));
    let l = s.execute("youtube.channels", json!({})).unwrap();
    assert_eq!(l["default"]["title"], "H4hacks", "{l}");
    let v = s.execute("youtube.upload", json!({"path": path, "title": "From the engine", "tags": "a, b", "wait": true})).unwrap();
    assert_eq!(v["youtube"]["videoId"], "vid00000001", "{v}");
    assert_eq!(v["checked"]["channel"]["id"], H4, "{v}");
    let e = s.execute("youtube.upload", json!({"path": path, "title": "x", "channel": "nobody"})).unwrap_err().to_string();
    assert!(e.contains("no connected YouTube channel"), "{e}");
    assert_eq!(mock.seen("POST", "/upload/youtube/v3/videos").len(), 1);
    // connect as a job, then disconnect
    mock.with(|st| st.picked = "mohesh".into());
    let c = s.execute("youtube.connect", json!({"wait": true})).unwrap();
    assert_eq!(c["youtube"]["channel"]["id"], MM, "{c}");
    let d = s.execute("youtube.disconnect", json!({"channel": "@h4hacks"})).unwrap();
    assert_eq!((d["revoked"].as_bool(), d["default"]["id"].as_str()), (Some(true), Some(MM)), "{d}");
    let _ = std::fs::remove_dir_all(&root);
}
