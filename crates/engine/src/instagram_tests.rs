//! Tests of [`crate::instagram`]: the commands run the host's publisher (sign-in and publish as
//! jobs with progress, cancel and their answer in `jobs.list`), are disabled with the
//! publisher's reason, and refuse hostile parameters before any job starts; the caption and URL
//! checks follow Instagram's limits.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::instagram::{self as ig, InstagramPublisher, ReelRequest};
use serde_json::{Value, json};

fn tmp_file(name: &str, bytes: usize) -> String {
    let d = std::env::temp_dir().join(format!("filmcraft-instagram-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    std::fs::write(&p, vec![7u8; bytes]).unwrap();
    p.to_string_lossy().into_owned()
}

#[derive(Default)]
struct Fake {
    not_ready: Option<String>,
    no_app: Option<String>,
    hang: bool,
    published: Mutex<Vec<ReelRequest>>,
}

impl InstagramPublisher for Fake {
    fn connect_ready(&self) -> std::result::Result<(), String> {
        self.no_app.clone().map_or(Ok(()), Err)
    }
    fn ready(&self) -> std::result::Result<(), String> {
        self.not_ready.clone().map_or(Ok(()), Err)
    }
    fn account(&self) -> std::result::Result<Value, String> {
        Ok(json!({"account": {"username": "moheshmohan"}}))
    }
    fn connect(&self, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        *progress.status.lock().unwrap() = "Log in to Instagram".into();
        Ok(json!({"account": {"username": "moheshmohan"}}))
    }
    fn disconnect(&self) -> std::result::Result<Value, String> {
        Ok(json!({"disconnected": true}))
    }
    fn limits(&self) -> std::result::Result<Value, String> {
        Ok(json!({"posts": 1, "postQuota": 100}))
    }
    fn check(&self, _req: &ReelRequest) -> std::result::Result<Value, String> {
        Ok(json!({"account": {"username": "moheshmohan"}}))
    }
    fn publish(&self, req: &ReelRequest, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        self.published.lock().unwrap().push(req.clone());
        let total = progress.total.load(Ordering::Relaxed);
        if self.hang {
            for _ in 0..1000 {
                if progress.cancel.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                progress.done.store(total / 2, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(5));
            }
            return Err("never cancelled".into());
        }
        progress.done.store(total, Ordering::Relaxed);
        Ok(json!({"mediaId": "18000000000001", "permalink": "https://www.instagram.com/reel/Cx/"}))
    }
}

fn session(fake: Fake) -> (Session, Arc<Fake>) {
    let mut s = Session::default();
    let fake = Arc::new(fake);
    s.instagram.publisher = Some(fake.clone());
    (s, fake)
}

fn wait_finished(s: &mut Session, job: u64) -> Value {
    for _ in 0..2000 {
        let jobs = s.execute("jobs.list", json!({})).unwrap();
        let j = jobs.as_array().unwrap().iter().find(|j| j["id"] == job).cloned().unwrap();
        if j["finished"] == true {
            return j;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("job {job} never finished");
}

#[test]
fn disabled_without_a_publisher_or_an_account() {
    let path = tmp_file("disabled.mp4", 10);
    let mut s = Session::default();
    for (id, p) in [
        ("instagram.connect", json!({})),
        ("instagram.account", json!({})),
        ("instagram.limits", json!({})),
        ("instagram.disconnect", json!({})),
        ("instagram.publish", json!({"path": path})),
    ] {
        let e = s.execute(id, p).unwrap_err().to_string();
        assert!(e.contains("Instagram is not available"), "{id}: {e}");
    }
    let (mut s, _) = session(Fake {
        not_ready: Some("Instagram is not connected: run `filmcraft-cli instagram auth`".into()),
        no_app: Some("not set up".into()),
        ..Default::default()
    });
    let e = s.execute("instagram.publish", json!({"path": path})).unwrap_err().to_string();
    assert!(e.contains("instagram auth"), "{e}");
    assert!(s.execute("instagram.connect", json!({})).unwrap_err().to_string().contains("not set up"));
    assert!(s.execute("instagram.account", json!({})).is_ok(), "the account listing works without one");
}

#[test]
fn publish_runs_as_a_job_with_its_answer() {
    let path = tmp_file("reel.mp4", 4096);
    let (mut s, fake) = session(Fake::default());
    let v = s.execute("instagram.publish", json!({"path": path, "caption": "Robots #ai", "coverSeconds": 2.5, "shareToFeed": false})).unwrap();
    let job = v["job"].as_u64().unwrap();
    let j = wait_finished(&mut s, job);
    assert_eq!(j["instagram"]["mediaId"], "18000000000001", "{j}");
    assert_eq!(j["done"], 4096);
    let req = fake.published.lock().unwrap()[0].clone();
    assert_eq!((req.caption.as_deref(), req.cover_ms, req.share_to_feed), (Some("Robots #ai"), Some(2500), false));
    let v = s.execute("instagram.publish", json!({"videoUrl": "https://cdn.example.com/a.mp4", "wait": true})).unwrap();
    assert_eq!(v["instagram"]["permalink"], "https://www.instagram.com/reel/Cx/");
    let v = s.execute("instagram.connect", json!({"wait": true})).unwrap();
    assert_eq!(v["instagram"]["account"]["username"], "moheshmohan");
}

#[test]
fn a_publish_is_cancelled_like_an_export() {
    let path = tmp_file("hang.mp4", 100);
    let (mut s, _) = session(Fake { hang: true, ..Default::default() });
    let job = s.execute("instagram.publish", json!({"path": path})).unwrap()["job"].as_u64().unwrap();
    std::thread::sleep(Duration::from_millis(30));
    s.execute("jobs.cancel", json!({"job": job})).unwrap();
    let j = wait_finished(&mut s, job);
    assert!(j["result"]["error"].as_str().is_some(), "{j}");
    assert_eq!(j["instagram"], Value::Null);
}

#[test]
fn hostile_parameters_are_refused_before_any_job() {
    let path = tmp_file("p.mp4", 10);
    let empty = tmp_file("empty.mp4", 0);
    let (mut s, fake) = session(Fake::default());
    for (p, want) in [
        (json!({}), "need `path`"),
        (json!({"path": 5}), "must be a string"),
        (json!({"path": "/no/such/file.mp4"}), "/no/such/file.mp4"),
        (json!({"path": empty}), "is empty"),
        (json!({"path": "a\u{0}b"}), "usable path"),
        (json!({"path": path, "caption": 7}), "must be a string"),
        (json!({"path": path, "caption": "x".repeat(2201)}), "2201 characters"),
        (json!({"path": path, "caption": "#a ".repeat(31)}), "31 hashtags"),
        (json!({"path": path, "caption": "@a ".repeat(21)}), "21 accounts"),
        (json!({"path": path, "caption": "bell\u{7}"}), "control character"),
        (json!({"path": path, "coverSeconds": -1}), "coverSeconds"),
        (json!({"path": path, "coverSeconds": "x"}), "coverSeconds"),
        (json!({"path": path, "coverSeconds": 1e300}), "coverSeconds"),
        (json!({"path": path, "coverUrl": "file:///etc/passwd"}), "https://"),
        (json!({"videoUrl": "https://127.0.0.1/a.mp4"}), "public server"),
        (json!({"videoUrl": "https://192.168.1.5/a.mp4"}), "public server"),
        (json!({"videoUrl": "https://user@localhost/a.mp4"}), "public server"),
        (json!({"videoUrl": "https://example.com/a b.mp4"}), "https://"),
        (json!({"path": path, "shareToFeed": "yes"}), "true or false"),
        (json!([1, 2]), "object"),
    ] {
        let e = s.execute("instagram.publish", p.clone()).unwrap_err().to_string();
        assert!(e.contains(want), "{p}: {e}");
    }
    assert!(fake.published.lock().unwrap().is_empty(), "no job started");
}

#[test]
fn captions_count_like_instagram() {
    assert_eq!(ig::caption_counts("#a #b c#d @x email@example.com @_y ## #"), (2, 2));
    assert_eq!(ig::caption_counts("#über #日本 @ünï"), (2, 1));
    assert_eq!(ig::check_caption(Some("  ")).unwrap(), None);
    assert_eq!(ig::check_caption(Some("a\r\nb")).unwrap().as_deref(), Some("a\nb"));
    let ok = format!("{}{}", "#t ".repeat(30), "é".repeat(2200 - 90));
    assert!(ig::check_caption(Some(&ok)).is_ok());
    assert!(ig::check_public_url("u", "https://cdn.example.com/a.mp4?sig=1").is_ok());
    for bad in
        ["https://localhost:8000/a", "https://foo.local/a", "https://10.0.0.1/a", "https://172.20.1.1/a", "https://[::1]/a", "https://intranet/a", "https://"]
    {
        assert!(ig::check_public_url("u", bad).is_err(), "{bad}");
    }
    assert!(ig::check_public_url("u", "https://172.32.0.1/a").is_ok(), "outside the private block");
}
