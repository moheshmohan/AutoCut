//! Tests of [`crate::youtube`]: the commands run the host's publisher (sign-in and upload as
//! jobs with progress, cancel and their answer in `jobs.list`), are disabled with the
//! publisher's reason, and refuse hostile parameters before any job starts; the metadata checks
//! follow YouTube's limits.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::youtube::{self as yt, ConnectOptions, UploadRequest, YouTubePublisher};
use serde_json::{Value, json};

fn tmp_file(name: &str, bytes: usize) -> String {
    let d = std::env::temp_dir().join(format!("filmcraft-youtube-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    std::fs::write(&p, vec![7u8; bytes]).unwrap();
    p.to_string_lossy().into_owned()
}

/// A publisher that reports progress in four steps, or waits to be cancelled.
#[derive(Default)]
struct Fake {
    not_ready: Option<String>,
    no_client: Option<String>,
    refuse: Option<String>,
    hang: bool,
    uploads: Mutex<Vec<UploadRequest>>,
    calls: Mutex<Vec<String>>,
}

impl YouTubePublisher for Fake {
    fn connect_ready(&self) -> std::result::Result<(), String> {
        self.no_client.clone().map_or(Ok(()), Err)
    }
    fn ready(&self) -> std::result::Result<(), String> {
        self.not_ready.clone().map_or(Ok(()), Err)
    }
    fn channels(&self) -> std::result::Result<Value, String> {
        Ok(json!({"channels": [{"id": "UCaaaaaaaaaaaaaaaaaaaaaa", "title": "H4hacks", "handle": "@h4hacks", "default": true}]}))
    }
    fn connect(&self, opts: ConnectOptions, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        self.calls.lock().unwrap().push(format!("connect default={} playlists={}", opts.make_default, opts.playlists));
        *progress.status.lock().unwrap() = "Pick the channel in Google's account chooser".into();
        Ok(json!({"channel": {"id": "UCaaaaaaaaaaaaaaaaaaaaaa", "title": "H4hacks"}, "default": opts.make_default}))
    }
    fn set_default(&self, channel: &str) -> std::result::Result<Value, String> {
        self.calls.lock().unwrap().push(format!("default {channel}"));
        if channel == "nobody" {
            return Err("no connected channel `nobody`".into());
        }
        Ok(json!({"default": channel}))
    }
    fn disconnect(&self, channel: &str) -> std::result::Result<Value, String> {
        self.calls.lock().unwrap().push(format!("disconnect {channel}"));
        Ok(json!({"disconnected": channel}))
    }
    fn check(&self, req: &UploadRequest) -> std::result::Result<Value, String> {
        if let Some(r) = &self.refuse {
            return Err(r.clone());
        }
        Ok(json!({"channel": {"title": req.channel.clone().unwrap_or_else(|| "H4hacks".into())}}))
    }
    fn upload(&self, req: &UploadRequest, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        self.uploads.lock().unwrap().push(req.clone());
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
        for i in 1..=4 {
            progress.done.store(total * i / 4, Ordering::Relaxed);
        }
        Ok(json!({"videoId": "dQw4w9WgXcQ", "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}))
    }
}

fn session(fake: Fake) -> (Session, Arc<Fake>) {
    let mut s = Session::default();
    let fake = Arc::new(fake);
    s.youtube.publisher = Some(fake.clone());
    (s, fake)
}

fn job_entry(s: &mut Session, job: u64) -> Value {
    let jobs = s.execute("jobs.list", json!({})).unwrap();
    jobs.as_array().unwrap().iter().find(|j| j["id"] == job).cloned().unwrap()
}

fn wait_finished(s: &mut Session, job: u64) -> Value {
    for _ in 0..2000 {
        let j = job_entry(s, job);
        if j["finished"] == true {
            return j;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    panic!("job {job} never finished");
}

#[test]
fn disabled_without_a_publisher_or_a_channel() {
    let path = tmp_file("disabled.mp4", 10);
    let mut s = Session::default();
    for (id, p) in [
        ("youtube.connect", json!({})),
        ("youtube.channels", json!({})),
        ("youtube.setDefault", json!({"channel": "x"})),
        ("youtube.disconnect", json!({"channel": "x"})),
        ("youtube.upload", json!({"path": path, "title": "T"})),
    ] {
        let e = s.execute(id, p).unwrap_err().to_string();
        assert!(e.contains("YouTube is not available"), "{id}: {e}");
    }
    let why = "YouTube is not connected: run `filmcraft-cli youtube auth`";
    let (mut s, fake) = session(Fake { not_ready: Some(why.into()), no_client: Some("no client_secret.json".into()), ..Default::default() });
    let e = s.execute("youtube.upload", json!({"path": path, "title": "T"})).unwrap_err().to_string();
    assert!(e.contains("youtube auth"), "{e}");
    let e = s.execute("youtube.connect", json!({})).unwrap_err().to_string();
    assert!(e.contains("client_secret.json"), "{e}");
    assert!(fake.uploads.lock().unwrap().is_empty() && s.jobs.is_empty());
    // the channel list still answers (it says what is missing)
    assert!(s.execute("youtube.channels", json!({})).is_ok());
    let listed = s.execute("command.list", json!({})).unwrap();
    let cmd = listed.as_array().unwrap().iter().find(|c| c["id"] == "youtube.upload").cloned().unwrap();
    assert_eq!(cmd["enabled"], false, "{cmd}");
}

#[test]
fn uploads_as_a_job_and_reports_the_video() {
    let path = tmp_file("export.mp4", 4096);
    let (mut s, fake) = session(Fake::default());
    let r = s
        .execute(
            "youtube.upload",
            json!({"path": path, "title": "  Every Robot  ", "description": "Robots\r\n2020", "tags": ["robots", "  expo 2020 ", "Robots"],
                   "category": "science & technology", "channel": "@H4hacks", "wait": true}),
        )
        .unwrap();
    assert_eq!(r["youtube"]["videoId"], "dQw4w9WgXcQ", "{r}");
    assert_eq!(r["checked"]["channel"]["title"], "@H4hacks", "{r}");
    let req = fake.uploads.lock().unwrap()[0].clone();
    assert_eq!(req.meta.title, "Every Robot");
    assert_eq!(req.meta.description.as_deref(), Some("Robots\n2020"));
    assert_eq!(req.meta.tags, ["robots", "expo 2020"], "repeats dropped, spaces trimmed");
    assert_eq!((req.meta.category.as_str(), req.meta.privacy, req.meta.made_for_kids), ("28", yt::Privacy::Private, false));
    assert!(req.sidecars && req.meta.notify_subscribers && req.channel.as_deref() == Some("@H4hacks"));
    let job = r["job"].as_u64().unwrap();
    let j = job_entry(&mut s, job);
    assert_eq!((j["done"].as_u64(), j["total"].as_u64(), &j["status"]), (Some(4096), Some(4096), &json!("Uploaded to YouTube")), "{j}");
    assert_eq!(j["youtube"]["url"], "https://www.youtube.com/watch?v=dQw4w9WgXcQ");
    assert!(j["label"].as_str().unwrap().contains("export.mp4") && j["label"].as_str().unwrap().contains("@H4hacks"), "{j}");
    assert!(crate::youtube::is_job(&s, job));

    // in the background: listed at once, finished with the video
    let r = s.execute("youtube.upload", json!({"path": path, "title": "T", "tags": "a, b ,c", "privacy": "Unlisted", "sidecars": false})).unwrap();
    let job = r["job"].as_u64().unwrap();
    assert!(r.get("youtube").is_none(), "only a blocking call carries the video: {r}");
    let j = wait_finished(&mut s, job);
    assert_eq!(j["youtube"]["videoId"], "dQw4w9WgXcQ", "{j}");
    let req = fake.uploads.lock().unwrap()[1].clone();
    assert_eq!((req.meta.tags.len(), req.meta.privacy, req.sidecars), (3, yt::Privacy::Unlisted, false));
}

#[test]
fn cancel_stops_the_upload() {
    let path = tmp_file("long.mov", 1000);
    let (mut s, _) = session(Fake { hang: true, ..Default::default() });
    let job = s.execute("youtube.upload", json!({"path": path, "title": "Long"})).unwrap()["job"].as_u64().unwrap();
    s.execute("jobs.cancel", json!({"job": job})).unwrap();
    let j = wait_finished(&mut s, job);
    assert_eq!(j["result"]["error"], "cancelled", "{j}");
    assert_eq!(j["youtube"], Value::Null, "{j}");
}

#[test]
fn connect_runs_as_a_job_and_channel_commands_answer() {
    let (mut s, fake) = session(Fake::default());
    let r = s.execute("youtube.connect", json!({"default": true, "wait": true})).unwrap();
    assert_eq!(r["youtube"]["channel"]["title"], "H4hacks", "{r}");
    let j = job_entry(&mut s, r["job"].as_u64().unwrap());
    assert_eq!(j["status"], "Connected H4hacks", "{j}");
    let r = s.execute("youtube.connect", json!({"playlists": true})).unwrap();
    wait_finished(&mut s, r["job"].as_u64().unwrap());
    assert_eq!(fake.calls.lock().unwrap()[..2], ["connect default=true playlists=false", "connect default=false playlists=true"]);
    assert!(s.execute("youtube.connect", json!({"default": "yes"})).unwrap_err().to_string().contains("true or false"));

    assert_eq!(s.execute("youtube.setDefault", json!({"channel": " @H4hacks "})).unwrap()["default"], "@H4hacks");
    assert!(s.execute("youtube.setDefault", json!({"channel": "nobody"})).unwrap_err().to_string().contains("no connected channel"));
    assert_eq!(s.execute("youtube.disconnect", json!({"channel": "UCaaaaaaaaaaaaaaaaaaaaaa"})).unwrap()["disconnected"], "UCaaaaaaaaaaaaaaaaaaaaaa");
    for (id, p, want) in [
        ("youtube.setDefault", json!({}), "need `channel`"),
        ("youtube.setDefault", json!({"channel": "  "}), "need `channel`"),
        ("youtube.disconnect", json!({"channel": 5}), "must be a string"),
        ("youtube.disconnect", json!({"channel": "x".repeat(500)}), "is not a channel"),
        ("youtube.disconnect", json!({"channel": "a\u{0}b"}), "is not a channel"),
    ] {
        let e = s.execute(id, p.clone()).unwrap_err().to_string();
        assert!(e.contains(want), "{id} {p}: {e}");
    }
    assert_eq!(s.execute("youtube.channels", json!({})).unwrap()["channels"][0]["handle"], "@h4hacks");
}

#[test]
fn hostile_parameters_start_no_job() {
    let (mut s, fake) = session(Fake::default());
    let empty = tmp_file("empty.mp4", 0);
    let ok = tmp_file("ok.mp4", 10);
    let dir = std::env::temp_dir().to_string_lossy().into_owned();
    let long_tags: Vec<String> = (0..60).map(|i| format!("tag number {i}")).collect();
    for (p, want) in [
        (json!({"title": "T"}), "need `path`"),
        (json!({"path": "", "title": "T"}), "need `path`"),
        (json!({"path": 42, "title": "T"}), "must be a string"),
        (json!({"path": "/no/such/file.mp4", "title": "T"}), "/no/such/file.mp4"),
        (json!({"path": dir, "title": "T"}), "not a file"),
        (json!({"path": empty, "title": "T"}), "is empty"),
        (json!({"path": ok}), "need `title`"),
        (json!({"path": ok, "title": "   "}), "title is empty"),
        (json!({"path": ok, "title": 3}), "must be a string"),
        (json!({"path": ok, "title": "x".repeat(101)}), "at most 100"),
        (json!({"path": ok, "title": "a <b>"}), "< or >"),
        (json!({"path": ok, "title": "two\nlines"}), "control character"),
        (json!({"path": ok, "title": "T", "description": "é".repeat(2501)}), "at most 5000"),
        (json!({"path": ok, "title": "T", "description": "1 > 0"}), "< or >"),
        (json!({"path": ok, "title": "T", "description": ["a"]}), "must be a string"),
        (json!({"path": ok, "title": "T", "tags": long_tags}), "at most 500"),
        (json!({"path": ok, "title": "T", "tags": [1, 2]}), "every tag must be a string"),
        (json!({"path": ok, "title": "T", "tags": ["a<b"]}), "contains <"),
        (json!({"path": ok, "title": "T", "tags": {"a": 1}}), "list of strings"),
        (json!({"path": ok, "title": "T", "tags": vec!["a"; 1000]}), "at most 500"),
        (json!({"path": ok, "title": "T", "privacy": "friends"}), "not one of private, unlisted, public"),
        (json!({"path": ok, "title": "T", "privacy": true}), "must be a string"),
        (json!({"path": ok, "title": "T", "category": "Cooking"}), "not a YouTube category"),
        (json!({"path": ok, "title": "T", "category": "123456"}), "not a YouTube category"),
        (json!({"path": ok, "title": "T", "madeForKids": "no"}), "true or false"),
        (json!({"path": ok, "title": "T", "publishAt": "tomorrow"}), "not a date and time"),
        (json!({"path": ok, "title": "T", "publishAt": "2026-13-01T10:00:00Z"}), "not a date and time"),
        (json!({"path": ok, "title": "T", "publishAt": "2026-11-01T10:00:00Z", "privacy": "public"}), "leave `privacy` private"),
        (json!({"path": ok, "title": "T", "channel": 7}), "must be a string"),
        (json!({"path": ok, "title": "T", "playlist": "p\u{7}"}), "is not a playlist"),
        (json!({"path": ok, "title": "T", "thumbnail": "a\u{0}b"}), "not a usable path"),
        (json!({"path": ok, "title": "T", "sidecars": 1}), "true or false"),
    ] {
        let e = s.execute("youtube.upload", p.clone()).unwrap_err().to_string();
        assert!(e.contains(want), "{p}: {e}");
    }
    assert!(s.jobs.is_empty() && fake.uploads.lock().unwrap().is_empty());
    // a blank channel is the default channel
    s.execute("youtube.upload", json!({"path": ok, "title": "T", "channel": " ", "wait": true})).unwrap();
    assert_eq!(fake.uploads.lock().unwrap()[0].channel, None);
    // the publisher's own check refuses before a job
    let (mut s, fake) = session(Fake { refuse: Some("the thumbnail is not a JPEG or PNG".into()), ..Default::default() });
    let e = s.execute("youtube.upload", json!({"path": ok, "title": "T", "thumbnail": "/x.gif"})).unwrap_err().to_string();
    assert!(e.contains("JPEG or PNG") && s.jobs.is_empty() && fake.uploads.lock().unwrap().is_empty(), "{e}");
}

#[test]
fn metadata_checks_follow_youtubes_limits() {
    assert_eq!(yt::check_title(&"é".repeat(100)).unwrap().chars().count(), 100);
    assert!(yt::check_title("").is_err() && yt::check_title("<").is_err());
    assert_eq!(yt::check_description(Some("  ")).unwrap(), None);
    assert_eq!(yt::check_description(Some(&"a".repeat(5000))).unwrap().unwrap().len(), 5000);
    assert!(yt::check_description(Some(&"a".repeat(5001))).is_err());
    assert!(yt::check_description(Some("tab\tand\nline")).is_ok() && yt::check_description(Some("bell\u{7}")).is_err());
    // commas count, and a tag with a space counts its quotes
    let t = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
    assert_eq!(yt::tags_length(&t(&["ab", "c d"])), 2 + 1 + 5);
    assert_eq!(yt::tags_length(&[]), 0);
    let exact: Vec<String> = vec!["a".repeat(250), "b".repeat(249)];
    assert_eq!(yt::check_tags(&exact).unwrap().len(), 2, "250 + 1 + 249 = 500");
    assert!(yt::check_tags(&[format!("{} x", "a".repeat(497))]).is_err(), "497 + 2 + 2 quotes = 501");
    assert_eq!(yt::check_category(None).unwrap(), "22");
    assert_eq!(yt::check_category(Some("Education")).unwrap(), "27");
    assert_eq!(yt::check_category(Some(" 28 ")).unwrap(), "28");
    assert_eq!(yt::check_category(Some("43")).unwrap(), "43", "another region's id is YouTube's to check");
    for ok in ["2026-11-01T18:00:00Z", "2026-11-01T18:00:00.5+04:00", "2026-02-28t00:00:00z"] {
        assert!(yt::check_publish_at(ok).is_ok(), "{ok}");
    }
    for bad in [
        "",
        "2026-11-01",
        "2026-11-01T18:00Z",
        "2026-11-01T24:00:00Z",
        "2026-11-01T18:00:00",
        "2026-11-01T18:00:00+4",
        "2026-11-01T18:00:00.Z",
        "２０２６-11-01T18:00:00Z",
    ] {
        assert!(yt::check_publish_at(bad).is_err(), "{bad}");
    }
    assert_eq!(yt::Privacy::parse(" PUBLIC ").unwrap(), yt::Privacy::Public);
    assert_eq!(yt::Privacy::default().as_str(), "private");
}
