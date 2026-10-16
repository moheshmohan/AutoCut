//! Tests of [`crate::photos`]: `photos.upload` runs the host's uploader as a job with byte
//! progress, reports the media item in `jobs.list`, stops on `jobs.cancel`, is disabled with the
//! uploader's reason, and refuses hostile parameters before any job starts.

use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::*;
use crate::photos::PhotosUploader;
use serde_json::{Value, json};

fn tmp_file(name: &str, bytes: usize) -> String {
    let d = std::env::temp_dir().join(format!("filmcraft-photos-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(name);
    std::fs::write(&p, vec![7u8; bytes]).unwrap();
    p.to_string_lossy().into_owned()
}

/// An uploader that reports progress in four steps, or waits to be cancelled.
#[derive(Default)]
struct Fake {
    not_ready: Option<String>,
    hang: bool,
    calls: Mutex<Vec<(String, Option<String>)>>,
}

impl PhotosUploader for Fake {
    fn ready(&self) -> std::result::Result<(), String> {
        self.not_ready.clone().map_or(Ok(()), Err)
    }
    fn accepts(&self, path: &str) -> std::result::Result<(), String> {
        if path.ends_with(".wav") { Err("Google Photos does not take a .wav file".into()) } else { Ok(()) }
    }
    fn upload(&self, path: &str, description: Option<&str>, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        self.calls.lock().unwrap().push((path.to_string(), description.map(str::to_string)));
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
        Ok(json!({"mediaItemId": "AF1Qip-new", "productUrl": "https://photos.google.com/lr/photo/AF1Qip-new", "bytes": total}))
    }
}

fn session(fake: Fake) -> (Session, Arc<Fake>) {
    let mut s = Session::default();
    let fake = Arc::new(fake);
    s.photos = Some(fake.clone());
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
fn disabled_without_an_uploader_or_when_not_ready() {
    let path = tmp_file("disabled.mp4", 10);
    let mut s = Session::default();
    let e = s.execute("photos.upload", json!({"path": path})).unwrap_err().to_string();
    assert!(e.contains("not available") && e.contains("Google Photos"), "{e}");
    let why = "Google Photos needs your consent: run `filmcraft-cli photos auth --upload`";
    let (mut s, fake) = session(Fake { not_ready: Some(why.into()), ..Default::default() });
    let e = s.execute("photos.upload", json!({"path": path})).unwrap_err().to_string();
    assert!(e.contains("photos auth --upload"), "{e}");
    assert!(fake.calls.lock().unwrap().is_empty() && s.jobs.is_empty());
    let listed = s.execute("command.list", json!({})).unwrap();
    let cmd = listed.as_array().unwrap().iter().find(|c| c["id"] == "photos.upload").cloned().unwrap();
    assert_eq!(cmd["enabled"], false, "{cmd}");
}

#[test]
fn uploads_as_a_job_and_reports_the_media_item() {
    let path = tmp_file("export.mp4", 4096);
    let (mut s, fake) = session(Fake::default());
    let r = s.execute("photos.upload", json!({"path": path, "description": "  Robots, 2020-2026 ", "wait": true})).unwrap();
    assert_eq!(r["upload"]["mediaItemId"], "AF1Qip-new", "{r}");
    assert_eq!((r["result"]["path"].as_str(), r["result"]["bytes"].as_u64()), (Some(path.as_str()), Some(4096)), "{r}");
    assert_eq!(fake.calls.lock().unwrap().as_slice(), [(path.clone(), Some("Robots, 2020-2026".to_string()))]);
    let job = r["job"].as_u64().unwrap();
    let j = job_entry(&mut s, job);
    assert_eq!((j["done"].as_u64(), j["total"].as_u64(), &j["status"]), (Some(4096), Some(4096), &json!("Uploaded to Google Photos")), "{j}");
    assert_eq!(j["upload"]["productUrl"], "https://photos.google.com/lr/photo/AF1Qip-new");
    assert!(j["label"].as_str().unwrap().contains("export.mp4"), "{j}");
    assert!(crate::photos::is_upload(&s, job));

    // in the background: the job is listed at once, and finishes with the item
    let r = s.execute("photos.upload", json!({"path": path})).unwrap();
    let job = r["job"].as_u64().unwrap();
    assert!(r.get("upload").is_none(), "only a blocking call carries the item: {r}");
    let j = wait_finished(&mut s, job);
    assert_eq!(j["upload"]["mediaItemId"], "AF1Qip-new", "{j}");
    assert!(j["result"].get("error").is_none(), "{j}");
}

#[test]
fn cancel_stops_the_upload() {
    let path = tmp_file("long.mov", 1000);
    let (mut s, _) = session(Fake { hang: true, ..Default::default() });
    let job = s.execute("photos.upload", json!({"path": path})).unwrap()["job"].as_u64().unwrap();
    s.execute("jobs.cancel", json!({"job": job})).unwrap();
    let j = wait_finished(&mut s, job);
    assert_eq!(j["result"]["error"], "cancelled", "{j}");
    assert_eq!(j["upload"], Value::Null, "{j}");
    assert!(crate::photos::last_export(&s).is_none(), "a cancelled upload is no export");
}

#[test]
fn hostile_parameters_start_no_job() {
    let (mut s, fake) = session(Fake::default());
    let empty = tmp_file("empty.mp4", 0);
    let audio = tmp_file("mix.wav", 10);
    let ok = tmp_file("ok.mp4", 10);
    let dir = std::env::temp_dir().to_string_lossy().into_owned();
    let long = "x".repeat(crate::photos::MAX_DESCRIPTION_CHARS + 1);
    for (p, want) in [
        (json!({}), "need `path`"),
        (json!({"path": ""}), "need `path`"),
        (json!({"path": "   "}), "need `path`"),
        (json!({"path": null}), "need `path`"),
        (json!({"path": 42}), "must be a string"),
        (json!({"path": ["/a.mp4"]}), "must be a string"),
        (json!({"path": "/no/such/file.mp4"}), "/no/such/file.mp4"),
        (json!({"path": dir}), "not a file"),
        (json!({"path": empty}), "is empty"),
        (json!({"path": audio}), ".wav"),
        (json!({"path": ok, "description": 7}), "must be a string"),
        (json!({"path": ok, "description": {"a": 1}}), "must be a string"),
        (json!({"path": ok, "description": long}), "at most 1000"),
    ] {
        let e = s.execute("photos.upload", p.clone()).unwrap_err().to_string();
        assert!(e.contains(want), "{p}: {e}");
    }
    assert!(s.jobs.is_empty() && s.photo_uploads.is_empty());
    assert!(fake.calls.lock().unwrap().is_empty());
    // exactly 1000 characters is fine, and a blank description is none
    let r = s.execute("photos.upload", json!({"path": ok, "description": "é".repeat(1000), "wait": true})).unwrap();
    assert_eq!(r["upload"]["mediaItemId"], "AF1Qip-new");
    s.execute("photos.upload", json!({"path": ok, "description": "  ", "wait": true})).unwrap();
    assert_eq!(fake.calls.lock().unwrap().last().unwrap().1, None);
}

#[test]
fn last_export_is_the_newest_finished_export_on_disk() {
    let (mut s, _) = session(Fake::default());
    assert_eq!(crate::photos::last_export(&s), None);
    let older = tmp_file("older.mp4", 10);
    let newer = tmp_file("newer.mp4", 10);
    let push = |s: &mut Session, path: &str, ok: bool| {
        let job = crate::narration::new_job(s, format!("Export {path}"));
        let r = if ok {
            Ok(filmcraft_export::Report { path: path.into(), frames: 1, seconds: 0.1, bytes: 10, render_fps: 1.0, extra_files: Vec::new() })
        } else {
            Err("encoder failed".to_string())
        };
        *job.result.lock().unwrap() = Some(r);
        s.export_offline.insert(job.id, Default::default());
        s.jobs.push(job);
    };
    push(&mut s, &older, true);
    push(&mut s, &newer, true);
    push(&mut s, "/tmp/failed.mp4", false);
    assert_eq!(crate::photos::last_export(&s).as_deref(), Some(newer.as_str()));
    // an upload job is not an export
    s.execute("photos.upload", json!({"path": older, "wait": true})).unwrap();
    assert_eq!(crate::photos::last_export(&s).as_deref(), Some(newer.as_str()));
    // a deleted export is not offered
    std::fs::remove_file(&newer).unwrap();
    assert_eq!(crate::photos::last_export(&s).as_deref(), Some(older.as_str()));
}
