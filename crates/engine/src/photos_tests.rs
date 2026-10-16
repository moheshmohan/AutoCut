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
    /// The album permissions are missing: this is why.
    albums_not_ready: Option<String>,
    hang: bool,
    calls: Mutex<Vec<(String, Option<String>)>>,
    /// Album calls: `(what, args)`.
    album_calls: Mutex<Vec<(String, Value)>>,
}

impl PhotosUploader for Fake {
    fn albums_ready(&self) -> std::result::Result<(), String> {
        self.albums_not_ready.clone().map_or(Ok(()), Err)
    }
    fn upload_to_album(&self, path: &str, description: Option<&str>, album: &str, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        self.album_calls.lock().unwrap().push(("upload".into(), json!({"path": path, "album": album})));
        let mut item = self.upload(path, description, progress)?;
        item["album"] = json!({"id": "AF1QipAlbum", "title": album});
        Ok(item)
    }
    fn create_album(&self, title: &str, description: Option<&str>) -> std::result::Result<Value, String> {
        self.album_calls.lock().unwrap().push(("create".into(), json!({"title": title, "description": description})));
        Ok(json!({"id": "AF1QipAlbum", "title": title, "created": true}))
    }
    fn albums(&self) -> std::result::Result<Value, String> {
        self.album_calls.lock().unwrap().push(("list".into(), Value::Null));
        Ok(json!({"albums": [{"id": "AF1QipAlbum", "title": "Robots"}], "complete": true}))
    }
    fn add_to_album(&self, album: &str, ids: &[String]) -> std::result::Result<Value, String> {
        self.album_calls.lock().unwrap().push(("add".into(), json!({"album": album, "ids": ids})));
        if album == "Missing" {
            return Err("FilmCraft has made no album `Missing`".into());
        }
        Ok(json!({"added": ids.len()}))
    }
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

fn enabled(s: &mut Session, id: &str) -> Value {
    let listed = s.execute("command.list", json!({})).unwrap();
    listed.as_array().unwrap().iter().find(|c| c["id"] == id).cloned().unwrap()["enabled"].clone()
}

const ALBUM_COMMANDS: [&str; 3] = ["photos.createAlbum", "photos.albums", "photos.addToAlbum"];

#[test]
fn album_commands_are_disabled_with_their_reason() {
    // no uploader at all
    let mut s = Session::default();
    for id in ALBUM_COMMANDS {
        let e = s.execute(id, json!({"title": "Robots", "album": "Robots", "mediaItemIds": ["a"]})).unwrap_err().to_string();
        assert!(e.contains("albums are not available"), "{id}: {e}");
        assert_eq!(enabled(&mut s, id), false, "{id}");
    }
    // an uploader that does no albums (the trait's defaults)
    struct UploadsOnly;
    impl PhotosUploader for UploadsOnly {
        fn ready(&self) -> std::result::Result<(), String> {
            Ok(())
        }
        fn upload(&self, _: &str, _: Option<&str>, _: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
            Ok(json!({}))
        }
    }
    let mut s = Session { photos: Some(Arc::new(UploadsOnly)), ..Default::default() };
    assert!(s.execute("photos.albums", json!({})).unwrap_err().to_string().contains("not available"));
    let path = tmp_file("no-albums.mp4", 10);
    let e = s.execute("photos.upload", json!({"path": path, "album": "Robots"})).unwrap_err().to_string();
    assert!(e.contains("albums are not available"), "{e}");
    assert!(s.jobs.is_empty());
    // signed in without the album permissions: the reason names the fix
    let why = "Google Photos needs your consent before FilmCraft can make albums: run `filmcraft-cli photos auth --albums`";
    let (mut s, fake) = session(Fake { albums_not_ready: Some(why.into()), ..Default::default() });
    for id in ALBUM_COMMANDS {
        let e = s.execute(id, json!({"title": "Robots", "album": "Robots", "mediaItemIds": ["a"]})).unwrap_err().to_string();
        assert!(e.contains("photos auth --albums"), "{id}: {e}");
        assert_eq!(enabled(&mut s, id), false, "{id}");
    }
    // uploads without an album still work; with one they say what to run
    assert_eq!(enabled(&mut s, "photos.upload"), true);
    let e = s.execute("photos.upload", json!({"path": path, "album": "Robots"})).unwrap_err().to_string();
    assert!(e.contains("photos auth --albums"), "{e}");
    assert!(s.jobs.is_empty() && fake.album_calls.lock().unwrap().is_empty());
}

#[test]
fn albums_are_created_listed_and_added_to() {
    let (mut s, fake) = session(Fake::default());
    for id in ALBUM_COMMANDS {
        assert_eq!(enabled(&mut s, id), true, "{id}");
    }
    let r = s.execute("photos.createAlbum", json!({"title": "  Robot showcase ", "description": " Every robot "})).unwrap();
    assert_eq!((r["id"].as_str(), r["title"].as_str()), (Some("AF1QipAlbum"), Some("Robot showcase")), "{r}");
    let r = s.execute("photos.albums", json!({})).unwrap();
    assert_eq!(r["albums"][0]["title"], "Robots");
    let r = s.execute("photos.addToAlbum", json!({"album": "Robots", "mediaItemIds": [" AF1Qip-a ", "AF1Qip-b"]})).unwrap();
    assert_eq!(r["added"], 2);
    let e = s.execute("photos.addToAlbum", json!({"album": "Missing", "mediaItemIds": ["AF1Qip-a"]})).unwrap_err().to_string();
    assert!(e.contains("no album `Missing`"), "{e}");
    let calls = fake.album_calls.lock().unwrap().clone();
    assert_eq!(calls[0], ("create".to_string(), json!({"title": "Robot showcase", "description": "Every robot"})));
    assert_eq!(calls[1].0, "list");
    assert_eq!(calls[2], ("add".to_string(), json!({"album": "Robots", "ids": ["AF1Qip-a", "AF1Qip-b"]})));
    // none of them is an undoable edit or a job
    assert!(s.jobs.is_empty());
}

#[test]
fn uploads_go_into_an_album_as_a_job() {
    let path = tmp_file("into-album.mp4", 2048);
    let (mut s, fake) = session(Fake::default());
    let r = s.execute("photos.upload", json!({"path": path, "album": " Robot showcase ", "description": "Robots", "wait": true})).unwrap();
    assert_eq!(r["upload"]["album"]["title"], "Robot showcase", "{r}");
    assert_eq!(fake.album_calls.lock().unwrap().as_slice(), [("upload".to_string(), json!({"path": path, "album": "Robot showcase"}))]);
    let j = job_entry(&mut s, r["job"].as_u64().unwrap());
    assert!(j["label"].as_str().unwrap().contains("album Robot showcase"), "{j}");
    // a blank album is no album
    let r = s.execute("photos.upload", json!({"path": path, "album": "  ", "wait": true})).unwrap();
    assert!(r["upload"].get("album").is_none(), "{r}");
    assert_eq!(fake.album_calls.lock().unwrap().len(), 1);
}

#[test]
fn hostile_album_parameters_call_nothing() {
    let (mut s, fake) = session(Fake::default());
    let ok = tmp_file("hostile-album.mp4", 10);
    let long_title = "t".repeat(crate::photos::MAX_ALBUM_TITLE_CHARS + 1);
    let long_note = "n".repeat(crate::photos::MAX_DESCRIPTION_CHARS + 1);
    let too_many: Vec<String> = (0..crate::photos::MAX_ALBUM_ITEMS + 1).map(|i| format!("id{i}")).collect();
    for (id, p, want) in [
        ("photos.createAlbum", json!({}), "need `title`"),
        ("photos.createAlbum", json!({"title": " "}), "need `title`"),
        ("photos.createAlbum", json!({"title": null}), "need `title`"),
        ("photos.createAlbum", json!({"title": 5}), "must be a string"),
        ("photos.createAlbum", json!({"title": ["Robots"]}), "must be a string"),
        ("photos.createAlbum", json!({"title": long_title}), "at most 500"),
        ("photos.createAlbum", json!({"title": "Robots", "description": long_note}), "at most 1000"),
        ("photos.createAlbum", json!({"title": "Robots", "description": false}), "must be a string"),
        ("photos.addToAlbum", json!({"mediaItemIds": ["a"]}), "need `album`"),
        ("photos.addToAlbum", json!({"album": "Robots"}), "need `mediaItemIds`"),
        ("photos.addToAlbum", json!({"album": "Robots", "mediaItemIds": []}), "is empty"),
        ("photos.addToAlbum", json!({"album": "Robots", "mediaItemIds": "a"}), "must be a list"),
        ("photos.addToAlbum", json!({"album": "Robots", "mediaItemIds": [1, 2]}), "must be a media item id"),
        ("photos.addToAlbum", json!({"album": "Robots", "mediaItemIds": ["a", ""]}), "must be a media item id"),
        ("photos.addToAlbum", json!({"album": "Robots", "mediaItemIds": ["x".repeat(513)]}), "must be a media item id"),
        ("photos.addToAlbum", json!({"album": "Robots", "mediaItemIds": too_many}), "at most 20000"),
        ("photos.addToAlbum", json!({"album": {"id": 1}, "mediaItemIds": ["a"]}), "must be a string"),
        ("photos.upload", json!({"path": ok, "album": 7}), "must be a string"),
        ("photos.upload", json!({"path": ok, "album": long_title}), "at most 500"),
    ] {
        let e = s.execute(id, p.clone()).unwrap_err().to_string();
        assert!(e.contains(want), "{id} {p}: {e}");
    }
    assert!(fake.album_calls.lock().unwrap().is_empty() && fake.calls.lock().unwrap().is_empty());
    assert!(s.jobs.is_empty());
}
