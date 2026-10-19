//! Tests of [`crate::photo_locations`]: places stored with media items matched by file name (one
//! undo step, nothing when nothing changes), the host's locator run as a job whose places are
//! stored when it is waited for, and hostile parameters refused before anything runs.

use std::sync::{Arc, Mutex};

use super::*;
use crate::photo_locations::{ESTIMATED_FIELD, LOCATION_FIELD, LocateRequest, PhotoLocator, name_key};
use filmcraft_project::ItemId;
use serde_json::{Value, json};

/// The demo project with two media items renamed as Google Photos names them.
fn project() -> (Session, ItemId, ItemId) {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    let media: Vec<ItemId> = s.project.items.values().filter(|it| matches!(it.kind, filmcraft_project::ItemKind::Media(_))).map(|it| it.id).collect();
    assert!(media.len() >= 2, "the demo has media");
    let (a, b) = (media[0], media[1]);
    s.execute("metadata.set", json!({"item": a.0, "field": "Name", "value": "VID_20211021_111656.mp4"})).unwrap();
    s.execute("metadata.set", json!({"item": b.0, "field": "Name", "value": "img_5926.heic"})).unwrap();
    (s, a, b)
}

fn field(s: &Session, id: ItemId, k: &str) -> Option<String> {
    s.project.item(id).and_then(|it| it.metadata.get(k).cloned())
}

#[test]
fn names_are_compared_without_case_folders_or_media_extensions() {
    assert_eq!(name_key("VID_20221011_132939.mp4"), "vid_20221011_132939");
    assert_eq!(name_key("VID_20221011_132939"), "vid_20221011_132939");
    assert_eq!(name_key("/Users/me/Movies/IMG_5926.HEIC"), "img_5926");
    assert_eq!(name_key("C:\\clips\\DJI_0496.MP4"), "dji_0496");
    assert_eq!(name_key("Every Robot I've Met - v4"), "every robot i've met - v4");
    assert_eq!(name_key("notes.txt"), "notes.txt", "only media extensions are dropped");
    assert_eq!(name_key(""), "");
}

#[test]
fn places_are_stored_with_the_items_they_name() {
    let (mut s, a, b) = project();
    let steps = s.history.undo.len();
    let r = s
        .execute(
            "photos.applyLocations",
            json!({"locations": [
                {"fileName": "VID_20211021_111656", "date": "2021-10-21", "gmtOffset": "+04:00", "place": "Dubai World Trade Centre", "estimated": false},
                {"fileName": "IMG_5926.HEIC", "place": "Busan, South Korea", "estimated": true},
                {"fileName": "IMG_0001.jpg", "place": "Nowhere"},
                {"fileName": "IMG_0002.jpg", "place": null},
                {"error": "could not be read"}
            ]}),
        )
        .unwrap();
    assert_eq!((r["places"].as_u64(), r["matched"].as_u64(), r["changed"].as_u64()), (Some(3), Some(2), Some(2)), "{r}");
    assert_eq!(r["unmatched"], json!(["IMG_0001.jpg"]));
    assert_eq!(field(&s, a, LOCATION_FIELD).as_deref(), Some("Dubai World Trade Centre"));
    assert_eq!(field(&s, a, ESTIMATED_FIELD), None);
    assert_eq!((field(&s, b, LOCATION_FIELD).as_deref(), field(&s, b, ESTIMATED_FIELD).as_deref()), (Some("Busan, South Korea"), Some("yes")));
    assert_eq!(s.history.undo.len(), steps + 1, "one undo step");
    // shown in the item's metadata
    let m = s.execute("metadata.get", json!({"item": a.0})).unwrap();
    assert_eq!(m["metadata"][LOCATION_FIELD], "Dubai World Trade Centre");
    // the same again changes nothing and adds no step
    let again =
        s.execute("photos.applyLocations", json!({"locations": [{"fileName": "vid_20211021_111656.MP4", "place": "Dubai World Trade Centre"}]})).unwrap();
    assert_eq!(again["changed"], 0);
    assert_eq!(s.history.undo.len(), steps + 1);
    // undo takes them away
    s.undo().unwrap();
    assert_eq!(field(&s, a, LOCATION_FIELD), None);
    assert_eq!(field(&s, b, ESTIMATED_FIELD), None);
    // `items` limits it
    let r = s
        .execute(
            "photos.applyLocations",
            json!({"items": [b.0], "locations": [{"fileName": "VID_20211021_111656.mp4", "place": "X"}, {"fileName": "IMG_5926", "place": "Y"}]}),
        )
        .unwrap();
    assert_eq!(r["matched"], 1);
    assert_eq!((field(&s, a, LOCATION_FIELD), field(&s, b, LOCATION_FIELD).as_deref()), (None, Some("Y")));
}

#[test]
fn places_are_read_from_the_file_a_read_wrote() {
    let (mut s, a, _) = project();
    let dir = std::env::temp_dir().join(format!("filmcraft-locations-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("locations.json");
    std::fs::write(&path, r#"[{"fileName": "VID_20211021_111656.mp4", "place": "Dubai World Trade Centre", "estimated": false}]"#).unwrap();
    let r = s.execute("photos.applyLocations", json!({"path": path.to_string_lossy()})).unwrap();
    assert_eq!(r["matched"], 1);
    assert_eq!(field(&s, a, LOCATION_FIELD).as_deref(), Some("Dubai World Trade Centre"));
    // a report with `locations` reads too
    std::fs::write(&path, r#"{"locations": [{"fileName": "VID_20211021_111656.mp4", "place": "Expo City Dubai"}]}"#).unwrap();
    s.execute("photos.applyLocations", json!({"path": path.to_string_lossy()})).unwrap();
    assert_eq!(field(&s, a, LOCATION_FIELD).as_deref(), Some("Expo City Dubai"));
    std::fs::write(&path, b"{not json").unwrap();
    assert!(s.execute("photos.applyLocations", json!({"path": path.to_string_lossy()})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn hostile_places_are_refused_and_change_nothing() {
    let (mut s, _, _) = project();
    let snap = (*s.project).clone();
    let many: Vec<Value> = (0..crate::photo_locations::MAX_LOCATIONS + 1).map(|k| json!({"fileName": format!("f{k}"), "place": "p"})).collect();
    for p in [
        json!({}),
        json!({"locations": "Dubai"}),
        json!({"locations": [5]}),
        json!({"locations": [{"fileName": 5, "place": "p"}]}),
        json!({"locations": [{"fileName": "a.mp4", "place": 5}]}),
        json!({"locations": [{"fileName": "a.mp4", "place": "p", "estimated": "yes"}]}),
        json!({"locations": [{"fileName": "a.mp4", "place": "x".repeat(10_000)}]}),
        json!({"locations": [{"fileName": "a\u{7}.mp4", "place": "p"}]}),
        json!({"locations": many}),
        json!({"locations": [], "path": "/tmp/x.json"}),
        json!({"path": 5}),
        json!({"path": "/no/such/locations.json"}),
        json!({"locations": [], "items": "all"}),
        json!({"locations": [], "items": [-1]}),
    ] {
        assert!(s.execute("photos.applyLocations", p.clone()).is_err(), "{p}");
        assert_eq!(*s.project, snap, "{p}");
    }
    // nothing named: nothing changes, no error
    let r = s.execute("photos.applyLocations", json!({"locations": []})).unwrap();
    assert_eq!(r["matched"], 0);
    assert_eq!(*s.project, snap);
}

/// A locator that answers a fixed album, or fails.
#[derive(Default)]
struct Fake {
    not_ready: Option<String>,
    fail: Option<String>,
    asked: Mutex<Vec<LocateRequest>>,
}

impl PhotoLocator for Fake {
    fn ready(&self) -> std::result::Result<(), String> {
        self.not_ready.clone().map_or(Ok(()), Err)
    }
    fn read(&self, request: &LocateRequest, progress: &filmcraft_export::Progress) -> std::result::Result<Value, String> {
        self.asked.lock().unwrap().push(request.clone());
        if let Some(e) = &self.fail {
            return Err(e.clone());
        }
        progress.done.store(2, std::sync::atomic::Ordering::Relaxed);
        Ok(json!({
            "albumUrl": "https://photos.google.com/album/A",
            "out": "/tmp/locations.json",
            "locations": [
                {"fileName": "VID_20211021_111656.mp4", "date": "2021-10-21", "gmtOffset": "+04:00", "place": "Dubai World Trade Centre", "estimated": false, "id": "P1"},
                {"fileName": "IMG_5926.HEIC", "date": "2023-10-18", "gmtOffset": "+09:00", "place": null, "estimated": false, "id": "P2"}
            ],
            "read": 2, "withoutPlace": 1, "failed": 0, "stopped": null
        }))
    }
}

#[test]
fn a_read_runs_as_a_job_and_its_places_are_stored() {
    let (mut s, a, b) = project();
    // without a locator the command is disabled and says why
    let e = s.execute("photos.locations", json!({"album": "Every Robot I've Met"})).unwrap_err().to_string();
    assert!(e.contains("not available"), "{e}");
    let fake = Arc::new(Fake::default());
    s.photo_locations.locator = Some(fake.clone());
    let r = s.execute("photos.locations", json!({"album": " Every Robot I've Met ", "maxItems": 5, "perItem": 60, "wait": true})).unwrap();
    assert_eq!(r["locations"]["read"], 2, "{r}");
    assert_eq!(r["applied"]["matched"], 1, "{r}");
    assert_eq!(field(&s, a, LOCATION_FIELD).as_deref(), Some("Dubai World Trade Centre"));
    assert_eq!(field(&s, b, LOCATION_FIELD), None, "no place: nothing stored");
    let asked = fake.asked.lock().unwrap().clone();
    assert_eq!(asked, vec![LocateRequest { album: Some("Every Robot I've Met".into()), max_items: Some(5), per_item_seconds: Some(60), ..Default::default() }]);
    // the job's entry carries the report
    let job = r["job"].as_u64().unwrap();
    let jobs = s.execute("jobs.list", json!({})).unwrap();
    let entry = jobs.as_array().unwrap().iter().find(|j| j["id"] == job).cloned().unwrap();
    assert_eq!(entry["locations"]["withoutPlace"], 1, "{entry}");
    // apply: false leaves the project be
    let (mut s, a, _) = project();
    s.photo_locations.locator = Some(Arc::new(Fake::default()));
    s.execute("photos.locations", json!({"albumUrl": "https://photos.google.com/album/A", "apply": false, "wait": true})).unwrap();
    assert_eq!(field(&s, a, LOCATION_FIELD), None);
}

#[test]
fn a_read_that_fails_or_cannot_start_says_why() {
    let (mut s, _, _) = project();
    s.photo_locations.locator = Some(Arc::new(Fake { not_ready: Some("Chrome control works on macOS only".into()), ..Default::default() }));
    let e = s.execute("photos.locations", json!({"album": "A"})).unwrap_err().to_string();
    assert!(e.contains("macOS"), "{e}");
    let listed = s.execute("command.list", json!({})).unwrap();
    let cmd = listed.as_array().unwrap().iter().find(|c| c["id"] == "photos.locations").cloned().unwrap();
    assert_eq!(cmd["enabled"], false, "{cmd}");
    let (mut s, _, _) = project();
    s.photo_locations.locator = Some(Arc::new(Fake { fail: Some("the chrome lock is held by another session".into()), ..Default::default() }));
    let e = s.execute("photos.locations", json!({"album": "A", "wait": true})).unwrap_err().to_string();
    assert!(e.contains("chrome lock"), "{e}");
    // hostile parameters: refused before any job
    let fake = Arc::new(Fake::default());
    let (mut s, _, _) = project();
    s.photo_locations.locator = Some(fake.clone());
    for p in [
        json!({}),
        json!({"album": ""}),
        json!({"album": 5}),
        json!({"album": "x".repeat(600)}),
        json!({"album": "a\u{1}b"}),
        json!({"albumUrl": "https://example.com/album/A"}),
        json!({"albumUrl": "javascript:alert(1)"}),
        json!({"album": "A", "maxItems": 0}),
        json!({"album": "A", "maxItems": 100_000}),
        json!({"album": "A", "maxItems": -3}),
        json!({"album": "A", "perItem": 1}),
        json!({"album": "A", "perItem": "slow"}),
        json!({"album": "A", "out": 7}),
    ] {
        assert!(s.execute("photos.locations", p.clone()).is_err(), "{p}");
    }
    assert!(fake.asked.lock().unwrap().is_empty());
    assert!(s.jobs.is_empty());
}
