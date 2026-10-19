//! Where a showcase's entries were filmed, from Google Photos.
//!
//! Google strips the location from the clips its API and picker hand out; `filmcraft-cli photos
//! locations` (the engine's `photos.locations`) reads each album item's place from the Google
//! Photos page into `[{fileName, place, estimated, …}]`, and `photos.applyLocations` stores it
//! with the project's items as their `Location` field. An entry's `place` (its lower third:
//! "Busan, South Korea · October 2023") is then filled — only where the plan gives none — from
//! its first shot that has one:
//!
//! - [`fill`]: from a locations file (`showcase --places-from-locations`), shots matched by
//!   file name as the engine matches them (case, folders and a photo or video extension
//!   ignored);
//! - [`from_items`]: from the shots' project items (`metadata.get` → `Location`), which
//!   [`crate::showcase::build`] does by itself.

use std::collections::HashMap;

use serde_json::{Value, json};

use crate::Executor;
use crate::showcase::Entry;
use filmcraft_engine::photo_locations::{LOCATION_FIELD, name_key, places_from};

/// Places by file name.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Places {
    by_name: HashMap<String, String>,
}

impl Places {
    /// The places of a locations file: an array of `{fileName, place, estimated}` or a report
    /// with `locations`. Items without a place are left out; hostile values are errors.
    pub fn from_json(v: &Value) -> Result<Self, String> {
        let list = places_from("places", v).map_err(|e| e.to_string())?;
        let mut by_name = HashMap::new();
        for p in list {
            by_name.entry(name_key(&p.file_name)).or_insert(p.place);
        }
        Ok(Self { by_name })
    }

    /// The place of the file `name`.
    pub fn of(&self, name: &str) -> Option<&str> {
        self.by_name.get(&name_key(name)).map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.by_name.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_name.is_empty()
    }
}

/// Give every entry without a `place` the place of its first shot that has one. How many
/// entries got one.
pub fn fill(entries: &mut [Entry], places: &Places) -> usize {
    let mut n = 0;
    for e in entries.iter_mut().filter(|e| e.place.as_deref().is_none_or(|p| p.trim().is_empty())) {
        if let Some(p) = e.shots.iter().find_map(|s| places.of(&s.name)) {
            e.place = Some(p.to_string());
            n += 1;
        }
    }
    n
}

/// Most items asked about by [`from_items`].
const MAX_ITEMS_ASKED: usize = 2000;

/// `entries` with places from their shots' project items (their `Location`, stored by
/// `photos.applyLocations`) where the plan gives none. An item that cannot be asked about
/// simply has no place.
pub async fn from_items(ex: &dyn Executor, entries: &[Entry]) -> Vec<Entry> {
    let mut out = entries.to_vec();
    let mut asked: HashMap<u64, Option<String>> = HashMap::new();
    for e in out.iter_mut().filter(|e| e.place.as_deref().is_none_or(|p| p.trim().is_empty())) {
        for s in &e.shots {
            if !asked.contains_key(&s.item) {
                if asked.len() >= MAX_ITEMS_ASKED {
                    break;
                }
                let place = match ex.execute("metadata.get", json!({"item": s.item})).await {
                    Ok(v) => v["metadata"][LOCATION_FIELD].as_str().map(str::trim).filter(|p| !p.is_empty()).map(|p| p.chars().take(160).collect()),
                    Err(_) => None,
                };
                asked.insert(s.item, place);
            }
            if let Some(Some(p)) = asked.get(&s.item) {
                e.place = Some(p.clone());
                break;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::showcase::Shot;
    use std::sync::Mutex;

    fn entry(label: &str, names: &[(u64, &str)], place: Option<&str>) -> Entry {
        Entry {
            label: label.into(),
            shots: names.iter().map(|(item, name)| Shot { item: *item, name: (*name).into(), ..Shot::default() }).collect(),
            place: place.map(str::to_string),
            ..Entry::default()
        }
    }

    const FILE: &str = r#"[
        {"fileName": "VID_20211021_111656.mp4", "date": "2021-10-21", "gmtOffset": "+04:00", "place": "Dubai World Trade Centre", "estimated": false},
        {"fileName": "IMG_5926.HEIC", "place": "Busan, South Korea", "estimated": true},
        {"fileName": "InShot_20200908_225922699.mp4", "place": null, "estimated": false},
        {"fileName": "", "error": "could not be read"}
    ]"#;

    #[test]
    fn entries_without_a_place_take_their_first_shots() {
        let places = Places::from_json(&serde_json::from_str(FILE).unwrap()).unwrap();
        assert_eq!(places.len(), 2);
        assert_eq!(places.of("vid_20211021_111656"), Some("Dubai World Trade Centre"));
        let mut entries = vec![
            entry("Promobot", &[(1, "VID_20211021_111656.mp4")], None),
            // the first shot has none, the second does (named without its extension in Photos)
            entry("Dilly", &[(2, "InShot_20200908_225922699.mp4"), (3, "IMG_5926.MOV"), (4, "img_5926.heic")], None),
            // the plan's own place stays
            entry("Ameca", &[(1, "VID_20211021_111656.mp4")], Some("Museum of the Future, Dubai")),
            entry("Mystery", &[(5, "unknown.mp4")], None),
        ];
        assert_eq!(fill(&mut entries, &places), 2);
        assert_eq!(entries[0].place.as_deref(), Some("Dubai World Trade Centre"));
        assert_eq!(entries[1].place.as_deref(), Some("Busan, South Korea"), "IMG_5926.MOV is IMG_5926 too");
        assert_eq!(entries[2].place.as_deref(), Some("Museum of the Future, Dubai"));
        assert_eq!(entries[3].place, None);
        // the field the plan carries
        assert_eq!(serde_json::to_value(&entries[0]).unwrap()["place"], "Dubai World Trade Centre");
    }

    #[test]
    fn hostile_locations_files_are_errors() {
        for bad in [json!("x"), json!([1]), json!([{"fileName": 3, "place": "p"}]), json!({"locations": 5})] {
            assert!(Places::from_json(&bad).is_err(), "{bad}");
        }
        assert!(Places::from_json(&json!([])).unwrap().is_empty());
    }

    struct Items(Mutex<Vec<u64>>);
    impl Executor for Items {
        fn execute(&self, id: &str, params: Value) -> crate::ExecFuture {
            let item = params["item"].as_u64().unwrap_or(0);
            self.0.lock().unwrap().push(item);
            let r = match (id, item) {
                ("metadata.get", 2) => Ok(json!({"item": 2, "metadata": {"Location": " Busan, South Korea "}})),
                ("metadata.get", 3) => Ok(json!({"item": 3, "metadata": {"Location": ""}})),
                ("metadata.get", _) => Ok(json!({"item": item, "metadata": {}})),
                _ => Err(format!("unexpected {id}")),
            };
            Box::pin(async move { r })
        }
    }

    #[tokio::test]
    async fn places_come_from_the_project_items() {
        let ex = Items(Mutex::new(Vec::new()));
        let entries = vec![
            entry("Dilly", &[(3, "a.mp4"), (2, "b.mp4"), (9, "c.mp4")], None),
            entry("Ameca", &[(2, "b.mp4")], Some("Museum of the Future")),
            entry("Again", &[(2, "b.mp4")], None),
        ];
        let out = from_items(&ex, &entries).await;
        assert_eq!(out[0].place.as_deref(), Some("Busan, South Korea"));
        assert_eq!(out[1].place.as_deref(), Some("Museum of the Future"));
        assert_eq!(out[2].place.as_deref(), Some("Busan, South Korea"));
        // each item asked once; none for an entry that has its place; none past the first found
        assert_eq!(*ex.0.lock().unwrap(), vec![3, 2]);
    }
}
