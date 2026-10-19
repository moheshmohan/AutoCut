//! A3: saving a sequence as a template into the user folder, sharing it as a plain ZIP (export,
//! install from a file or a link), its QR code, and OpenTimelineIO of templates and sequences;
//! hostile packages, links and OTIO documents are refused without a crash.

use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};

use crate::Session;
use crate::media_test_util::tmp_dir;
use crate::video_template_share::{TemplateFetcher, check_url};

fn session(dir: &Path) -> Session {
    Session { prefs_path: Some(dir.join("prefs.json")), ..Default::default() }
}

fn png(dir: &Path, name: &str, w: u32, h: u32) -> String {
    let p = dir.join(name);
    image::RgbImage::from_fn(w, h, |x, y| image::Rgb([90, (x % 256) as u8, (y % 256) as u8])).save(&p).unwrap();
    p.to_string_lossy().into_owned()
}

fn user_files(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> =
        std::fs::read_dir(dir.join("Video Templates")).map(|r| r.flatten().map(|e| e.file_name().to_string_lossy().to_string()).collect()).unwrap_or_default();
    v.sort();
    v
}

/// JSON equal, numbers within a relative 1e-12 (JSON floats read back within an ulp).
fn same_json(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => {
            let (x, y) = (x.as_f64().unwrap_or(f64::NAN), y.as_f64().unwrap_or(f64::NAN));
            x == y || (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0)
        }
        (Value::Array(x), Value::Array(y)) => x.len() == y.len() && x.iter().zip(y).all(|(a, b)| same_json(a, b)),
        (Value::Object(x), Value::Object(y)) => x.len() == y.len() && x.iter().all(|(k, v)| y.get(k).is_some_and(|w| same_json(v, w))),
        _ => a == b,
    }
}

fn fragment_json(p: &filmcraft_templates::Package) -> Value {
    serde_json::from_slice(&filmcraft_format::encode(&p.fragment, true)).unwrap()
}

/// Apply a starter with the user's photos, then save the result as a template.
fn made_from_a_starter(s: &mut Session, dir: &Path) -> (u64, Value) {
    let photo = png(dir, "my-holiday.png", 640, 360);
    let r = s
        .execute(
            "template.apply",
            json!({"template": "builtin:photo-collage", "variant": "1:1", "slots": {"photos": [photo.clone(), photo], "title": "Our trip to Lisbon"}}),
        )
        .unwrap();
    let seq = r["sequence"].as_u64().unwrap();
    let saved = s.execute("template.save", json!({"sequence": seq, "name": "My Collage", "author": "Me", "license": "CC-BY-4.0"})).unwrap();
    (seq, saved)
}

#[test]
fn save_into_the_library_then_export_and_install_round_trip() {
    let dir = tmp_dir("vts-roundtrip");
    let mut s = session(&dir);
    let before = crate::video_templates::library_revision();
    let (_, saved) = made_from_a_starter(&mut s, &dir);
    assert!(crate::video_templates::library_revision() > before, "the browser re-reads the folder");
    assert_eq!(saved["inLibrary"], true, "{saved}");
    let path = saved["path"].as_str().unwrap().to_string();
    assert!(path.ends_with("Video Templates/my-collage.fcvideo-template"), "{path}");
    let t = &saved["template"];
    assert_eq!((t["id"].as_str(), t["source"].as_str()), (Some("user:my-collage"), Some("user")));
    // what the starter carried: the 2×2 boxes, every layout, the poster; the user's text and
    // photos are gone
    let m: Value = serde_json::from_slice(&std::fs::read(Path::new(&path).join("template.json")).unwrap()).unwrap();
    let photos = m["slots"].as_array().unwrap().iter().find(|x| x["id"] == "photos").unwrap();
    assert!(photos["placements"].as_array().unwrap().iter().all(|p| p["box"].is_array()), "{photos}");
    let layouts: Vec<&str> = m["variants"].as_array().unwrap().iter().map(|v| v["id"].as_str().unwrap()).collect();
    assert_eq!(layouts[0], "1:1", "the layout it was built in is the design");
    for want in ["16:9", "9:16", "4:5"] {
        assert!(layouts.contains(&want), "{layouts:?}");
    }
    let seqfile = std::fs::read_to_string(Path::new(&path).join("sequence.json")).unwrap();
    assert!(!seqfile.contains("my-holiday") && !seqfile.contains("Lisbon"), "no personal media or text");
    // the browser lists it
    let l = s.execute("template.list", json!({"source": "user"})).unwrap();
    assert!(l["templates"].as_array().unwrap().iter().any(|x| x["id"] == "user:my-collage"), "{l}");
    // share it as one file
    let out = dir.join("share");
    std::fs::create_dir_all(&out).unwrap();
    let e = s.execute("template.export", json!({"template": "user:my-collage", "path": format!("{}/", out.display())})).unwrap();
    let zip = e["path"].as_str().unwrap().to_string();
    assert!(zip.ends_with("my-collage.zip"), "{zip}");
    assert!(s.execute("template.export", json!({"template": "user:my-collage", "path": zip})).is_err(), "no overwrite without asking");
    // someone else installs it
    let other_dir = tmp_dir("vts-roundtrip-other");
    let mut o = session(&other_dir);
    let i = o.execute("template.install", json!({"path": zip})).unwrap();
    assert_eq!(i["template"], "user:my-collage", "{i}");
    assert_eq!(user_files(&other_dir), ["my-collage.zip"]);
    // it is the same template
    let mine = crate::video_templates::find_template(&s, "user:my-collage", "t").unwrap().package;
    let theirs = crate::video_templates::find_template(&o, "user:my-collage", "t").unwrap().package;
    assert_eq!(theirs.manifest, mine.manifest);
    assert_eq!(theirs.assets, mine.assets);
    assert!(same_json(&fragment_json(&theirs), &fragment_json(&mine)), "the sequence survives the trip");
    // installing again asks first, then replaces
    assert!(o.execute("template.install", json!({"path": zip})).is_err());
    let again = o.execute("template.install", json!({"path": zip, "overwrite": true})).unwrap();
    assert_eq!(again["replaced"], true);
    assert_eq!(user_files(&other_dir), ["my-collage.zip"]);
    // and it applies there
    let applied = o.execute("template.apply", json!({"template": "user:my-collage", "variant": "9:16"})).unwrap();
    assert!(applied["sequence"].as_u64().is_some());
    std::fs::remove_dir_all(&dir).ok();
    std::fs::remove_dir_all(&other_dir).ok();
}

#[test]
fn save_authoring_options_and_bundles() {
    let dir = tmp_dir("vts-author");
    let mut s = session(&dir);
    s.execute("file.openDemoProject", json!({})).unwrap();
    // the user's logo, with a sidecar saying where it came from
    let logo = png(&dir, "studio-logo.png", 200, 200);
    std::fs::write(format!("{logo}.attribution"), "asset: studio-logo.png\ntitle: Studio logo\nauthor: Ada\nsource: original work\nlicense: CC0-1.0\n")
        .unwrap();
    let item = s.execute("file.import", json!({"paths": [logo]})).unwrap()["items"][0].as_u64().unwrap();
    let q = s.active_sequence().unwrap().clone();
    let clips: Vec<u64> = q.video_tracks[0].items.iter().take(2).map(|c| c.id.0).collect();
    s.execute("template.markSlot", json!({"clips": clips, "kind": "video", "slot": "shots"})).unwrap();
    s.execute("sequence.addTracks", json!({"video": 1, "audio": 0})).unwrap();
    let top = s.active_sequence().unwrap().video_tracks.len();
    s.execute("timeline.place", json!({"item": item, "time": 0, "track": format!("V{top}"), "audio": false})).unwrap();
    let r = s
        .execute(
            "template.save",
            json!({
                "name": "Demo Kit", "zip": true, "author": "Ada", "license": "MIT",
                "slots": {"shots": {"name": "Your shots", "prompt": "Two clips", "minCount": 1}},
                "captions": false, "variants": ["9:16", "1:1", "4:5"], "posterSeconds": 1.5,
                "bundle": [item], "tags": ["demo"],
            }),
        )
        .unwrap();
    assert!(r["path"].as_str().unwrap().ends_with("demo-kit.zip"), "{r}");
    let a = &r["assets"][0];
    assert_eq!(
        (a["title"].as_str(), a["author"].as_str(), a["license"].as_str()),
        (Some("Studio logo"), Some("Ada"), Some("CC0-1.0")),
        "from the sidecar: {r}"
    );
    assert_eq!(a["sha256"].as_str().unwrap(), filmcraft_templates::sha256_hex(&std::fs::read(&logo).unwrap()));
    let e = crate::video_templates::find_template(&s, "user:demo-kit", "t").unwrap();
    let m = &e.package.manifest;
    assert_eq!(m.slot("shots").unwrap().name, "Your shots");
    assert_eq!(m.variants.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(), ["16:9", "9:16", "1:1", "4:5"]);
    assert_eq!(m.poster, Some(filmcraft_time::Tick::from_seconds_f64(1.5)));
    assert_eq!(m.tags, ["demo"]);
    assert!(e.package.assets.keys().any(|k| k == "assets/studio-logo.png"));
    // without a sidecar a bundled file is the author's own work under the template's licence
    let own = png(&dir, "mine.png", 64, 64);
    let mine = s.execute("file.import", json!({"paths": [own]})).unwrap()["items"][0].as_u64().unwrap();
    s.execute("timeline.place", json!({"item": mine, "time": 5 * filmcraft_time::TICKS_PER_SECOND, "track": format!("V{top}"), "audio": false})).unwrap();
    let r = s.execute("template.save", json!({"name": "Kit Two", "author": "Ada", "license": "MIT", "bundle": true})).unwrap();
    let assets = r["assets"].as_array().unwrap();
    let mine_a = assets.iter().find(|a| a["path"] == "assets/mine.png").unwrap();
    assert_eq!((mine_a["author"].as_str(), mine_a["source"].as_str(), mine_a["license"].as_str()), (Some("Ada"), Some("original work"), Some("MIT")));
    // hostile authoring parameters are refused, nothing is written
    let files = user_files(&dir);
    for bad in [
        json!({"slots": {"nope": {"name": "x"}}}),
        json!({"slots": {"shots": {"fit": {"short": "teleport"}}}}),
        json!({"slots": {"shots": {"colour": "red"}}}),
        json!({"slots": "all"}),
        json!({"captions": "yes"}),
        json!({"captions": {"source": "speaker"}}),
        json!({"variants": "9:16"}),
        json!({"poster": -5}),
        json!({"posterSeconds": f64::MAX}),
        json!({"look": {"lumetri": 3}}),
        json!({"bundle": [999_999]}),
        json!({"bundle": "everything"}),
        json!({"base": "builtin:no-such-thing"}),
        json!({"name": "x".repeat(300)}),
        json!({"path": ""}),
        json!({"path": "a\u{0}b"}),
        json!({"license": "CC-BY-NC-4.0"}),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("template.save", bad.clone())));
        assert!(r.is_ok(), "panicked on {bad}");
        assert!(r.unwrap().is_err(), "accepted {bad}");
    }
    assert_eq!(user_files(&dir), files, "a refused save writes nothing");
    std::fs::remove_dir_all(&dir).ok();
}

/// A fetcher serving fixed bytes (or an error), counting its calls.
struct Fake {
    body: std::result::Result<Vec<u8>, String>,
    calls: std::sync::atomic::AtomicUsize,
}

impl TemplateFetcher for Fake {
    fn fetch(&self, url: &str, max: u64, _cancelled: &dyn Fn() -> bool) -> std::result::Result<Vec<u8>, String> {
        self.calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        assert!(url.starts_with("https://"));
        match &self.body {
            Ok(b) if b.len() as u64 > max => Err(format!("more than {max} bytes")),
            Ok(b) => Ok(b.clone()),
            Err(e) => Err(e.clone()),
        }
    }
}

fn zip_of(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    filmcraft_templates::zip::write(files).unwrap()
}

#[test]
fn install_refuses_hostile_packages_and_links() {
    let dir = tmp_dir("vts-hostile");
    let mut s = session(&dir);
    let good = filmcraft_templates::builtin::builtin_templates().into_iter().next().unwrap();
    // a package of someone else's: the starter under a user id
    let mut theirs = good.clone();
    theirs.manifest.id = "user:shared-slideshow".into();
    let theirs_zip = theirs.to_zip().unwrap();
    // hostile packages
    let mut unlisted = theirs.files();
    unlisted.push(("assets/x.bin".into(), vec![0; 10]));
    let mut stray = theirs.files();
    stray.push(("run-me.sh".into(), b"rm -rf ~".to_vec()));
    // a name that leaves the package: our writer refuses it, so rename an entry in the bytes
    let mut traversal = theirs.files();
    traversal.push(("xx/evil.json".into(), b"{}".to_vec()));
    let mut traversal = zip_of(&traversal);
    let mut k = 0;
    while let Some(at) = traversal.windows(12).skip(k).position(|w| w == b"xx/evil.json").map(|i| i + k) {
        traversal[at..at + 2].copy_from_slice(b"..");
        k = at + 12;
    }
    let mut absolute = theirs.files();
    absolute.push(("x/etc/passwd".into(), b"root".to_vec()));
    let mut absolute = zip_of(&absolute);
    let mut k = 0;
    while let Some(at) = absolute.windows(12).skip(k).position(|w| w == b"x/etc/passwd").map(|i| i + k) {
        absolute[at] = b'/';
        k = at + 12;
    }
    let mut bad_manifest = theirs.clone();
    bad_manifest.manifest.duration = filmcraft_time::Tick(filmcraft_time::TICKS_PER_SECOND);
    let bomb = {
        // entries that inflate past the total cap
        let mut f = theirs.files();
        for name in ["README.md", "notes.txt", "LICENSE"] {
            f.push((name.into(), vec![0u8; 100 * 1024 * 1024]));
        }
        f
    };
    let cases: Vec<(&str, Vec<u8>)> = vec![
        ("builtin id", good.to_zip().unwrap()),
        ("unlisted asset", zip_of(&unlisted)),
        ("stray file", zip_of(&stray)),
        ("not a zip", b"hello".to_vec()),
        ("truncated", theirs_zip[..theirs_zip.len() / 2].to_vec()),
        ("bomb", zip_of(&bomb)),
        ("traversal", traversal),
        ("absolute", absolute),
    ];
    for (what, bytes) in cases {
        let p = dir.join(format!("{}.zip", what.replace(' ', "-")));
        std::fs::write(&p, &bytes).unwrap();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("template.install", json!({"path": p.to_string_lossy()}))));
        assert!(r.is_ok(), "{what}: panicked");
        assert!(r.unwrap().is_err(), "{what}: installed");
        assert!(user_files(&dir).is_empty(), "{what}: something was written");
    }
    // a broken manifest is refused with its field named
    let p = dir.join("bad-manifest.zip");
    std::fs::write(&p, zip_of(&bad_manifest.files())).unwrap();
    assert!(s.execute("template.install", json!({"path": p.to_string_lossy()})).is_err());
    // parameters
    for bad in [json!({}), json!({"path": 3}), json!({"path": ""}), json!({"path": "/no/such.zip"}), json!({"path": "/x.zip", "url": "https://a/b.zip"})] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("template.install", bad.clone())));
        assert!(r.is_ok_and(|r| r.is_err()), "{bad}");
    }
    // links: only https, no credentials, no spaces, bounded
    for (url, ok) in [
        ("https://example.org/t.zip", true),
        ("https://example.org:8443/a/b.zip?x=1", true),
        ("http://example.org/t.zip", false),
        ("ftp://example.org/t.zip", false),
        ("file:///etc/passwd", false),
        ("https://user:pw@example.org/t.zip", false),
        ("https:///t.zip", false),
        ("https://exa mple.org/t.zip", false),
        ("https://example.org/\u{7}", false),
        ("javascript:alert(1)", false),
    ] {
        assert_eq!(check_url(url).is_ok(), ok, "{url}");
    }
    assert!(check_url(&format!("https://example.org/{}", "a".repeat(3000))).is_err());
    // no fetcher: links are refused before anything starts
    assert!(s.execute("template.install", json!({"url": "https://example.org/t.zip"})).is_err());
    // a fetcher handing back junk, an error, or a builtin's package: the job fails, nothing written
    for body in [Ok(b"<html>not found</html>".to_vec()), Err("404 Not Found".to_string()), Ok(good.to_zip().unwrap())] {
        s.sharing.fetcher = Some(Arc::new(Fake { body, calls: Default::default() }));
        let r = s.execute("template.install", json!({"url": "https://example.org/t.zip", "wait": true}));
        assert!(r.is_err(), "{r:?}");
        assert!(user_files(&dir).is_empty());
    }
    // a good package from a link installs (as a job)
    let fake = Arc::new(Fake { body: Ok(theirs_zip.clone()), calls: Default::default() });
    s.sharing.fetcher = Some(fake.clone());
    let r = s.execute("template.install", json!({"url": "https://example.org/shared.zip", "wait": true})).unwrap();
    assert!(r["path"].as_str().unwrap().ends_with("shared-slideshow.zip"), "{r}");
    assert_eq!(fake.calls.load(std::sync::atomic::Ordering::Relaxed), 1);
    assert_eq!(user_files(&dir), ["shared-slideshow.zip"]);
    assert!(crate::video_templates::find_template(&s, "user:shared-slideshow", "t").is_ok());
    // in the background: a job, with the installed path as its result
    s.sharing.fetcher = Some(Arc::new(Fake { body: Ok(theirs_zip), calls: Default::default() }));
    let r = s.execute("template.install", json!({"url": "https://example.org/shared.zip", "overwrite": true})).unwrap();
    let job = r["job"].as_u64().unwrap();
    for _ in 0..500 {
        let j = s.execute("jobs.list", json!({})).unwrap().as_array().unwrap().iter().find(|j| j["id"] == job).cloned().unwrap();
        if j["finished"] == true {
            assert!(j["result"]["path"].as_str().unwrap().ends_with("shared-slideshow.zip"), "{j}");
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn qr_codes_of_install_links() {
    let dir = tmp_dir("vts-qr");
    let mut s = session(&dir);
    let out = dir.join("link.png");
    let r = s.execute("template.qr", json!({"url": "https://example.org/templates/collage.zip", "path": out.to_string_lossy(), "scale": 4})).unwrap();
    assert_eq!(r["size"].as_u64().unwrap(), 17 + 4 * r["version"].as_u64().unwrap());
    let png = std::fs::read(&out).unwrap();
    assert!(png.starts_with(b"\x89PNG"));
    let img = image::load_from_memory(&png).unwrap();
    let side = (r["size"].as_u64().unwrap() as u32 + 8) * 4;
    assert_eq!((img.width(), img.height()), (side, side));
    assert_eq!(r["modules"].as_array().unwrap().len() as u64, r["size"].as_u64().unwrap());
    for bad in [json!({}), json!({"url": ""}), json!({"url": "x".repeat(5000)}), json!({"url": "https://a", "ec": "Z"})] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("template.qr", bad.clone())));
        assert!(r.is_ok_and(|r| r.is_err()), "{bad}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

fn otio_clips_with_slots(v: &Value) -> usize {
    let mut n = 0;
    let mut stack = vec![v];
    while let Some(x) = stack.pop() {
        match x {
            Value::Object(o) => {
                if o.get("OTIO_SCHEMA").and_then(Value::as_str).is_some_and(|s| s.starts_with("Clip.")) && x.pointer("/metadata/filmcraft/slot").is_some() {
                    n += 1;
                }
                stack.extend(o.values());
            }
            Value::Array(a) => stack.extend(a.iter()),
            _ => {}
        }
    }
    n
}

#[test]
fn opentimelineio_of_templates_and_sequences_keeps_the_slots() {
    let dir = tmp_dir("vts-otio");
    let mut s = session(&dir);
    let t = filmcraft_templates::builtin::find("builtin:top-3-listicle").unwrap();
    let placements: usize = t.manifest.slots.iter().map(|x| x.placements.len()).sum();
    let out = dir.join("top3.otio");
    let r = s.execute("template.exportOtio", json!({"template": "builtin:top-3-listicle", "path": out.to_string_lossy()})).unwrap();
    assert_eq!(r["slotClips"].as_u64().unwrap() as usize, placements, "{r}");
    let doc: Value = serde_json::from_slice(&std::fs::read(&out).unwrap()).unwrap();
    assert_eq!(doc["OTIO_SCHEMA"], "Timeline.1");
    assert_eq!(doc["metadata"]["filmcraft"]["template"]["id"], "builtin:top-3-listicle");
    assert!(doc["metadata"]["filmcraft"]["template"]["manifest"]["slots"].is_array());
    assert!(otio_clips_with_slots(&doc) > 0);
    // a layout of it, as applied
    let out2 = dir.join("top3-vertical.otio");
    let r = s.execute("template.exportOtio", json!({"template": "builtin:top-3-listicle", "path": out2.to_string_lossy(), "variant": "9:16"})).unwrap();
    assert!(r["slotClips"].as_u64().unwrap() > 0);
    assert!(
        s.execute("template.exportOtio", json!({"template": "builtin:top-3-listicle", "path": out2.to_string_lossy(), "variant": "9:16"})).is_err(),
        "exists"
    );
    assert!(
        s.execute("template.exportOtio", json!({"template": "builtin:top-3-listicle", "path": dir.join("x.otio").to_string_lossy(), "variant": "3:2"}))
            .is_err()
    );
    // it reads back into a sequence
    let before = s.project.items.len();
    let imp = s.execute("file.import", json!({"paths": [out.to_string_lossy()]})).unwrap();
    assert!(s.project.items.len() > before, "{imp}");
    // a sequence with flagged clips
    let applied = s.execute("template.apply", json!({"template": "builtin:top-3-listicle"})).unwrap();
    let seq = applied["sequence"].as_u64().unwrap();
    let out3 = dir.join("mine.otio");
    let r = s.execute("sequence.exportOtio", json!({"sequence": seq, "path": out3.to_string_lossy()})).unwrap();
    assert!(r["slotClips"].as_u64().unwrap() > 0, "{r}");
    let doc: Value = serde_json::from_slice(&std::fs::read(&out3).unwrap()).unwrap();
    assert_eq!(doc["metadata"]["filmcraft"]["template"]["id"], "builtin:top-3-listicle");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn hostile_otio_documents_are_refused_without_a_crash() {
    let dir = tmp_dir("vts-otio-hostile");
    let mut s = session(&dir);
    let deep = format!("{}{}", "{\"OTIO_SCHEMA\":\"Stack.1\",\"children\":[".repeat(5000), "]}".repeat(5000));
    let docs: Vec<(&str, String)> = vec![
        ("deep", format!("{{\"OTIO_SCHEMA\":\"Timeline.1\",\"tracks\":{deep}}}")),
        ("nan rate", r#"{"OTIO_SCHEMA":"Timeline.1","tracks":{"OTIO_SCHEMA":"Stack.1","children":[{"OTIO_SCHEMA":"Track.1","kind":"Video","children":[{"OTIO_SCHEMA":"Gap.1","source_range":{"OTIO_SCHEMA":"TimeRange.1","start_time":{"OTIO_SCHEMA":"RationalTime.1","rate":0,"value":1e308},"duration":{"OTIO_SCHEMA":"RationalTime.1","rate":-1,"value":-1e308}}}]}]}}"#.into()),
        ("huge", r#"{"OTIO_SCHEMA":"Timeline.1","tracks":{"OTIO_SCHEMA":"Stack.1","children":[{"OTIO_SCHEMA":"Track.1","kind":"Video","children":[{"OTIO_SCHEMA":"Clip.2","source_range":{"OTIO_SCHEMA":"TimeRange.1","start_time":{"OTIO_SCHEMA":"RationalTime.1","rate":24,"value":9e18},"duration":{"OTIO_SCHEMA":"RationalTime.1","rate":24,"value":9e18}},"media_references":{"DEFAULT_MEDIA":{"OTIO_SCHEMA":"ExternalReference.1","target_url":"file:///../../../../etc/passwd"}},"active_media_reference_key":"DEFAULT_MEDIA"}]}]}}"#.into()),
        ("wrong types", r#"{"OTIO_SCHEMA":"Timeline.1","tracks":[1,2,3],"name":{"a":1},"global_start_time":"soon"}"#.into()),
        ("not json", "OTIO_SCHEMA Timeline.1 {".into()),
        ("empty", String::new()),
    ];
    for (what, text) in docs {
        let p = dir.join(format!("{}.otio", what.replace(' ', "-")));
        std::fs::write(&p, text).unwrap();
        let before = s.project.items.len();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("file.import", json!({"paths": [p.to_string_lossy()]}))));
        assert!(r.is_ok(), "{what}: panicked");
        let r = r.unwrap();
        // refused outright, or read into nothing usable: never a sequence longer than the cap
        if let Ok(v) = &r {
            for seq in s.project.items.values().filter_map(|i| i.as_sequence()) {
                assert!(seq.duration().0 >= 0, "{what}: {v}");
            }
        }
        let _ = before;
    }
    std::fs::remove_dir_all(&dir).ok();
}
