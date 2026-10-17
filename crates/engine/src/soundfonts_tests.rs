//! Tests of [`crate::soundfonts`] (`music.soundfonts`, `music.downloadSoundfont`,
//! `music.useSoundfont`) and [`crate::music_preview`] (`music.preview`): music on a user's `.sf2`
//! (a tiny generated one; GeneralUser GS is never downloaded in tests), the same arrangement as on
//! the oscillators, the preview placed as heard, consent before any download, hostile parameters.

use super::*;
use filmcraft_soundfont::{catalog, synthetic};
use serde_json::json;

fn tmp(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("filmcraft-sf-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

/// The tiny test font written to `dir`.
fn tiny_font(dir: &std::path::Path) -> String {
    let p = dir.join("tiny.sf2");
    std::fs::write(&p, synthetic::tiny()).unwrap();
    p.to_string_lossy().into_owned()
}

fn gugs_installed() -> bool {
    crate::transcript::models_dir().is_some_and(|d| catalog::installed(&d, &catalog::GENERALUSER_GS))
}

#[test]
fn soundfonts_lists_generaluser_gs_with_its_licence() {
    let mut s = demo();
    let r = s.execute("music.soundfonts", json!({})).unwrap();
    let f = &r["soundfonts"][0];
    assert_eq!(f["id"], "generaluser-gs");
    assert_eq!(f["author"], "S. Christian Collins");
    assert_eq!(f["size"].as_u64(), Some(catalog::size(&catalog::GENERALUSER_GS)));
    assert!(f["license"].as_str().unwrap().contains("GeneralUser GS License v2.0"));
    assert!(f["licenseText"].as_str().unwrap().contains("I cannot be 100% sure where all of the samples originated"));
    assert!(f["provenance"].as_str().unwrap().contains("100% sure"));
    assert!(f["source"].as_str().unwrap().starts_with("https://github.com/mrbumpy409/GeneralUser-GS"));
    assert_eq!(r["selected"], "auto");
    assert!(r["choices"].as_array().unwrap().iter().any(|c| c == "oscillators"));
    if !gugs_installed() {
        assert_eq!(r["active"]["kind"], "oscillators", "{r}");
        assert_eq!(f["installed"], false);
    }
}

#[test]
fn a_download_needs_consent_and_hostile_requests_are_refused() {
    let mut s = demo();
    let jobs = s.jobs.len();
    if !gugs_installed() {
        let e = s.execute("music.downloadSoundfont", json!({})).unwrap_err().to_string();
        assert!(e.contains("consent") && e.contains("MB") && e.contains("github.com/mrbumpy409") && e.contains("GeneralUser GS License"), "{e}");
        let e = s.execute("music.downloadSoundfont", json!({"consent": false})).unwrap_err().to_string();
        assert!(e.contains("consent"), "{e}");
        // and a request for it names the download
        let e = s.execute("music.useSoundfont", json!({"soundfont": "generaluser-gs"})).unwrap_err().to_string();
        assert!(e.contains("music.downloadSoundfont"), "{e}");
        let e = s.execute("audio.generate", json!({"kind": "music", "duration": 2, "soundfont": "generaluser-gs", "place": false})).unwrap_err().to_string();
        assert!(e.contains("music.downloadSoundfont"), "{e}");
    }
    for p in [json!({"soundfont": 5, "consent": true}), json!({"soundfont": "fluidr3", "consent": true}), json!({"soundfont": ["generaluser-gs"]})] {
        assert!(s.execute("music.downloadSoundfont", p.clone()).is_err(), "{p}");
    }
    assert_eq!(s.jobs.len(), jobs, "nothing was downloaded");
}

#[test]
fn music_plays_on_a_users_soundfont_with_the_same_arrangement() {
    let dir = tmp("use");
    let font = tiny_font(&dir);
    let mut s = demo();
    let r = s.execute("music.useSoundfont", json!({"soundfont": font})).unwrap();
    assert_eq!(r["active"]["kind"], "soundfont");
    assert_eq!(r["active"]["name"], "FilmCraft Test Font");
    assert!(std::path::Path::new(r["selected"].as_str().unwrap()).is_absolute());
    assert_eq!(s.prefs.music.soundfont, r["selected"].as_str().unwrap());
    let l = s.execute("music.soundfonts", json!({})).unwrap();
    assert_eq!(l["active"]["name"], "FilmCraft Test Font");

    let params = json!({"kind": "music", "mood": "calm", "genre": "piano", "duration": 6.5, "seed": 4, "place": false, "dir": dir.to_string_lossy()});
    let a = s.execute("audio.generate", params.clone()).unwrap();
    assert!(a["instruments"].as_str().unwrap().starts_with("FilmCraft Test Font (SoundFont)"), "{a}");
    assert_eq!(a["soundfont"]["kind"], "soundfont");
    let item = filmcraft_project::ItemId(a["item"].as_u64().unwrap());
    assert!(s.project.item(item).unwrap().metadata.get(crate::audio_generate::INSTRUMENTS_KEY).is_some_and(|v| v.starts_with("FilmCraft Test Font")));
    // the same request on the oscillators: the same track, other instruments
    let mut q = params.clone();
    q["soundfont"] = json!("oscillators");
    let b = s.execute("audio.generate", q).unwrap();
    assert_eq!(b["instruments"], "oscillators");
    for k in ["sections", "beats", "downbeats", "ending", "tail", "samples", "chords", "bpm", "key"] {
        assert_eq!(a[k], b[k], "{k}");
    }
    let (wa, wb) = (std::fs::read(a["path"].as_str().unwrap()).unwrap(), std::fs::read(b["path"].as_str().unwrap()).unwrap());
    assert_eq!(wa.len(), wb.len());
    assert_ne!(wa, wb, "the font played");
    // back to the oscillators for good
    let r = s.execute("music.useSoundfont", json!({"soundfont": "oscillators"})).unwrap();
    assert_eq!((r["selected"].as_str(), r["active"]["kind"].as_str()), (Some("oscillators"), Some("oscillators")));
    let c = s.execute("audio.generate", params).unwrap();
    assert_eq!(c["instruments"], "oscillators");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn unusable_soundfonts_are_refused_and_the_choice_stays() {
    let dir = tmp("bad");
    let junk = dir.join("junk.sf2");
    std::fs::write(&junk, b"RIFF\x10\0\0\0sfbkLIST\x04\0\0\0INFO").unwrap();
    // a font whose sample chunk has an odd size (rustysynth would write past its buffer)
    let mut odd = synthetic::tiny();
    let at = odd.windows(4).position(|w| w == b"smpl").unwrap();
    let size = u32::from_le_bytes(odd[at + 4..at + 8].try_into().unwrap());
    odd[at + 4..at + 8].copy_from_slice(&(size - 1).to_le_bytes());
    let oddp = dir.join("odd.sf2");
    std::fs::write(&oddp, odd).unwrap();
    let mut s = demo();
    let before = s.prefs.music.soundfont.clone();
    for p in [
        json!({}),
        json!({"soundfont": null}),
        json!({"soundfont": ""}),
        json!({"soundfont": "   "}),
        json!({"soundfont": 7}),
        json!({"soundfont": {"path": "x"}}),
        json!({"soundfont": "a\u{0}b.sf2"}),
        json!({"soundfont": "x".repeat(5000)}),
        json!({"soundfont": "/definitely/not/here.sf2"}),
        json!({"soundfont": dir.to_string_lossy()}),
        json!({"soundfont": junk.to_string_lossy()}),
        json!({"soundfont": oddp.to_string_lossy()}),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("music.useSoundfont", p.clone())));
        assert!(r.expect("no panic").is_err(), "{p}");
        assert_eq!(s.prefs.music.soundfont, before, "{p}");
    }
    let e = s.execute("music.useSoundfont", json!({"soundfont": oddp.to_string_lossy()})).unwrap_err().to_string();
    assert!(e.contains("odd size"), "{e}");
    // per call, too
    let e = s.execute("audio.generate", json!({"kind": "music", "duration": 2, "soundfont": junk.to_string_lossy(), "place": false})).unwrap_err();
    assert!(e.to_string().contains("SoundFont"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_chosen_font_that_disappears_never_stops_the_music() {
    let dir = tmp("gone");
    let font = tiny_font(&dir);
    let mut s = demo();
    s.execute("music.useSoundfont", json!({"soundfont": font})).unwrap();
    std::fs::remove_file(&font).unwrap();
    // the preference's font is gone: the oscillators play it, and the answer says why
    let g = s.execute("audio.generate", json!({"kind": "music", "duration": 2, "seed": 1, "place": false, "dir": dir.to_string_lossy()})).unwrap();
    assert_eq!(g["instruments"], "oscillators");
    assert!(g["warning"].as_str().is_some_and(|w| w.contains("Preferences")), "{g}");
    let l = s.execute("music.soundfonts", json!({})).unwrap();
    assert!(l["active"]["error"].as_str().is_some(), "{l}");
    // named in the call, it is an error
    assert!(s.execute("audio.generate", json!({"kind": "music", "duration": 2, "place": false, "soundfont": font})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_preview_is_heard_then_placed_as_is() {
    let dir = tmp("preview");
    let font = tiny_font(&dir);
    let mut s = demo();
    let wav = dir.join("preview.wav");
    let p = json!({"mood": "uplifting", "genre": "pop", "duration": 5, "soundfont": font, "mute": ["drums"], "path": wav.to_string_lossy(), "wait": true});
    let r = s.execute("music.preview", p).unwrap();
    assert_eq!(r["ready"], true, "{r}");
    assert_eq!(r["instruments"]["name"], "FilmCraft Test Font");
    assert!(r["seed"].as_u64().is_some());
    assert_eq!(r["seconds"].as_f64(), Some(5.0));
    assert!(r["beats"].as_array().is_some_and(|b| b.len() > 4));
    assert!(std::fs::metadata(&wav).is_ok_and(|m| m.len() > 44 + 5 * 48_000 * 8 - 16));
    let heard = s.music_preview.latest().unwrap();
    assert_eq!(heard.track.mix.left.len(), 5 * 48_000);
    assert!(heard.request.levels[0].is_none(), "drums muted");
    // the same request again needs no new job
    let seed = r["seed"].clone();
    let jobs = s.jobs.len();
    let again =
        s.execute("music.preview", json!({"mood": "uplifting", "genre": "pop", "duration": 5, "soundfont": font, "mute": ["drums"], "seed": seed})).unwrap();
    assert_eq!((again["job"].clone(), again["ready"].clone()), (json!(null), json!(true)));
    assert_eq!(s.jobs.len(), jobs);
    // placing it with the same parameters and seed places what was heard
    let g = s
        .execute(
            "audio.generate",
            json!({"kind": "music", "mood": "uplifting", "genre": "pop", "duration": 5, "soundfont": font, "mute": ["drums"], "seed": seed, "dir": dir.to_string_lossy()}),
        )
        .unwrap();
    assert!(g["clip"].as_u64().is_some(), "{g}");
    let bytes = std::fs::read(g["path"].as_str().unwrap()).unwrap();
    let first: Vec<f32> = bytes[44..44 + 8 * 2000].chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    let heard_first: Vec<f32> = heard.track.mix.left.iter().zip(&heard.track.mix.right).take(2000).flat_map(|(l, r)| [*l, *r]).collect();
    assert_eq!(first, heard_first);
    // a job without wait finishes in the background and is then playable
    let r = s.execute("music.preview", json!({"mood": "dark", "duration": 3, "soundfont": "oscillators"})).unwrap();
    let job = r["job"].as_u64().unwrap();
    for _ in 0..600 {
        if s.music_preview.outcome(job).is_some() {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    assert!(s.music_preview.outcome(job).is_some_and(|o| o.is_ok()));
    assert_eq!(s.music_preview.latest().unwrap().track.plan.mood, filmcraft_audio_dsp::music::Mood::Dark);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn preview_refuses_hostile_parameters() {
    let mut s = demo();
    let jobs = s.jobs.len();
    for p in [
        json!({"kind": "whoosh"}),
        json!({"stems": true}),
        json!({"mood": "jazzy"}),
        json!({"genre": 3}),
        json!({"energy": [0.5, "x"]}),
        json!({"duration": -1}),
        json!({"duration": "long"}),
        json!({"levels": {"drums": "loud"}}),
        json!({"mute": ["drums", "bass", "melody", "pad"]}),
        json!({"soundfont": "/definitely/not/here.sf2"}),
        json!({"soundfont": 5}),
        json!({"path": 5}),
    ] {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| s.execute("music.preview", p.clone())));
        assert!(r.expect("no panic").is_err(), "{p}");
    }
    assert_eq!(s.jobs.len(), jobs, "no job started for a bad request");
}
