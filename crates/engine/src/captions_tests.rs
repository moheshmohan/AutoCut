use serde_json::json;

use crate::Session;
use filmcraft_project::ClipId;
use filmcraft_time::{TICKS_PER_SECOND, Tick};

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

#[test]
fn captions_import_edit_export_and_burn_in() {
    let dir = std::env::temp_dir().join(format!("fc-cap-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srt = dir.join("talk.srt");
    std::fs::write(
        &srt,
        "\u{feff}1\r\n00:00:00,500 --> 00:00:02,000\r\nHello there\r\n\r\n2\r\n00:00:02,500 --> 00:00:04,000\r\nSecond line\r\nwith two rows\r\n",
    )
    .unwrap();
    let mut s = demo();
    let r = s.execute("file.import", json!({"paths": [srt.to_string_lossy()]})).unwrap();
    assert_eq!(r["documents"][0]["captions"], 2, "{r}");
    let q = s.active_sequence().unwrap();
    let rate = q.settings.frame_rate;
    let tr = q.caption_tracks[0].clone();
    assert_eq!(tr.name, "talk");
    assert_eq!(tr.captions[0].start, rate.snap_nearest(Tick(TICKS_PER_SECOND / 2)), "snapped to frames");
    let first = tr.captions[0].id;
    let second = tr.captions[1].id;

    // edit text, split, merge, trim, move, add at playhead, delete — each one undoable
    s.execute("captions.setText", json!({"caption": first.0, "text": "Hi!", "speaker": "Ann"})).unwrap();
    let split_at = rate.tick_of(rate.frame_at(tr.captions[1].start) + 10);
    let r = s.execute("captions.split", json!({"caption": second.0, "time": split_at.0})).unwrap();
    let right = ClipId(r["captions"][0].as_u64().unwrap());
    assert_eq!(s.active_sequence().unwrap().caption_tracks[0].captions.len(), 3);
    s.execute("captions.merge", json!({"captions": [second.0, right.0]})).unwrap();
    assert_eq!(s.active_sequence().unwrap().caption_tracks[0].captions.len(), 2);
    s.execute("captions.trim", json!({"caption": second.0, "edge": "out", "deltaFrames": 12})).unwrap();
    s.execute("captions.move", json!({"captions": [second.0], "deltaFrames": 24})).unwrap();
    s.execute("playhead.set", json!({"seconds": 8.0})).unwrap();
    let r = s.execute("captions.add", json!({"text": "Added"})).unwrap();
    let added = ClipId(r["caption"].as_u64().unwrap());
    assert_eq!(s.state.caption_selection, vec![added]);
    s.execute("edit.clear", json!({})).unwrap();
    assert!(s.active_sequence().unwrap().find_caption(added).is_none(), "Clear deletes selected captions");
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.active_sequence().unwrap().find_caption(added).is_some());
    let list = s.execute("captions.list", json!({})).unwrap();
    assert_eq!(list["tracks"][0]["captions"][0]["text"], "Hi!");
    assert_eq!(list["tracks"][0]["captions"][0]["speaker"], "Ann");
    s.active_sequence().unwrap().check().unwrap();

    // export in four formats and read back: same texts
    for ext in ["srt", "vtt", "scc", "ass"] {
        let path = dir.join(format!("out.{ext}")).to_string_lossy().to_string();
        let r = s.execute("captions.export", json!({"path": path})).unwrap();
        assert_eq!(r["captions"], 3, "{ext}: {r}");
        let back = filmcraft_captions::parse(&std::fs::read(&path).unwrap(), filmcraft_captions::Format::from_name(ext).unwrap()).unwrap();
        let texts: Vec<String> = back.cues.iter().map(|c| filmcraft_project::plain_text(&c.text)).collect();
        assert_eq!(texts.len(), 3, "{ext}");
        assert_eq!(texts[0], "Hi!", "{ext}");
    }

    // the program renders captions; hiding the track removes them
    s.execute("playhead.set", json!({"seconds": 1.0})).unwrap();
    let with = s.render_program(0.25).unwrap();
    s.execute("captions.hideAll", json!({})).unwrap();
    let without = s.render_program(0.25).unwrap();
    let diff = with.px.iter().zip(&without.px).filter(|(a, b)| (*a - *b).abs() > 0.05).count();
    assert!(diff > 200, "captions change the picture ({diff} samples)");
    let half = with.w * 4 * (with.h / 2);
    let top_diff = with.px[..half].iter().zip(&without.px[..half]).filter(|(a, b)| (*a - *b).abs() > 0.05).count();
    assert_eq!(top_diff, 0, "bottom-anchored captions stay in the lower half");
    s.execute("captions.showAll", json!({})).unwrap();
    let plan = filmcraft_render::plan::plan_frame(
        &s.project,
        s.state.active_sequence.unwrap(),
        s.playhead(),
        filmcraft_render::RenderOptions { scale: 0.25, captions: true, ..Default::default() },
        &s.media.provider(s.project.clone(), s.services.clone()),
    );
    let cpu = filmcraft_render::plan::execute_cpu(&plan);
    let pd = cpu.px.iter().zip(&with.px).filter(|(a, b)| (*a - *b).abs() > 0.02).count();
    assert!(pd < cpu.px.len() / 200, "the GPU plan matches the CPU render ({pd})");

    // save/open keeps caption tracks
    let proj = dir.join("cap.fcproj").to_string_lossy().to_string();
    s.execute("file.saveAs", json!({"path": proj})).unwrap();
    let before = s.active_sequence().unwrap().caption_tracks.clone();
    s.execute("file.open", json!({"path": proj})).unwrap();
    assert_eq!(s.active_sequence().unwrap().caption_tracks, before);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn caption_track_follows_sync_locked_insert() {
    let mut s = demo();
    s.execute("captions.newTrack", json!({"format": "CEA-608"})).unwrap();
    s.execute("playhead.set", json!({"seconds": 4.0})).unwrap();
    s.execute("captions.add", json!({"text": "later"})).unwrap();
    let start = s.active_sequence().unwrap().caption_tracks[0].captions[0].start;
    let item = s.project.items.values().find(|i| i.as_media().is_some_and(|m| m.info.has_video())).unwrap().id;
    s.execute("source.open", json!({"item": item.0})).unwrap();
    s.execute("playhead.set", json!({"seconds": 0.0})).unwrap();
    s.execute("source.insert", json!({})).unwrap();
    let after = s.active_sequence().unwrap().caption_tracks[0].captions[0].start;
    assert!(after > start, "caption moved right with the insert");
    s.active_sequence().unwrap().check().unwrap();
}

#[test]
fn caption_commands_need_tracks_and_navigate() {
    let mut s = demo();
    assert!(s.execute("captions.split", json!({})).is_err(), "disabled without captions");
    for (sec, text) in [(1.0, "one"), (3.0, "two"), (6.0, "three")] {
        s.execute("playhead.set", json!({"seconds": sec})).unwrap();
        s.execute("captions.add", json!({"text": text, "durationSeconds": 1.0})).unwrap();
    }
    s.execute("playhead.set", json!({"seconds": 0.0})).unwrap();
    s.execute("captions.next", json!({})).unwrap();
    s.execute("captions.next", json!({})).unwrap();
    let id = s.state.caption_selection[0];
    assert_eq!(s.active_sequence().unwrap().find_caption(id).unwrap().1.text, "two");
    s.execute("captions.previous", json!({})).unwrap();
    let id = s.state.caption_selection[0];
    assert_eq!(s.active_sequence().unwrap().find_caption(id).unwrap().1.text, "one");
    s.execute("captions.setStyle", json!({"color": "#ffff00", "anchor": "top", "size": 40})).unwrap();
    let st = &s.active_sequence().unwrap().caption_tracks[0].style;
    assert_eq!(st.color, [255, 255, 0, 255]);
    assert!(s.execute("captions.setStyle", json!({"anchor": "sideways"})).is_err());
    s.execute("captions.setTimes", json!({"caption": id.0, "startSeconds": 0.5, "endSeconds": 2.5})).unwrap();
    assert!(s.execute("captions.setTimes", json!({"caption": id.0, "endSeconds": 3.5})).is_err(), "would overlap `two`");
    s.execute("captions.deleteTrack", json!({"track": "C1"})).unwrap();
    assert!(s.active_sequence().unwrap().caption_tracks.is_empty());
}

#[test]
fn caption_list_track_filter_never_falls_back_to_every_track() {
    let mut s = demo();
    s.execute("captions.newTrack", json!({})).unwrap();
    s.execute("captions.newTrack", json!({})).unwrap();
    let all = s.execute("captions.list", json!({})).unwrap();
    let ids: Vec<u64> = all["tracks"].as_array().unwrap().iter().map(|t| t["id"].as_u64().unwrap()).collect();
    assert_eq!(ids.len(), 2, "{all}");
    for (i, id) in ids.iter().enumerate() {
        for sel in [json!(id), json!(format!("C{}", i + 1))] {
            let r = s.execute("captions.list", json!({"track": sel})).unwrap();
            let got: Vec<u64> = r["tracks"].as_array().unwrap().iter().map(|t| t["id"].as_u64().unwrap()).collect();
            assert_eq!(got, vec![*id], "{sel}");
        }
    }
    let missing = ids.iter().max().unwrap() + 999;
    for sel in [json!(missing), json!("C3"), json!("C0"), json!("nope")] {
        assert!(s.execute("captions.list", json!({"track": sel})).is_err(), "{sel} names no caption track");
    }
}

#[test]
fn ass_import_sets_the_track_style_and_round_trips() {
    let dir = std::env::temp_dir().join(format!("fc-cap-ass-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ass = dir.join("anime.ass");
    std::fs::write(
        &ass,
        "\u{feff}[Script Info]\r\nScriptType: v4.00+\r\nPlayResX: 1920\r\nPlayResY: 1080\r\n\r\n[V4+ Styles]\r\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\r\nStyle: Default,Noto Sans,64,&H0000FFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,3,0,2,40,40,60,1\r\n\r\n[Events]\r\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\r\nDialogue: 0,0:00:00.50,0:00:02.00,Default,Ann,0,0,0,,Hello {\\i1}there{\\i0}\\Nfriend\r\nDialogue: 0,0:00:02.50,0:00:04.00,Default,,0,0,0,,{\\an8}On top\r\n",
    )
    .unwrap();
    // File ▸ Import detects it, like the other caption formats
    let mut s = demo();
    let undo_before = s.history.undo.len();
    let r = s.execute("file.import", json!({"paths": [ass.to_string_lossy()]})).unwrap();
    assert_eq!(r["documents"][0]["captions"], 2, "{r}");
    let tr = s.active_sequence().unwrap().caption_tracks[0].clone();
    assert_eq!(tr.name, "anime");
    assert_eq!((tr.style.font.as_str(), tr.style.size, tr.style.color, tr.style.background), ("Noto Sans", 64.0, [255, 255, 0, 255], false));
    assert_eq!(tr.captions[0].text, "Hello <i>there</i>\nfriend");
    assert_eq!(tr.captions[0].speaker.as_deref(), Some("Ann"));
    assert_eq!(tr.captions[1].settings, "line:5.56% align:center");
    // one undo step takes it back
    assert_eq!(s.history.undo.len(), undo_before + 1);
    s.execute("edit.undo", json!({})).unwrap();
    assert!(s.active_sequence().unwrap().caption_tracks.is_empty());
    // captions.import too, then export as ASS and read it back
    s.execute("captions.import", json!({"path": ass.to_string_lossy()})).unwrap();
    let out = dir.join("out.ass").to_string_lossy().to_string();
    let r = s.execute("captions.export", json!({"path": out})).unwrap();
    assert_eq!(r["captions"], 2, "{r}");
    let back = filmcraft_captions::parse(&std::fs::read(&out).unwrap(), filmcraft_captions::Format::Ass).unwrap();
    assert_eq!(back.cues[0].text, "Hello <i>there</i>\nfriend");
    assert_eq!(back.cues[1].settings, "line:5.56% align:center");
    assert_eq!(back.style.map(|st| (st.font, st.size)), Some(("Noto Sans".to_string(), 64.0)));
    // a file that is not a script is refused with an error, never a panic
    let bad = dir.join("bad.ass");
    std::fs::write(&bad, "this is not a script").unwrap();
    assert!(s.execute("captions.import", json!({"path": bad.to_string_lossy()})).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

fn caption_spans(s: &Session) -> Vec<(f64, f64, String)> {
    let secs = |t: Tick| (t.seconds() * 10.0).round() / 10.0; // the demo sequence snaps to 23.976 fps frames
    s.active_sequence().unwrap().caption_tracks[0].captions.iter().map(|c| (secs(c.start), secs(c.end()), c.text.clone())).collect()
}

#[test]
fn captions_duplicate_copies_and_overwrites() {
    let mut s = demo();
    for (sec, text) in [(1.0, "one"), (3.0, "two")] {
        s.execute("playhead.set", json!({"seconds": sec})).unwrap();
        s.execute("captions.add", json!({"text": text, "durationSeconds": 1.0})).unwrap();
    }
    let one = s.active_sequence().unwrap().caption_tracks[0].captions[0].id;
    // Option-drag: a copy 5 s later, the original stays; the copy is selected
    let r = s.execute("captions.duplicate", json!({"captions": [one.0], "delta": 5 * TICKS_PER_SECOND})).unwrap();
    assert_eq!(caption_spans(&s), vec![(1.0, 2.0, "one".into()), (3.0, 4.0, "two".into()), (6.0, 7.0, "one".into())]);
    assert_eq!(s.state.caption_selection.len(), 1);
    assert_eq!(r["captions"][0].as_u64(), Some(s.state.caption_selection[0].0));
    // a copy dropped half over `two` overwrites that half
    s.execute("captions.duplicate", json!({"captions": [one.0], "delta": TICKS_PER_SECOND * 3 / 2})).unwrap();
    assert_eq!(caption_spans(&s)[1..3], [(2.5, 3.5, "one".into()), (3.5, 4.0, "two".into())]);
    // never before zero; zero delta is nothing
    s.execute("captions.duplicate", json!({"captions": [one.0], "delta": -100 * TICKS_PER_SECOND})).unwrap();
    assert_eq!(caption_spans(&s)[0], (0.0, 1.0, "one".into()));
    assert!(s.execute("captions.duplicate", json!({"captions": [one.0], "delta": 0})).is_err());
    assert!(s.execute("captions.duplicate", json!({"captions": [one.0]})).is_err(), "needs a delta");
    assert!(s.execute("captions.duplicate", json!({"captions": [], "delta": TICKS_PER_SECOND})).is_err(), "needs a caption");
    let before = s.active_sequence().unwrap().caption_tracks[0].captions.len();
    s.execute("captions.duplicate", json!({"captions": [one.0, one.0], "delta": 20 * TICKS_PER_SECOND})).unwrap();
    assert_eq!(s.active_sequence().unwrap().caption_tracks[0].captions.len(), before + 1, "a caption named twice is copied once");
    s.active_sequence().unwrap().check().unwrap();
}

#[test]
fn captions_copy_cut_and_paste() {
    let mut s = demo();
    // Copy / Paste with only captions selected copies captions, pasted at the playhead
    s.execute("playhead.set", json!({"seconds": 1.0})).unwrap();
    s.execute("captions.add", json!({"text": "hi", "durationSeconds": 1.0})).unwrap();
    assert!(s.state.selection.is_empty());
    assert_eq!(s.execute("edit.copy", json!({})).unwrap()["copied"], 1);
    s.execute("playhead.set", json!({"seconds": 10.0})).unwrap();
    s.execute("edit.paste", json!({})).unwrap();
    assert_eq!(caption_spans(&s), vec![(1.0, 2.0, "hi".into()), (10.0, 11.0, "hi".into())]);
    assert!((s.playhead().seconds() - 11.0).abs() < 0.05, "the playhead moves past the paste");
    // Cut takes it away and Paste puts it back
    s.execute("edit.cut", json!({})).unwrap();
    assert_eq!(caption_spans(&s), vec![(1.0, 2.0, "hi".into())]);
    s.execute("playhead.set", json!({"seconds": 4.0})).unwrap();
    s.execute("edit.paste", json!({})).unwrap();
    assert_eq!(caption_spans(&s)[1], (4.0, 5.0, "hi".into()));
    // copying clips afterwards makes Paste place clips again, not captions
    let clip = s.active_sequence().unwrap().video_tracks[0].items[0].id;
    s.execute("timeline.select", json!({"clips": [clip.0]})).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    assert!(s.state.caption_clipboard.is_empty() && !s.state.clipboard.is_empty());
    // undo takes back a caption paste as one step
    let mut s = demo();
    s.execute("captions.add", json!({"text": "x", "durationSeconds": 1.0})).unwrap();
    s.execute("edit.copy", json!({})).unwrap();
    s.execute("playhead.set", json!({"seconds": 5.0})).unwrap();
    s.execute("edit.paste", json!({})).unwrap();
    s.execute("edit.undo", json!({})).unwrap();
    assert_eq!(caption_spans(&s).len(), 1);
}
/// The caption track language reaches TTML, and import can set it (#4025).
#[test]
fn caption_track_language_survives_import_and_ttml_export() {
    let dir = std::env::temp_dir().join(format!("fc-cap-lang-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srt = dir.join("pt.srt");
    std::fs::write(&srt, "1\n00:00:01,000 --> 00:00:02,000\nOlá\n").unwrap();
    let ttml = dir.join("out.ttml").to_string_lossy().to_string();
    let mut s = demo();
    let r = s.execute("captions.import", json!({"path": srt.to_string_lossy()})).unwrap();
    s.execute("captions.setTrack", json!({"track": "C1", "language": "pt-BR"})).unwrap();
    s.execute("captions.export", json!({"path": ttml, "track": "C1"})).unwrap();
    let xml = std::fs::read_to_string(&ttml).unwrap();
    assert!(xml.contains(r#"xml:lang="pt-BR""#), "{xml}");
    assert_eq!(r["language"], "en", "an SRT says nothing of its language: {r}");
    // import takes a language, and a TTML file brings its own
    let r = s.execute("captions.import", json!({"path": srt.to_string_lossy(), "language": "es-419"})).unwrap();
    assert_eq!(r["language"], "es-419");
    let r = s.execute("captions.import", json!({"path": ttml})).unwrap();
    assert_eq!(r["language"], "pt-BR");
    let list = s.execute("captions.list", json!({})).unwrap();
    let langs: Vec<&str> = list["tracks"].as_array().unwrap().iter().map(|t| t["language"].as_str().unwrap()).collect();
    assert_eq!(langs, ["pt-BR", "es-419", "pt-BR"], "new tracks go on top: {list}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// SCC and MCC exports count their time codes from the sequence's start timecode (its label, so
/// 01:00:00:00 at 25 fps is 01:00:00;00 or 01:00:00:00 at 29.97), and importing such a file into
/// the sequence puts the captions back where they were.
#[test]
fn scc_and_mcc_time_codes_count_from_the_sequence_start_timecode() {
    let dir = std::env::temp_dir().join(format!("fc-cap-tc-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let srt = dir.join("tc.srt");
    std::fs::write(&srt, "1\n00:00:02,000 --> 00:00:04,000\nFirst\n\n2\n00:00:05,000 --> 00:00:06,000\nSecond\n").unwrap();
    let mut s = Session::default();
    s.execute("file.newSequence", json!({"name": "TC", "fps": 25, "width": 16, "height": 16, "video": 1, "audio": 0})).unwrap();
    s.execute("captions.import", json!({"path": srt.to_string_lossy()})).unwrap();
    let starts = |s: &Session| -> Vec<Tick> { s.active_sequence().unwrap().caption_tracks[0].captions.iter().map(|c| c.start).collect() };
    let original = starts(&s);
    let first_label = |path: &str| -> String {
        let text = std::fs::read_to_string(path).unwrap();
        text.lines().find_map(|l| l.split_once('\t')).unwrap().0.to_string()
    };
    for (start_timecode, hour) in [(90_000, "01:00:0"), (0, "00:00:0")] {
        s.edit_sequence("Start Time", |q, _, _| {
            q.start_timecode = start_timecode;
            Ok(())
        })
        .unwrap();
        for ext in ["scc", "mcc"] {
            for drop_frame in [true, false] {
                let path = dir.join(format!("tc-{start_timecode}-{drop_frame}.{ext}")).to_string_lossy().to_string();
                s.execute("captions.export", json!({"path": path, "track": "C1", "dropFrame": drop_frame})).unwrap();
                let label = first_label(&path);
                assert!(label.starts_with(hour), "{ext} drop-frame {drop_frame}: {label}");
                assert_eq!(label.contains(';'), drop_frame, "{label}");
                let r = s.execute("captions.import", json!({"path": path})).unwrap();
                assert_eq!(r["fromStartTimecode"], start_timecode > 0, "{r}");
                assert_eq!(starts(&s), original, "{ext} drop-frame {drop_frame} from {start_timecode}");
                s.execute("edit.undo", json!({})).unwrap();
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}
