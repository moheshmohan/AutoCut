//! Preferences, auto-save ring and crash recovery through the command interface.

use super::*;
use serde_json::json;

fn demo() -> Session {
    let mut s = Session::default();
    s.execute("file.openDemoProject", json!({})).unwrap();
    s
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("fc-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn wait_for(what: &str, mut f: impl FnMut() -> bool) {
    let t0 = std::time::Instant::now();
    while !f() {
        assert!(t0.elapsed() < std::time::Duration::from_secs(10), "timed out waiting for {what}");
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

fn start(s: &mut Session, data: &std::path::Path) {
    let mut cfg = autosave::AutosaveConfig::new(data);
    cfg.journal_debounce = std::time::Duration::from_millis(20);
    s.start_autosave(cfg).unwrap();
}

fn journaled(s: &mut Session) -> u64 {
    s.poll_persistence();
    s.persistence.as_ref().unwrap().status.journaled_revision
}

#[test]
fn preferences_get_set_and_persist() {
    let d = temp_dir("prefs");
    let mut s = Session::default();
    start(&mut s, &d);
    assert_eq!(s.execute("prefs.get", json!({"key": "autoSave.intervalMinutes"})).unwrap(), json!(5));
    assert_eq!(s.execute("prefs.get", json!({"key": "autoSave.maxVersions"})).unwrap(), json!(20));
    s.execute("prefs.set", json!({"key": "autoSave.intervalMinutes", "value": 12.0})).unwrap();
    s.execute("prefs.set", json!({"values": {"autoSave.maxVersions": 5000, "autoSave.saveCurrentProject": true}})).unwrap();
    assert_eq!(s.prefs.auto_save.interval_minutes, 12);
    assert_eq!(s.prefs.auto_save.max_versions, 1000, "clamped");
    assert!(s.execute("prefs.set", json!({"key": "autoSave.nope", "value": 1})).is_err());
    assert!(s.execute("prefs.set", json!({"key": "autoSave.enabled", "value": "yes"})).is_err());
    let keys = s.execute("prefs.get", json!({})).unwrap()["keys"].clone();
    assert!(keys.as_array().unwrap().iter().any(|k| k == "autoSave.recoveryJournal"));
    s.shutdown();
    let mut t = Session::default();
    start(&mut t, &d);
    assert_eq!(t.prefs.auto_save.interval_minutes, 12);
    assert!(t.prefs.auto_save.save_current_project);
    t.shutdown();
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn crash_recovery_end_to_end() {
    let d = temp_dir("recover");
    let proj = d.join("film.fcproj").to_string_lossy().to_string();
    // Session A: saved project, then unsaved edits, then a crash.
    let mut a = demo();
    start(&mut a, &d);
    a.execute("file.saveAs", json!({"path": proj})).unwrap();
    a.execute("sequence.addEdit", json!({"seconds": 1.5})).unwrap();
    a.execute("sequence.addEdit", json!({"seconds": 2.5})).unwrap();
    let rev = a.revision;
    wait_for("journal", || journaled(&mut a) == rev);
    let st = a.execute("file.autoSaveStatus", json!({})).unwrap();
    assert!(st["status"]["lastBytes"].as_u64().unwrap() > 1000, "{st}");
    let edited = (*a.project).clone();
    // Session B starts while A is alive: A's journal is not stale.
    let mut b = Session::default();
    start(&mut b, &d);
    assert!(b.recovery_candidates().is_empty(), "a live session's journal is never offered");
    b.shutdown();
    a.persistence.take().unwrap().simulate_crash();

    // Session C: next launch.
    let mut c = Session::default();
    start(&mut c, &d);
    assert_eq!(c.recovery_candidates().len(), 1);
    let list = c.execute("file.recoveryList", json!({})).unwrap();
    assert_eq!(list[0]["projectName"], "film", "named after the file it was saved as");
    assert_eq!(list[0]["projectPath"], proj.as_str());
    assert_eq!(list[0]["cleanExit"], false);
    assert_eq!(list[0]["savedAt"].as_str().unwrap().len(), 19);
    // nothing saved over the project since: Recover is the primary choice, as always
    assert_eq!(list[0]["projectFile"], "unchanged", "{list}");
    assert_eq!(list[0]["primaryAction"], "recover");
    c.execute("file.recover", json!({})).unwrap();
    assert_eq!(*c.project, edited);
    assert!(c.is_dirty(), "recovered changes are unsaved");
    assert_eq!(c.path.as_deref(), Some(proj.as_str()));
    assert!(c.recovery_candidates().is_empty());
    assert!(!c.is_enabled("file.recover"));
    // The recovered state is already in C's own journal (a crash now loses nothing).
    assert_eq!(journaled(&mut c), c.revision);
    c.execute("file.save", json!({})).unwrap();
    wait_for("journal cleared", || {
        c.poll_persistence();
        !c.persistence.as_ref().unwrap().session_dir.join("snapshot.fcproj").exists()
    });
    let session_dir = c.persistence.as_ref().unwrap().session_dir.clone();
    c.shutdown();
    assert!(!session_dir.exists(), "clean exit with nothing unsaved removes the journal");
    let mut e = Session::default();
    start(&mut e, &d);
    assert!(e.recovery_candidates().is_empty());
    e.shutdown();
    let _ = std::fs::remove_dir_all(&d);
}

fn set_mtime(path: &str, t: std::time::SystemTime) {
    std::fs::File::options().write(true).open(path).unwrap().set_modified(t).unwrap();
}

/// 2026-10-16: the app was killed with unsaved changes (snapshot 06:55), then the headless CLI
/// rebuilt and saved the same project (09:50). The next launch offered Recover as the primary
/// choice, which would have replaced the newer saved project with the older snapshot.
#[test]
fn recovery_never_offers_an_auto_save_over_a_newer_saved_project() {
    let d = temp_dir("recover-newer");
    let proj = d.join("Robot Showcase.fcproj").to_string_lossy().to_string();
    let mut a = demo();
    start(&mut a, &d);
    a.execute("file.saveAs", json!({"path": proj})).unwrap();
    a.execute("sequence.addEdit", json!({"seconds": 1.5})).unwrap();
    let rev = a.revision;
    wait_for("journal", || journaled(&mut a) == rev);
    a.persistence.take().unwrap().simulate_crash();

    // Next launch, before anything else touched the file: Recover, as always.
    let mut c = Session::default();
    start(&mut c, &d);
    let list = c.execute("file.recoveryList", json!({})).unwrap();
    assert_eq!((list[0]["projectFile"].as_str(), list[0]["primaryAction"].as_str()), (Some("unchanged"), Some("recover")), "{list}");

    // Another program (the CLI) opens the saved project, changes it and saves it, with a clock
    // that is behind: the file looks older than the snapshot, but its contents changed.
    let mut cli = Session::default();
    cli.execute("file.open", json!({"path": proj})).unwrap();
    cli.execute("sequence.addEdit", json!({"seconds": 3.0})).unwrap();
    cli.execute("file.save", json!({})).unwrap();
    set_mtime(&proj, std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000));
    let saved = std::fs::read(&proj).unwrap();

    // The open dialog's list is checked again; so is the next launch's.
    let list = c.execute("file.recoveryList", json!({})).unwrap();
    assert_eq!(list[0]["projectFile"], "newer", "{list}");
    assert_eq!(list[0]["primaryAction"], "openSaved", "{list}");
    c.shutdown();
    let mut c = Session::default();
    start(&mut c, &d);
    let list = c.execute("file.recoveryList", json!({})).unwrap();
    assert_eq!(list[0]["projectFile"], "newer", "{list}");
    assert_ne!(list[0]["primaryAction"], "recover", "{list}");
    assert!(list[0]["projectModifiedAt"].is_string(), "{list}");

    // Keeping the saved project opens it and keeps the snapshot for later.
    c.execute("file.open", json!({"path": proj})).unwrap();
    assert_eq!(*c.project, *cli.project);
    assert_eq!(c.recovery_candidates().len(), 1, "recovery data is never deleted automatically");
    assert!(c.is_enabled("file.recover"));
    c.shutdown();

    // Recover Anyway still works and leaves the file alone until the project is saved.
    let mut e = Session::default();
    start(&mut e, &d);
    let r = e.execute("file.recover", json!({})).unwrap();
    assert_eq!(r["projectFile"], "newer");
    assert_eq!(e.path.as_deref(), Some(proj.as_str()));
    assert_eq!(std::fs::read(&proj).unwrap(), saved, "recovering writes nothing over the file");
    // A crash now still warns: the recovered changes build on the older file, not the newer one.
    let rev = e.revision;
    wait_for("journal", || journaled(&mut e) == rev);
    e.persistence.take().unwrap().simulate_crash();
    let mut f = Session::default();
    start(&mut f, &d);
    let list = f.execute("file.recoveryList", json!({})).unwrap();
    assert_eq!(list.as_array().unwrap().len(), 1);
    assert_eq!(list[0]["projectFile"], "newer", "{list}");
    f.execute("file.discardRecovery", json!({"all": true})).unwrap();
    f.shutdown();
    let _ = std::fs::remove_dir_all(&d);
}

/// "Auto Save also saves the current project" writes the file from the worker: unsaved changes
/// made after that build on the new file, so a crash then is not mistaken for a newer save.
#[test]
fn auto_saving_the_project_file_moves_the_recovery_base() {
    let d = temp_dir("recover-base");
    let proj = d.join("base.fcproj").to_string_lossy().to_string();
    let mut a = demo();
    start(&mut a, &d);
    a.execute("prefs.set", json!({"key": "autoSave.saveCurrentProject", "value": true})).unwrap();
    a.execute("file.saveAs", json!({"path": proj})).unwrap();
    a.execute("sequence.addEdit", json!({"seconds": 1.0})).unwrap();
    a.execute("file.autoSaveNow", json!({})).unwrap();
    wait_for("project saved", || {
        a.poll_persistence();
        !a.is_dirty()
    });
    let written = std::fs::read(&proj).unwrap();
    assert_eq!(a.based_on, Some((proj.clone(), autosave::FileStamp::of_bytes(&written))));
    a.execute("sequence.addEdit", json!({"seconds": 2.0})).unwrap();
    let rev = a.revision;
    wait_for("journal", || journaled(&mut a) == rev);
    a.persistence.take().unwrap().simulate_crash();
    let mut b = Session::default();
    start(&mut b, &d);
    let list = b.execute("file.recoveryList", json!({})).unwrap();
    assert_eq!(list[0]["projectFile"], "unchanged", "{list}");
    assert_eq!(list[0]["primaryAction"], "recover");
    b.execute("file.discardRecovery", json!({"all": true})).unwrap();
    b.shutdown();
    let _ = std::fs::remove_dir_all(&d);
}

/// The project-file check under hostile journals and file systems: never a crash, never Recover
/// as the primary choice unless the file is known not to hold newer work.
#[test]
fn recovery_check_survives_hostile_inputs_and_distrusts_clocks() {
    use autosave::{FileStamp, JournalMeta, ProjectFileState as S, RecoveryAction as A, check_project_file};
    let d = temp_dir("recover-hostile");
    let file = d.join("p.fcproj");
    let path = file.to_string_lossy().to_string();
    std::fs::write(&file, b"saved project bytes").unwrap();
    let now = autosave::unix_now();
    let meta = |saved_unix: i64, based_on: Option<FileStamp>| JournalMeta { saved_unix, project_path: Some(path.clone()), based_on, ..Default::default() };
    let check = |m: &JournalMeta| {
        let (state, _) = check_project_file(m, now);
        (state, state.primary_action())
    };
    let stamp = FileStamp::of_bytes(b"saved project bytes");
    assert_eq!(FileStamp::of_file(&file).unwrap(), stamp);

    // fingerprints: clock-free either way
    set_mtime(&path, std::time::SystemTime::now() + std::time::Duration::from_secs(10 * 365 * 86_400));
    assert_eq!(check(&meta(now, Some(stamp.clone()))), (S::Unchanged, A::Recover), "touched, same bytes");
    assert_eq!(check(&meta(i64::MAX, Some(stamp.clone()))), (S::Unchanged, A::Recover), "snapshot clock in the future");
    let other = FileStamp::of_bytes(b"SAVED PROJECT BYTES");
    assert_eq!(other.bytes, stamp.bytes);
    assert_eq!(check(&meta(now, Some(other))), (S::Newer, A::OpenSaved), "same size, other bytes");
    let hostile = FileStamp { bytes: u64::MAX, hash: "not hex \u{1F4A5}".into() };
    assert_eq!(check(&meta(now, Some(hostile))), (S::Newer, A::OpenSaved));
    let hostile = FileStamp { bytes: stamp.bytes, hash: String::new() };
    assert_eq!(check(&meta(now, Some(hostile))), (S::Newer, A::OpenSaved));

    // journals without a fingerprint: the clocks, distrusted when the snapshot's is odd
    set_mtime(&path, std::time::UNIX_EPOCH + std::time::Duration::from_secs(1_000_000));
    assert_eq!(check(&meta(2_000_000, None)), (S::Unchanged, A::Recover), "saved before the snapshot");
    assert_eq!(check(&meta(1_000_000, None)), (S::Unknown, A::OpenSaved), "the same second");
    assert_eq!(check(&meta(999_999, None)), (S::Newer, A::OpenSaved), "saved after the snapshot");
    for odd in [0, -1, i64::MIN, now + 3600, i64::MAX] {
        assert_eq!(check(&meta(odd, None)), (S::Unknown, A::OpenSaved), "snapshot time {odd}");
    }

    // no file, not a file, no path
    let gone = JournalMeta { project_path: Some(d.join("moved away.fcproj").to_string_lossy().to_string()), ..meta(now, Some(stamp.clone())) };
    assert_eq!(check(&gone), (S::Missing, A::Later));
    let dir = JournalMeta { project_path: Some(d.to_string_lossy().to_string()), ..meta(now, Some(stamp.clone())) };
    assert_eq!(check(&dir), (S::Unknown, A::OpenSaved));
    assert!(FileStamp::of_file(&d).is_err());
    for p in [None, Some(String::new())] {
        assert_eq!(check(&JournalMeta { project_path: p, ..meta(now, None) }), (S::Untitled, A::Recover));
    }
    let nul = JournalMeta { project_path: Some("bad\0path.fcproj".into()), ..meta(now, Some(stamp.clone())) };
    assert_ne!(check(&nul).1, A::Recover);

    // through the scan: hand-written journals as crashed sessions leave them
    let journal = |id: &str, meta: serde_json::Value| {
        let sess = autosave::recovery_root(&d).join(id);
        std::fs::create_dir_all(&sess).unwrap();
        std::fs::write(sess.join("snapshot.fcproj"), filmcraft_format::encode(&filmcraft_project::Project::default(), false)).unwrap();
        std::fs::write(sess.join("session.json"), serde_json::to_vec(&meta).unwrap()).unwrap();
    };
    journal("1-1", json!({"savedUnix": now, "projectPath": d.join("gone.fcproj"), "basedOn": stamp}));
    journal("1-2", json!({"savedUnix": i64::MAX, "projectPath": path}));
    journal("1-3", json!({"savedUnix": now, "projectPath": path, "basedOn": {"bytes": -5, "hash": 7}}));
    let found = autosave::scan_candidates(&d, None);
    let state = |id: &str| found.iter().find(|c| c.id == id).map(|c| (c.project_file, c.primary_action()));
    assert_eq!(state("1-1"), Some((S::Missing, A::Later)));
    assert_eq!(state("1-2"), Some((S::Unknown, A::OpenSaved)));
    // an unreadable journal still lists its snapshot (#894), without a path: nothing on disk to replace
    assert_eq!(state("1-3"), Some((S::Untitled, A::Recover)));
    for id in ["1-1", "1-2", "1-3"] {
        assert!(autosave::recovery_root(&d).join(id).join("snapshot.fcproj").is_file(), "{id}: checking never deletes");
    }
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn quitting_with_unsaved_changes_keeps_them_for_next_launch() {
    let d = temp_dir("quitdirty");
    let mut a = demo();
    start(&mut a, &d);
    a.execute("sequence.addEdit", json!({"seconds": 1.0})).unwrap();
    a.shutdown();
    let mut b = Session::default();
    start(&mut b, &d);
    let list = b.execute("file.recoveryList", json!({})).unwrap();
    assert_eq!(list.as_array().unwrap().len(), 1);
    assert_eq!(list[0]["cleanExit"], true);
    b.execute("file.discardRecovery", json!({})).unwrap();
    assert!(b.recovery_candidates().is_empty());
    b.shutdown();
    let mut c = Session::default();
    start(&mut c, &d);
    assert!(c.recovery_candidates().is_empty(), "discarded for good");
    c.shutdown();
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn auto_save_ring_rotates_and_can_save_the_project() {
    let d = temp_dir("ring");
    let proj = d.join("ring.fcproj").to_string_lossy().to_string();
    let mut s = demo();
    start(&mut s, &d);
    s.execute("prefs.set", json!({"values": {"autoSave.maxVersions": 2}})).unwrap();
    s.execute("file.saveAs", json!({"path": proj})).unwrap();
    // Nothing unsaved: no auto-save.
    assert!(s.execute("file.autoSaveNow", json!({})).unwrap()["path"].is_null());
    let mut written = Vec::new();
    for i in 0..3 {
        s.execute("sequence.addEdit", json!({"seconds": 1.0 + i as f64})).unwrap();
        let r = s.execute("file.autoSaveNow", json!({})).unwrap();
        written.push(r["path"].as_str().unwrap().to_string());
        // File names have one-second resolution.
        std::thread::sleep(std::time::Duration::from_millis(1100));
    }
    let sep = std::path::MAIN_SEPARATOR;
    assert!(written[0].contains(&format!("{sep}Auto-Save{sep}ring-")), "{}", written[0]);
    let list = s.execute("file.listAutoSaves", json!({})).unwrap();
    let files: Vec<String> = list["files"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    assert_eq!(files, vec![written[2].clone(), written[1].clone()], "newest first, oldest pruned");
    let mut t = Session::default();
    t.execute("file.open", json!({"path": written[2]})).unwrap();
    assert_eq!(*t.project, *s.project);
    assert!(s.is_dirty(), "auto-save does not save the project itself by default");
    s.execute("prefs.set", json!({"key": "autoSave.saveCurrentProject", "value": true})).unwrap();
    s.execute("sequence.addEdit", json!({"seconds": 5.0})).unwrap();
    s.execute("file.autoSaveNow", json!({})).unwrap();
    wait_for("project saved", || {
        s.poll_persistence();
        !s.is_dirty()
    });
    let mut t = Session::default();
    t.execute("file.open", json!({"path": proj})).unwrap();
    assert_eq!(*t.project, *s.project);
    s.shutdown();
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn scan_keeps_snapshot_when_session_metadata_is_unreadable() {
    // #894: a stale session with a snapshot must not be deleted because session.json is missing
    // or corrupt — the recovered project would vanish from the list and the file would be gone.
    let d = temp_dir("recover-nometa");
    let root = autosave::recovery_root(&d);
    let sess = root.join("123-1");
    std::fs::create_dir_all(&sess).unwrap();
    let snap = filmcraft_format::encode(&filmcraft_project::Project::default(), false);
    std::fs::write(sess.join("snapshot.fcproj"), snap).unwrap();
    std::fs::write(sess.join("session.json"), b"not json {").unwrap();
    let found = autosave::scan_candidates(&d, None);
    assert_eq!(found.len(), 1);
    assert_eq!(found[0].id, "123-1");
    assert!(sess.join("snapshot.fcproj").is_file(), "snapshot must survive the scan");
    std::fs::remove_file(sess.join("session.json")).unwrap();
    let found = autosave::scan_candidates(&d, None);
    assert_eq!(found.len(), 1, "missing metadata still lists the snapshot");
    assert!(sess.join("snapshot.fcproj").is_file());
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn portable_marker_keeps_data_next_to_the_executable() {
    // #139: a portable install must not write to %APPDATA%.
    let d = temp_dir("portable");
    assert_eq!(autosave::portable_data_dir(&d), None);
    std::fs::create_dir_all(d.join(autosave::PORTABLE_MARKER)).unwrap();
    assert_eq!(autosave::portable_data_dir(&d), None, "a directory is not the marker");
    std::fs::remove_dir_all(d.join(autosave::PORTABLE_MARKER)).unwrap();
    std::fs::write(d.join(autosave::PORTABLE_MARKER), "").unwrap();
    assert_eq!(autosave::portable_data_dir(&d), Some(d.join("data")));
    let _ = std::fs::remove_dir_all(&d);
}

#[test]
fn journal_costs_the_ui_thread_only_a_handoff() {
    // Per change the UI thread pays an Arc clone and a channel send; encoding happens on the worker.
    let d = temp_dir("cost");
    let mut s = demo();
    start(&mut s, &d);
    let t0 = std::time::Instant::now();
    for i in 0..50 {
        s.execute("playhead.set", json!({"seconds": i as f64 * 0.1})).unwrap();
        let _ = s.execute("sequence.addEdit", json!({}));
    }
    let per_cmd = t0.elapsed() / 100;
    let rev = s.revision;
    wait_for("journal", || journaled(&mut s) == rev);
    assert!(per_cmd < std::time::Duration::from_millis(50), "{per_cmd:?}");
    s.shutdown();
    let _ = std::fs::remove_dir_all(&d);
}
