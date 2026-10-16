//! `~/…` in a project path is the home folder, as in a shell: Save As, Save and Open used to take
//! it literally, writing into a folder named `~` under the working directory.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use serde_json::json;

use crate::{Services, Session};

/// Files only through `Services`, so the test never touches the real home folder.
#[derive(Default)]
struct MemFs(Mutex<BTreeMap<String, Vec<u8>>>);

impl Services for MemFs {
    fn read_file(&self, path: &str) -> std::io::Result<Vec<u8>> {
        self.0.lock().unwrap().get(path).cloned().ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, path.to_string()))
    }
    fn write_file(&self, path: &str, data: &[u8]) -> std::io::Result<()> {
        self.0.lock().unwrap().insert(path.into(), data.to_vec());
        Ok(())
    }
}

#[test]
fn project_paths_starting_with_a_tilde_are_in_the_home_folder() {
    let Some(home) = crate::media_browser::std_home_dir() else { return };
    let fs = Arc::new(MemFs::default());
    let mut s = Session::new(fs.clone());
    s.execute("file.openDemoProject", json!({})).unwrap();
    s.execute("file.saveAs", json!({"path": "~/Projects/Home Test.fcproj"})).unwrap();
    let expected = format!("{home}/Projects/Home Test.fcproj");
    let written: Vec<String> = fs.0.lock().unwrap().keys().cloned().collect();
    assert!(written.contains(&expected), "{written:?}");
    assert!(!written.iter().any(|p| p.starts_with('~')), "nothing under a literal `~`: {written:?}");
    assert_eq!(s.path.as_deref(), Some(expected.as_str()), "the project adopts the real path");
    // Save writes there again, and Open reads it back through the same spelling
    s.execute("file.save", json!({})).unwrap();
    let mut t = Session::new(fs);
    t.execute("file.open", json!({"path": "~/Projects/Home Test.fcproj"})).unwrap();
    assert_eq!(t.path.as_deref(), Some(expected.as_str()));
    assert_eq!(crate::export_tools::expand_home("~"), home);
    assert_eq!(crate::export_tools::expand_home("/abs/~/x"), "/abs/~/x", "only a leading tilde");
}
