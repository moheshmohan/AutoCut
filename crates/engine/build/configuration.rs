//! Build inputs that change renderer semantics; unrelated OS environment need not be Unicode.
use std::{ffi::OsString, io};

pub fn selected(values: impl IntoIterator<Item = (OsString, OsString)>) -> io::Result<Vec<(String, String)>> {
    let mut result = Vec::new();
    for (key, value) in values {
        let Some(key) = key.to_str() else { continue };
        if key.starts_with("CARGO_FEATURE_")
            || ["TARGET", "PROFILE", "OPT_LEVEL", "DEBUG", "CARGO_CFG_TARGET_FEATURE", "CARGO_ENCODED_RUSTFLAGS"].contains(&key)
        {
            let value = value.into_string().map_err(|_| io::Error::new(io::ErrorKind::InvalidData, format!("non-Unicode build configuration: {key}")))?;
            result.push((key.to_string(), value));
        }
    }
    result.sort();
    Ok(result)
}

/// A crate manifest's package name and the `filmcraft-*` crates in its dependency tables (build
/// and target ones too; dev-dependencies are not part of the built crate).
pub fn manifest(text: &str) -> Option<(String, Vec<String>)> {
    let mut section = "";
    let mut name = None;
    let mut dependencies = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.starts_with('[') {
            section = line.trim_matches(|c| c == '[' || c == ']');
            continue;
        }
        let Some((key, value)) = line.split_once('=') else { continue };
        let key = key.trim();
        if section == "package" && key == "name" {
            name = Some(value.trim().trim_matches('"').to_string());
        } else if section.ends_with("dependencies") && !section.ends_with("dev-dependencies") {
            let dependency = key.split('.').next().unwrap_or(key).trim();
            if dependency.starts_with("filmcraft-") {
                dependencies.push(dependency.to_string());
            }
        }
    }
    Some((name?, dependencies))
}

/// The packages that depend on `root`, directly or through one another. They are built after it,
/// so nothing in them can change what `root` renders.
pub fn dependents(manifests: &[(String, Vec<String>)], root: &str) -> std::collections::BTreeSet<String> {
    let mut found = std::collections::BTreeSet::new();
    loop {
        let before = found.len();
        for (name, dependencies) in manifests {
            if name != root && dependencies.iter().any(|d| d == root || found.contains(d)) {
                found.insert(name.clone());
            }
        }
        if found.len() == before {
            return found;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crates_built_on_the_engine_are_found_through_one_another() {
        let ui = manifest("[package]\nname = \"filmcraft-ui-egui\"\n[dependencies]\nfilmcraft-engine = { workspace = true }\nserde = \"1\"\n").unwrap();
        assert_eq!(ui, ("filmcraft-ui-egui".to_string(), vec!["filmcraft-engine".to_string()]));
        let app = manifest("[package]\nname = \"filmcraft-shell\"\n[target.'cfg(unix)'.dependencies]\nfilmcraft-ui-egui.workspace = true\n").unwrap();
        // a dev-dependency does not put a crate above the engine
        let render = manifest("[package]\nname = \"filmcraft-render\"\n[dependencies]\nfilmcraft-time = { workspace = true }\n[dev-dependencies]\nfilmcraft-engine = { workspace = true }\n").unwrap();
        let engine = manifest("[package]\nname = \"filmcraft-engine\"\n[dependencies]\nfilmcraft-render = { workspace = true }\n").unwrap();
        let above = dependents(&[app, render, engine, ui], "filmcraft-engine");
        assert_eq!(above.into_iter().collect::<Vec<_>>(), ["filmcraft-shell", "filmcraft-ui-egui"]);
        assert!(manifest("[workspace]\nmembers = []\n").is_none());
    }
    #[cfg(unix)]
    use std::os::unix::ffi::OsStringExt;

    #[test]
    #[cfg(unix)]
    fn unrelated_non_unicode_environment_is_not_a_build_panic() {
        let result = selected([(OsString::from("UNRELATED"), OsString::from_vec(vec![255]))]).unwrap();
        assert!(result.is_empty());
        let error = selected([(OsString::from("CARGO_ENCODED_RUSTFLAGS"), OsString::from_vec(vec![255]))]).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::InvalidData);
        assert!(error.to_string().contains("CARGO_ENCODED_RUSTFLAGS"));
    }

    #[test]
    fn target_features_and_flags_are_retained_in_stable_order() {
        let inputs =
            [("TARGET", "aarch64-apple-darwin"), ("CARGO_FEATURE_GPU", "1"), ("CARGO_ENCODED_RUSTFLAGS", "-Ctarget-cpu=native"), ("UNRELATED", "ignored")];
        let one = selected(inputs.map(|(k, v)| (k.into(), v.into()))).unwrap();
        let two = selected(inputs.into_iter().rev().map(|(k, v)| (k.into(), v.into()))).unwrap();
        assert_eq!(one, two);
        assert_eq!(one.len(), 3);
        assert!(one.iter().any(|(key, _)| key == "CARGO_FEATURE_GPU"));
    }
}
