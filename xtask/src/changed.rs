//! `cargo xtask ci --changed [--base REF]`: the gates for what changed, for agents iterating
//! between commits. The full `cargo xtask ci` still runs before every push.
//!
//! What changed is everything since `REF` (default: the branch's upstream, else `main`), plus
//! staged, unstaged and untracked files. Each file belongs to the crate whose directory holds it;
//! a change to the workspace itself (the root manifest, `Cargo.lock`, `xtask/`, toolchain or
//! cargo config) runs the full gates instead. Files outside any crate that cannot affect a build
//! (docs, plans, workflows, packaging scripts) only go through the formatting, layering and
//! asset checks, which always run.

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

use serde_json::Value;

/// Files whose change can affect every crate: they run the full gates.
fn is_workspace_file(path: &str) -> bool {
    matches!(path, "Cargo.toml" | "Cargo.lock" | "rust-toolchain" | "rust-toolchain.toml" | "clippy.toml" | "rustfmt.toml")
        || path.starts_with("xtask/")
        || path.starts_with(".cargo/")
}

/// The changed files: committed since `base`, plus staged, unstaged and untracked ones.
fn changed_files(base: &str) -> Result<BTreeSet<String>, String> {
    let git = |args: &[&str]| -> Result<String, String> {
        let out = Command::new("git").args(args).output().map_err(|e| format!("git: {e}"))?;
        if !out.status.success() {
            return Err(format!("git {}: {}", args.join(" "), String::from_utf8_lossy(&out.stderr).trim()));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let range = format!("{base}...HEAD");
    let mut files = BTreeSet::new();
    for list in [git(&["diff", "--name-only", &range])?, git(&["diff", "--name-only", "HEAD"])?, git(&["ls-files", "--others", "--exclude-standard"])?] {
        files.extend(list.lines().map(str::trim).filter(|l| !l.is_empty()).map(str::to_string));
    }
    Ok(files)
}

/// The branch's upstream (`origin/main`, `autocut/main`…), else `main`.
fn default_base() -> String {
    Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "main".into())
}

/// Each workspace package's directory (relative to `root`, with a trailing `/`) → its name.
fn package_dirs(md: &Value, root: &std::path::Path) -> Vec<(String, String)> {
    let mut dirs: Vec<(String, String)> = md["packages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| {
            let name = p["name"].as_str()?;
            let manifest = std::path::Path::new(p["manifest_path"].as_str()?);
            let dir = manifest.parent()?.strip_prefix(root).ok()?;
            let dir = dir.to_string_lossy().replace('\\', "/");
            (!dir.is_empty()).then(|| (format!("{dir}/"), name.to_string()))
        })
        .collect();
    // the longest directory first, so a nested crate wins over its parent
    dirs.sort_by_key(|(d, _)| std::cmp::Reverse(d.len()));
    dirs
}

/// What the changed files mean for the gates.
#[derive(Debug, Default, PartialEq)]
pub(crate) struct Plan {
    /// A workspace file changed: run everything.
    pub full: bool,
    /// Crates with changed files (their tests run).
    pub changed: BTreeSet<String>,
    /// Those and every crate that depends on them, by any kind of dependency (linted).
    pub affected: BTreeSet<String>,
}

/// The plan for `files`, given the packages' directories and each package's dependencies (all
/// kinds: a dev-dependency's change can break the dependent's tests).
pub(crate) fn plan(files: &BTreeSet<String>, dirs: &[(String, String)], deps: &Deps) -> Plan {
    let mut p = Plan::default();
    for f in files {
        if is_workspace_file(f) {
            p.full = true;
        }
        if let Some((_, name)) = dirs.iter().find(|(d, _)| f.starts_with(d.as_str())) {
            p.changed.insert(name.clone());
        }
    }
    // every crate that depends on a changed one, transitively
    p.affected = p.changed.clone();
    loop {
        let more: Vec<String> =
            deps.iter().filter(|(name, ds)| !p.affected.contains(*name) && ds.iter().any(|d| p.affected.contains(d))).map(|(n, _)| n.clone()).collect();
        if more.is_empty() {
            break;
        }
        p.affected.extend(more);
    }
    p
}

/// Each package's dependencies by name.
pub(crate) type Deps = BTreeMap<String, BTreeSet<String>>;

/// Every workspace package's dependencies, all kinds.
fn all_deps(md: &Value) -> Deps {
    md["packages"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| {
            let name = p["name"].as_str()?.to_string();
            let deps = p["dependencies"].as_array().into_iter().flatten().filter_map(|d| d["name"].as_str().map(str::to_string)).collect();
            Some((name, deps))
        })
        .collect()
}

pub(crate) fn run_changed(args: &[String]) -> Result<(), String> {
    let base = match args.iter().position(|a| a == "--base") {
        Some(i) => args.get(i + 1).cloned().ok_or("--base wants a git ref (a branch, tag or commit)")?,
        None => default_base(),
    };
    let root = crate::root();
    let md = crate::metadata()?;
    let files = changed_files(&base)?;
    let p = plan(&files, &package_dirs(&md, &root), &all_deps(&md));
    if p.full {
        eprintln!("ci --changed: a workspace file changed since {base}: the full gates");
        return crate::ci();
    }
    eprintln!(
        "ci --changed: {} file(s) changed since {base}; crates changed: {:?}; also linted (depend on them): {}",
        files.len(),
        p.changed,
        p.affected.len() - p.changed.len()
    );
    let cargo = env!("CARGO");
    crate::run(Command::new(cargo).args(["fmt", "--check"]))?;
    if !p.affected.is_empty() {
        let mut clippy = Command::new(cargo);
        clippy.args(["clippy", "--all-targets", "--release"]);
        for c in &p.affected {
            clippy.args(["-p", c]);
        }
        crate::run(clippy.args(["--", "-D", "warnings"]))?;
        let mut test = Command::new(cargo);
        test.args(["test", "--release", "--no-fail-fast"]);
        for c in &p.changed {
            test.args(["-p", c]);
        }
        crate::run(&mut test)?;
    }
    // EffectCraft linked (off by default): again with the feature, for what of it is affected
    let (lint, test) = effectcraft_plan(&p);
    if !lint.is_empty() {
        crate::effectcraft_gate(&lint, &test)?;
    }
    crate::layers()?;
    crate::assets()?;
    // the wasm check only when a crate that builds for the web is affected
    if p.affected.iter().any(|c| crate::layer_of(c).is_some_and(|l| l <= 4) || crate::WEB_CRATES.contains(&c.as_str())) {
        crate::wasm()?;
    }
    eprintln!("ci --changed: ok — run the full `cargo xtask ci` before pushing");
    Ok(())
}

/// The EffectCraft gate for a plan: clippy with the feature on the affected crates of
/// [`crate::EFFECTCRAFT_CRATES`]; the tests of the adapter whenever it is affected, of the engine
/// and the CLI when they changed themselves.
pub(crate) fn effectcraft_plan(p: &Plan) -> (Vec<&'static str>, Vec<&'static str>) {
    let lint: Vec<&str> = crate::EFFECTCRAFT_CRATES.iter().copied().filter(|c| p.affected.contains(*c)).collect();
    let test = lint.iter().copied().filter(|c| *c == "filmcraft-effectcraft" || p.changed.contains(*c)).collect();
    (lint, test)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(items: &[&str]) -> BTreeSet<String> {
        items.iter().map(|s| s.to_string()).collect()
    }

    fn workspace() -> (Vec<(String, String)>, Deps) {
        let mut dirs = vec![
            ("crates/time/".to_string(), "filmcraft-time".to_string()),
            ("crates/engine/".to_string(), "filmcraft-engine".to_string()),
            ("crates/autocut/".to_string(), "filmcraft-autocut".to_string()),
            ("apps/filmcraft-cli/".to_string(), "filmcraft-cli".to_string()),
        ];
        dirs.sort_by_key(|(d, _)| std::cmp::Reverse(d.len()));
        let deps = [
            ("filmcraft-time", vec![]),
            ("filmcraft-engine", vec!["filmcraft-time"]),
            ("filmcraft-autocut", vec!["filmcraft-engine"]),
            ("filmcraft-cli", vec!["filmcraft-autocut", "serde"]),
        ]
        .into_iter()
        .map(|(n, d)| (n.to_string(), set(&d)))
        .collect();
        (dirs, deps)
    }

    #[test]
    fn a_crate_change_tests_it_and_lints_its_dependents() {
        let (dirs, deps) = workspace();
        let p = plan(&set(&["crates/engine/src/narration.rs"]), &dirs, &deps);
        assert!(!p.full);
        assert_eq!(p.changed, set(&["filmcraft-engine"]));
        assert_eq!(p.affected, set(&["filmcraft-engine", "filmcraft-autocut", "filmcraft-cli"]));
        // a leaf: only itself
        let p = plan(&set(&["apps/filmcraft-cli/src/main.rs"]), &dirs, &deps);
        assert_eq!((p.changed.clone(), p.affected), (set(&["filmcraft-cli"]), set(&["filmcraft-cli"])));
    }

    #[test]
    fn the_effectcraft_gate_follows_what_is_affected() {
        let (mut dirs, mut deps) = workspace();
        dirs.push(("crates/effectcraft/".to_string(), "filmcraft-effectcraft".to_string()));
        dirs.sort_by_key(|(d, _)| std::cmp::Reverse(d.len()));
        deps.insert("filmcraft-effectcraft".into(), set(&["filmcraft-time"]));
        deps.get_mut("filmcraft-engine").unwrap().insert("filmcraft-effectcraft".into());
        // the adapter changed: lint it and what forwards the feature, test it
        let p = plan(&set(&["crates/effectcraft/src/lib.rs"]), &dirs, &deps);
        assert_eq!(effectcraft_plan(&p), (vec!["filmcraft-effectcraft", "filmcraft-engine", "filmcraft-cli"], vec!["filmcraft-effectcraft"]));
        // the engine changed: lint the engine and the CLI, test the engine's EffectCraft part
        let p = plan(&set(&["crates/engine/src/effectcraft.rs"]), &dirs, &deps);
        assert_eq!(effectcraft_plan(&p), (vec!["filmcraft-engine", "filmcraft-cli"], vec!["filmcraft-engine"]));
        // a crate below everything: all three linted, the adapter tested
        let p = plan(&set(&["crates/time/src/lib.rs"]), &dirs, &deps);
        assert_eq!(effectcraft_plan(&p).1, vec!["filmcraft-effectcraft"]);
        // nothing of it: no gate
        let p = plan(&set(&["docs/agents.md"]), &dirs, &deps);
        assert_eq!(effectcraft_plan(&p), (vec![], vec![]));
    }

    #[test]
    fn docs_change_nothing_and_workspace_files_run_everything() {
        let (dirs, deps) = workspace();
        let p = plan(&set(&["plan/execution-plan-2026-10.md", "docs/agents.md", ".github/workflows/autocut-checks.yml"]), &dirs, &deps);
        assert_eq!(p, Plan::default());
        for f in ["Cargo.lock", "Cargo.toml", "xtask/src/main.rs", ".cargo/config.toml"] {
            assert!(plan(&set(&[f]), &dirs, &deps).full, "{f}");
        }
        // a crate's own manifest is that crate's change, not the workspace's
        let p = plan(&set(&["crates/time/Cargo.toml"]), &dirs, &deps);
        assert!(!p.full && p.changed == set(&["filmcraft-time"]) && p.affected.len() == 4);
    }
}
