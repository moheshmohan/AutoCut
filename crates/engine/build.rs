//! Cache identity includes the actual local product source and pinned dependency graph, so a
//! development build at the same package version cannot reuse another renderer's artifacts.
use sha2::{Digest, Sha256};
#[path = "build/configuration.rs"]
mod configuration;
use std::{
    fs,
    io::{Error, ErrorKind, Read},
    path::Path,
};
fn collect(path: &Path, depth: usize, files: &mut Vec<std::path::PathBuf>) -> std::io::Result<()> {
    if depth > 20 || files.len() > 30000 {
        return Err(Error::new(ErrorKind::InvalidData, "render source tree exceeds fingerprint limits"));
    }
    let meta = fs::symlink_metadata(path)?;
    if meta.is_symlink() {
        return Ok(());
    }
    if meta.is_dir() {
        let mut children = fs::read_dir(path)?.collect::<std::io::Result<Vec<_>>>()?;
        children.sort_by_key(|e| e.file_name());
        for entry in children {
            let p = entry.path();
            if p.file_name().is_some_and(|s| s == "target" || s == ".git") {
                continue;
            }
            collect(&p, depth + 1, files)?;
        }
    } else if meta.is_file() && path.extension().is_some_and(|s| ["rs", "wgsl", "toml"].iter().any(|e| s == *e)) {
        files.push(path.into());
    }
    Ok(())
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new("../..");
    let mut files = Vec::new();
    // Crates built on the engine (the UI, automation, the publishing adapters) cannot change what
    // it renders. Hashing them too rebuilt the engine, and everything above it, after every edit
    // to one of them.
    let mut crates = fs::read_dir(root.join("crates"))?.collect::<std::io::Result<Vec<_>>>()?;
    crates.sort_by_key(|e| e.file_name());
    let manifests: Vec<_> = crates
        .iter()
        .map(|e| {
            let manifest = e.path().join("Cargo.toml");
            println!("cargo:rerun-if-changed={}", manifest.display());
            fs::read_to_string(manifest).ok().and_then(|text| configuration::manifest(&text))
        })
        .collect();
    let known: Vec<_> = manifests.iter().flatten().cloned().collect();
    let above = configuration::dependents(&known, "filmcraft-engine");
    for (entry, manifest) in crates.iter().zip(&manifests) {
        if manifest.as_ref().is_some_and(|(name, _)| above.contains(name)) {
            continue;
        }
        collect(&entry.path(), 1, &mut files)?;
    }
    files.push(root.join("Cargo.lock"));
    files.push(root.join("Cargo.toml"));
    files.sort();
    let mut hash = Sha256::new();
    // Equal source at different feature/target configurations is not the same renderer.
    // In particular, workspace tests and a standalone CLI can unify different features.
    for (key, value) in configuration::selected(std::env::vars_os())? {
        println!("cargo:rerun-if-env-changed={key}");
        hash.update(key.as_bytes());
        hash.update([0]);
        hash.update(value.as_bytes());
        hash.update([0]);
    }
    let mut buf = [0u8; 65536];
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        hash.update(path.to_string_lossy().as_bytes());
        let mut f = fs::File::open(path)?;
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 {
                break;
            }
            hash.update(buf.get(..n).ok_or("invalid hash read")?);
        }
    }
    println!("cargo:rustc-env=AUTOCUT_RENDER_ENGINE_HASH={:x}", hash.finalize());
    Ok(())
}
