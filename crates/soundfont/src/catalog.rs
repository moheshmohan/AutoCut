//! The downloadable SoundFont: **GeneralUser GS** by S. Christian Collins.
//!
//! It is never bundled with FilmCraft or committed: the engine downloads it on first use, after
//! the user agreed to its size, source and licence (shown in full: [`GENERALUSER_GS_LICENSE`]),
//! into `<data dir>/models/generaluser-gs/`, every file checked against its pinned SHA-256 and
//! size (`filmcraft_speech::models::download_files`, as the voice and speech models are).
//!
//! The source is the author's own GitHub repository, which he keeps "to offer an easier source for
//! automated packaging of GeneralUser GS and its support files", pinned to the commit that
//! published v2.0.3 (2026-02-23). Pins checked 2026-10-17: the SHA-256 below, and the git blob id
//! of the download matching the repository's (`298b552d…`).
//!
//! **Licence** (GeneralUser GS License v2.0): use without restriction for music creation, private
//! or commercial; free to use in software projects and to modify. The author notes he cannot be
//! 100% sure where every sample originated (none from commercial packages or sample CDs, as far as
//! he knows) — shown to the user with the licence before they agree.

use std::path::{Path, PathBuf};

/// One file of a package.
#[derive(Clone, Copy, Debug)]
pub struct FontFile {
    /// Local name inside the package directory.
    pub name: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub size: u64,
}

/// A downloadable SoundFont.
#[derive(Clone, Copy, Debug)]
pub struct Package {
    /// Id in commands and the package's directory under the models directory.
    pub id: &'static str,
    pub name: &'static str,
    pub version: &'static str,
    pub author: &'static str,
    pub description: &'static str,
    /// The licence's name.
    pub license: &'static str,
    /// The licence in full, as the author publishes it.
    pub license_text: &'static str,
    pub license_url: &'static str,
    /// What the author says about where the samples came from.
    pub provenance: &'static str,
    /// Human-readable source page.
    pub source: &'static str,
    /// The SoundFont file among `files`.
    pub sf2: &'static str,
    pub files: &'static [FontFile],
}

const COMMIT: &str = "97049183643d5fc5a9322a69c5b09efb667c6c3a";

/// The GeneralUser GS License v2.0, as published with v2.0.3 (`documentation/LICENSE.txt`).
pub const GENERALUSER_GS_LICENSE: &str = "*** GeneralUser GS v2.0.3 ***
***      License v2.0     ***

** License of the complete work **
You may use GeneralUser GS without restriction for your own music creation,
private or commercial. This SoundFont bank is provided to the community free of
charge. Please feel free to use it in your software projects, and to modify the
SoundFont bank or its packaging to suit your needs.

** License of contained samples **
GeneralUser GS inherits the usage rights of the samples contained within, all of
which allow full use in music production, including the ability to make profit
from musical recordings created with GeneralUser GS.

Many of the samples are original, but some were taken from other banks freely
(and legally) available on the Internet from various SoundFont websites. Because
GeneralUser GS originated as a personal project with no intention for
publication, I cannot be 100% sure where all of the samples originated, although
I do know that none of them came from commercially published SoundFont packages
or sample CDs. Regardless, many \"free\" SoundFonts available on the web may
indeed contain samples of questionable origin. My understanding of the
copyrights of all samples is only as good as the information provided by the
original sources. If you become aware of any restricted samples being used in
GeneralUser GS, please let me know so I can replace them.

This uncertainty may concern you if you intend to use GeneralUser GS in a
commercial software product. That being said, I have never received any
complaint regarding sample ownership since I published the original GeneralUser
GS back in 2000, and as far as I am aware, neither have any of the companies
creating commercial software products using GeneralUser GS.

** More info **
If you plan to feature GeneralUser GS on your own website, please do not link
directly to my download files. Either link to my website, or provide your own
local copy instead.

I hope you enjoy GeneralUser GS! This SoundFont bank is the product of many
years of hard work.

You can find updates to GeneralUser GS and more of my virtual instruments at:
http://www.schristiancollins.com

I can be reached via the contact page on my website here:
https://www.schristiancollins.com/contact

Thank you!
-~Chris
";

/// GeneralUser GS v2.0.3.
pub static GENERALUSER_GS: Package = Package {
    id: "generaluser-gs",
    name: "GeneralUser GS",
    version: "2.0.3",
    author: "S. Christian Collins",
    description: "A General MIDI / Roland GS SoundFont: 261 instrument presets and 13 drum kits in 32 MB. The procedural music plays its pianos, strings, pads, brass, basses and drum kits on it instead of the built-in oscillator voices.",
    license: "GeneralUser GS License v2.0 (use without restriction for music creation, private or commercial)",
    license_text: GENERALUSER_GS_LICENSE,
    license_url: "https://github.com/mrbumpy409/GeneralUser-GS/blob/97049183643d5fc5a9322a69c5b09efb667c6c3a/documentation/LICENSE.txt",
    provenance: "The author notes that some samples came from other freely available SoundFonts and that he cannot be 100% sure where all of them originated (none from commercial packages or sample CDs, as far as he knows).",
    source: "https://github.com/mrbumpy409/GeneralUser-GS",
    sf2: "GeneralUser-GS.sf2",
    files: &[
        FontFile {
            name: "GeneralUser-GS.sf2",
            url: "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/97049183643d5fc5a9322a69c5b09efb667c6c3a/GeneralUser-GS.sf2",
            sha256: "9575028c7a1f589f5770fccc8cff2734566af40cd26ed836944e9a5152688cfe",
            size: 32_319_396,
        },
        FontFile {
            name: "LICENSE.txt",
            url: "https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/97049183643d5fc5a9322a69c5b09efb667c6c3a/documentation/LICENSE.txt",
            sha256: "7b32efefdf95ce38a043799f0659853ddc00fbaa14d8c50f0aca16b9b8b405be",
            size: 2317,
        },
    ],
};

/// Every downloadable SoundFont.
pub static PACKAGES: &[&Package] = &[&GENERALUSER_GS];

/// The package with this id.
pub fn package(id: &str) -> Option<&'static Package> {
    PACKAGES.iter().copied().find(|p| p.id == id)
}

/// The commit every URL is pinned to.
pub fn commit() -> &'static str {
    COMMIT
}

/// Where a package lives under the models directory.
pub fn package_dir(models_dir: &Path, p: &Package) -> PathBuf {
    models_dir.join(p.id)
}

/// The package's `.sf2` file under the models directory.
pub fn sf2_path(models_dir: &Path, p: &Package) -> PathBuf {
    package_dir(models_dir, p).join(p.sf2)
}

/// Download size, bytes.
pub fn size(p: &Package) -> u64 {
    p.files.iter().map(|f| f.size).sum()
}

/// Bytes still to download (files missing or of the wrong size).
pub fn missing_bytes(models_dir: &Path, p: &Package) -> u64 {
    let d = package_dir(models_dir, p);
    p.files.iter().filter(|f| !std::fs::metadata(d.join(f.name)).is_ok_and(|md| md.len() == f.size)).map(|f| f.size).sum()
}

/// Every file is present with its pinned size.
pub fn installed(models_dir: &Path, p: &Package) -> bool {
    missing_bytes(models_dir, p) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_are_well_formed() {
        for p in PACKAGES {
            assert!(package(p.id).is_some());
            assert!(p.files.iter().any(|f| f.name == p.sf2));
            assert!(!p.license.contains("NC") && !p.license.contains("ND"));
            assert!(p.source.starts_with("https://github.com/mrbumpy409/"));
            for f in p.files {
                assert_eq!(f.sha256.len(), 64, "{}", f.name);
                assert!(f.sha256.chars().all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()));
                assert!(f.url.starts_with("https://raw.githubusercontent.com/mrbumpy409/GeneralUser-GS/") && f.url.contains(COMMIT), "{}", f.url);
                assert!(!f.name.contains('/') && !f.name.contains(".."));
                assert!(f.size > 0);
            }
        }
        // the licence shown is the one downloaded: same length (CRLF line ends in the file)
        let crlf = GENERALUSER_GS_LICENSE.lines().count() + GENERALUSER_GS_LICENSE.len();
        assert_eq!(crlf as u64, GENERALUSER_GS.files[1].size);
        assert!(GENERALUSER_GS_LICENSE.contains("cannot be 100% sure where all of the samples originated"));
    }

    #[test]
    fn install_state_follows_the_files() {
        let d = std::env::temp_dir().join(format!("filmcraft-sf-cat-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let p = &GENERALUSER_GS;
        assert!(!installed(&d, p));
        assert_eq!(missing_bytes(&d, p), size(p));
        std::fs::create_dir_all(package_dir(&d, p)).unwrap();
        std::fs::write(package_dir(&d, p).join("LICENSE.txt"), vec![b'x'; 2317]).unwrap();
        assert_eq!(missing_bytes(&d, p), 32_319_396);
        std::fs::write(sf2_path(&d, p), b"short").unwrap();
        assert!(!installed(&d, p), "a file of the wrong size is missing");
        let _ = std::fs::remove_dir_all(&d);
    }
}
