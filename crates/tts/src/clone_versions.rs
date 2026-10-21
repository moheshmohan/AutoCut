//! Versioned references of a natural cloned voice: [`promote`] puts a new recording in place of
//! the voice sample (VoxCPM, English) or of the Malayalam reference (IndicF5), keeping what was
//! there as a version to go back to ([`rollback`], [`versions`]).
//!
//! - **Where**: `<voices>/<name>.versions/r<revision>/`, a copy of every reference file the voice
//!   had ([`super::REFERENCE_FILES`] and the profile) and `version.json` ([`Version`]).
//! - **Consent** carries over: the same speaker, recordings of their own, under the profile that
//!   recorded their consent (the caller checks the new recording is the same speaker).
//! - **The narration cache** keys natural voices on their reference's bytes (the voice sample's,
//!   the Malayalam clip's and words), so readings made with an earlier reference are never
//!   served for the new one; a reference rolled back to hits its own readings again. In memory,
//!   prompts are keyed by the profile's [`super::Profile::revision`] too.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::*;

/// What [`promote`] replaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Part {
    /// The voice sample VoxCPM speaks from (English).
    Sample,
    /// The Malayalam reference IndicF5 continues.
    Malayalam,
}

impl Part {
    pub fn id(self) -> &'static str {
        match self {
            Part::Sample => "sample",
            Part::Malayalam => "malayalam",
        }
    }
}

/// A recording to make a voice's reference from.
#[derive(Clone, Debug, PartialEq)]
pub struct NewReference<'a> {
    pub part: Part,
    /// The stretch of the recording, mono, at the rate it was read at (44.1 kHz keeps the full
    /// band the Malayalam reference and `<name>.full.wav` want).
    pub audio: &'a [f32],
    pub rate: u32,
    /// Where it is from: the recording and the stretch (`IMG_1234.MOV [12.0–30.5 s]`, as a
    /// clone's source is written), and its times in that recording.
    pub source: &'a str,
    pub from: f64,
    pub to: f64,
    /// Its words: required for the Malayalam reference (IndicF5 aligns its reading to them), and
    /// for a voice that continues its sample.
    pub transcript: Option<&'a str>,
    /// Why (the audition it won, its speaker similarity), kept with the version.
    pub note: &'a str,
}

/// A reference a voice had.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Version {
    /// The profile's revision when this was its reference.
    pub revision: u32,
    /// The folder it is kept in, under `<name>.versions`.
    pub dir: String,
    /// When it was put aside (Unix seconds).
    pub saved: u64,
    /// Why it was put aside.
    pub reason: String,
    /// Where its references came from.
    pub source: String,
    #[serde(default)]
    pub malayalam_source: String,
    /// SHA-256 of its voice sample and Malayalam clip.
    pub sample_sha: String,
    #[serde(default)]
    pub malayalam_sha: String,
    /// The files kept.
    pub files: Vec<String>,
}

/// Versions kept per voice at most (oldest removed first).
pub const MAX_VERSIONS: usize = 50;
const VERSION_FILE: &str = "version.json";

fn versions_dir(voices: &Path, p: &Profile) -> Result<PathBuf, TtsError> {
    Ok(profile_path(voices, &p.id)?.with_extension(VERSIONS_EXT))
}

fn io(path: &Path) -> impl Fn(std::io::Error) -> TtsError + '_ {
    move |e| TtsError::Model(format!("{}: {e}", path.display()))
}

/// The voice's reference files that exist now (file names next to the profile), the profile first.
fn present(voices: &Path, p: &Profile) -> Result<Vec<String>, TtsError> {
    let base = profile_path(voices, &p.id)?;
    let mut out = Vec::new();
    for ext in std::iter::once("json").chain(REFERENCE_FILES.iter().copied()) {
        let f = base.with_extension(ext);
        if f.is_file()
            && let Some(name) = f.file_name()
        {
            out.push(name.to_string_lossy().into_owned());
        }
    }
    Ok(out)
}

/// Every version kept for `id`, oldest first.
pub fn versions(models_dir: &Path, id: &str) -> Result<Vec<Version>, TtsError> {
    let voices = voices_dir(models_dir);
    let p = load_profile(&voices, id)?;
    let dir = versions_dir(&voices, &p)?;
    let Ok(rd) = std::fs::read_dir(&dir) else { return Ok(Vec::new()) };
    let mut out: Vec<Version> = rd
        .filter_map(|e| e.ok())
        .take(MAX_VERSIONS * 4)
        .filter_map(|e| {
            let path = e.path().join(VERSION_FILE);
            let meta = std::fs::metadata(&path).ok()?;
            if meta.len() > MAX_PROFILE_BYTES {
                return None;
            }
            let mut v = serde_json::from_slice::<Version>(&std::fs::read(&path).ok()?).ok()?;
            // the folder is where it was found, whatever the file says
            v.dir = e.file_name().to_string_lossy().into_owned();
            Some(v)
        })
        .collect();
    out.sort_by(|a, b| a.saved.cmp(&b.saved).then(a.revision.cmp(&b.revision)));
    Ok(out)
}

/// Put the voice's current references aside as a version; returns it.
fn archive(voices: &Path, p: &Profile, reason: &str) -> Result<Version, TtsError> {
    let root = versions_dir(voices, p)?;
    std::fs::create_dir_all(&root).map_err(io(&root))?;
    let mut name = format!("r{}", p.revision);
    let mut k = 1;
    while root.join(&name).exists() {
        k += 1;
        if k > 1000 {
            return Err(TtsError::Model(format!("{}: too many versions", root.display())));
        }
        name = format!("r{}-{k}", p.revision);
    }
    let dir = root.join(&name);
    std::fs::create_dir_all(&dir).map_err(io(&dir))?;
    let files = present(voices, p)?;
    for f in &files {
        std::fs::copy(voices.join(f), dir.join(f)).map_err(io(&dir))?;
    }
    let cross = cross_path(voices, p).ok().and_then(|c| std::fs::read(c).ok()).and_then(|b| serde_json::from_slice::<CrossReference>(&b).ok());
    let v = Version {
        revision: p.revision,
        dir: name,
        saved: now(),
        reason: reason.chars().take(500).collect(),
        source: p.source.clone(),
        malayalam_source: cross.as_ref().map(|c| c.source.clone()).unwrap_or_default(),
        sample_sha: sample_hash(voices, p).unwrap_or_default(),
        malayalam_sha: cross.map(|c| c.clip_sha).unwrap_or_default(),
        files,
    };
    let bytes = serde_json::to_vec_pretty(&v).map_err(|e| TtsError::Model(e.to_string()))?;
    std::fs::write(dir.join(VERSION_FILE), bytes).map_err(io(&dir))?;
    prune(&root);
    Ok(v)
}

/// Keep at most [`MAX_VERSIONS`] versions (the oldest go).
fn prune(root: &Path) {
    let Ok(rd) = std::fs::read_dir(root) else { return };
    let mut dirs: Vec<(std::time::SystemTime, PathBuf)> = rd
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join(VERSION_FILE).is_file())
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path())))
        .collect();
    if dirs.len() <= MAX_VERSIONS {
        return;
    }
    dirs.sort();
    for (_, d) in dirs.iter().take(dirs.len() - MAX_VERSIONS) {
        let _ = std::fs::remove_dir_all(d);
    }
}

/// Write `bytes` to `path` through a temporary file.
fn put(path: &Path, bytes: &[u8]) -> Result<(), TtsError> {
    let tmp = path.with_extension("part");
    std::fs::write(&tmp, bytes).map_err(io(&tmp))?;
    std::fs::rename(&tmp, path).map_err(io(path))
}

fn put_wav(path: &Path, x: &[f32], rate: u32) -> Result<(), TtsError> {
    let tmp = path.with_extension("part.wav");
    crate::wav::write(&tmp, x, rate)?;
    std::fs::rename(&tmp, path).map_err(io(path))
}

/// The voice sample a recording makes: every bit of it (it was chosen already), pauses over half
/// a second shortened, the edges trimmed and the loudness at [`SAMPLE_LUFS`] — as a clone's own
/// sample is made.
pub fn sample_from(audio: &[f32], rate: u32) -> Result<Sample, TtsError> {
    let secs = audio.len() as f64 / f64::from(rate.max(1));
    if rate == 0 || secs < MIN_SPEECH_SECONDS {
        return Err(TtsError::Invalid(format!("the recording is {secs:.1} s; a reference needs at least {MIN_SPEECH_SECONDS} s")));
    }
    pick_sample_within(audio, rate, (secs * 0.95, secs + 1.0))
}

/// Make `r` the reference of the natural cloned voice `id` (see the module docs): the voice's
/// current references are kept as a version first. Returns that version.
pub fn promote(models_dir: &Path, id: &str, r: &NewReference) -> Result<Version, TtsError> {
    let voices = voices_dir(models_dir);
    let mut p = load_profile(&voices, id)?;
    let n =
        p.natural.clone().ok_or_else(|| TtsError::Invalid(format!("{id} is a timbre-only clone: its reference cannot be replaced (clone the voice again)")))?;
    if r.rate == 0 || r.audio.iter().any(|v| !v.is_finite()) {
        return Err(TtsError::Invalid("the new reference's audio is damaged".into()));
    }
    if !(r.from.is_finite() && r.to.is_finite() && r.to > r.from) {
        return Err(TtsError::Invalid("the new reference's times make no sense".into()));
    }
    let words: Option<String> = r.transcript.map(|t| t.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|t| t.chars().any(char::is_alphanumeric));
    if words.as_ref().is_some_and(|w| w.chars().count() > MAX_TRANSCRIPT_CHARS) {
        return Err(TtsError::Invalid("the new reference's words are too long".into()));
    }
    let secs = r.audio.len() as f64 / f64::from(r.rate);
    let next = versions(models_dir, id)?.iter().map(|v| v.revision).chain(std::iter::once(p.revision)).max().unwrap_or(0).saturating_add(1);
    match r.part {
        Part::Sample => {
            let bounds = if n.mode == PromptMode::Continuation { (CONTINUATION_SECONDS.0, 30.0) } else { (SAMPLE_SECONDS.0, 30.5) };
            if secs < bounds.0 || secs > bounds.1 {
                return Err(TtsError::Invalid(format!("a voice sample is {:.0}–{:.0} s; this one is {secs:.1} s", bounds.0, bounds.1)));
            }
            if n.mode == PromptMode::Continuation && words.is_none() {
                return Err(TtsError::Invalid("this voice continues its sample: give the new sample's exact words".into()));
            }
            let sample = sample_from(r.audio, r.rate)?;
            let version = archive(&voices, &p, &format!("replaced by {} ({})", r.source, r.note))?;
            let encode = natural_encode_rate(&n.model);
            let clip = filmcraft_audio_dsp::resample::resample_sinc(&sample.audio, r.rate, encode);
            put_wav(&voices.join(&n.clip), &clip, encode)?;
            let full = full_sample_path(&voices, &p)?;
            if r.rate > encode {
                put_wav(&full, &sample.audio, r.rate)?;
            } else {
                let _ = std::fs::remove_file(&full);
            }
            let cross_before = cross_reference_unchecked(&voices, &p);
            let mut nn = n.clone();
            // as a clone's own sample: seconds into the stretch its source names
            nn.from = sample.from;
            nn.to = sample.to;
            nn.transcript = if n.mode == PromptMode::Continuation { words.unwrap_or_default() } else { String::new() };
            nn.gender = estimate_gender(&sample.audio, r.rate).or(n.gender);
            p.natural = Some(nn);
            p.reference_seconds = clip.len() as f64 / f64::from(encode);
            p.source = r.source.chars().take(500).collect();
            p.revision = next;
            p.check()?;
            save(&voices, &p)?;
            // the Malayalam reference stays the voice's: it is tied to the new sample now
            if let Some(mut c) = cross_before {
                c.sample = sample_hash(&voices, &p)?;
                put(&cross_path(&voices, &p)?, &serde_json::to_vec_pretty(&c).map_err(|e| TtsError::Model(e.to_string()))?)?;
            }
            Ok(version)
        }
        Part::Malayalam => {
            let (lo, hi) = (SENTENCE_REFERENCE_MIN_SECONDS - 2.0, CROSS_REFERENCE_SECONDS.1 + 0.5);
            if secs < lo || secs > hi {
                return Err(TtsError::Invalid(format!("a Malayalam reference is {lo:.0}–{hi:.0} s; this one is {secs:.1} s")));
            }
            let words =
                words.ok_or_else(|| TtsError::Invalid("a Malayalam reference needs the clip's exact words (IndicF5 aligns its reading to them)".into()))?;
            let version = archive(&voices, &p, &format!("Malayalam reference replaced by {} ({})", r.source, r.note))?;
            let mut clip = filmcraft_audio_dsp::resample::resample_sinc(r.audio, r.rate, CROSS_REFERENCE_RATE);
            crate::voice_lab::signal::fade(&mut clip, CROSS_REFERENCE_RATE);
            let name = format!("{}.indicf5.wav", slug(p.id.trim_start_matches(CLONE_PREFIX)));
            put_wav(&voices.join(&name), &clip, CROSS_REFERENCE_RATE)?;
            // the clip before a trim belonged to the old reference (kept in the version)
            let (uw, uj) = untrimmed_paths(&voices, &p)?;
            let _ = std::fs::remove_file(uw);
            let _ = std::fs::remove_file(uj);
            let c = CrossReference {
                version: CROSS_REFERENCE_VERSION,
                sample: sample_hash(&voices, &p)?,
                source: r.source.chars().take(500).collect(),
                clip_sha: sha_hex(&read_capped(&voices.join(&name))?),
                clip: name,
                // seconds into the stretch its source names
                from: 0.0,
                to: r.to - r.from,
                transcript: words,
                given: false,
            };
            put(&cross_path(&voices, &p)?, &serde_json::to_vec_pretty(&c).map_err(|e| TtsError::Model(e.to_string()))?)?;
            p.revision = next;
            save(&voices, &p)?;
            Ok(version)
        }
    }
}

/// The Malayalam reference file of `p` as it is on disk, fresh or not.
fn cross_reference_unchecked(voices: &Path, p: &Profile) -> Option<CrossReference> {
    let path = cross_path(voices, p).ok()?;
    if std::fs::metadata(&path).ok()?.len() > MAX_PROFILE_BYTES {
        return None;
    }
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// Go back to the version of `id` kept in `dir` (a [`Version::dir`]): the current references are
/// kept as a version first, so a rollback can be undone too. Returns the version restored.
pub fn rollback(models_dir: &Path, id: &str, dir: &str) -> Result<Version, TtsError> {
    let voices = voices_dir(models_dir);
    let p = load_profile(&voices, id)?;
    let v = versions(models_dir, id)?
        .into_iter()
        .find(|v| v.dir == dir)
        .ok_or_else(|| TtsError::Invalid(format!("{id} has no version `{dir}` (voice.versions lists them)")))?;
    let from = versions_dir(&voices, &p)?.join(&v.dir);
    // only the voice's own reference files come back, and the profile must still be this voice
    let own = present_names(&voices, &p)?;
    if v.files.iter().any(|f| !own.contains(f)) {
        return Err(TtsError::Invalid(format!("version `{dir}` names files that are not this voice's")));
    }
    let profile_name = own.first().cloned().unwrap_or_default();
    let restored: Profile = serde_json::from_slice(&read_capped(&from.join(&profile_name))?).map_err(|e| TtsError::Invalid(format!("version `{dir}`: {e}")))?;
    restored.check()?;
    if restored.id != p.id {
        return Err(TtsError::Invalid(format!("version `{dir}` is of {}", restored.id)));
    }
    archive(&voices, &p, &format!("rolled back to {}", v.dir))?;
    for f in own.iter().skip(1) {
        let src = from.join(f);
        let dst = voices.join(f);
        if v.files.contains(f) {
            let tmp = dst.with_extension("part");
            std::fs::copy(&src, &tmp).map_err(io(&src))?;
            std::fs::rename(&tmp, &dst).map_err(io(&dst))?;
        } else {
            let _ = std::fs::remove_file(&dst);
        }
    }
    save(&voices, &restored)?;
    Ok(v)
}

/// The file names a voice's references may have, the profile first.
fn present_names(voices: &Path, p: &Profile) -> Result<Vec<String>, TtsError> {
    let base = profile_path(voices, &p.id)?;
    Ok(std::iter::once("json")
        .chain(REFERENCE_FILES.iter().copied())
        .filter_map(|ext| base.with_extension(ext).file_name().map(|n| n.to_string_lossy().into_owned()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::super::tests::natural_profile;
    use super::*;

    fn data(name: &str) -> (PathBuf, PathBuf) {
        let d = std::env::temp_dir().join(format!("filmcraft-versions-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        let (models, voices) = (d.join("models"), d.join("voices"));
        std::fs::create_dir_all(&models).unwrap();
        std::fs::create_dir_all(&voices).unwrap();
        (models, voices)
    }

    /// Speech-like audio: syllables of a 130 Hz tone with short pauses.
    fn talk(secs: f64, rate: u32, f0: f32) -> Vec<f32> {
        (0..(secs * f64::from(rate)) as usize)
            .map(|i| {
                let t = i as f32 / rate as f32;
                let env = if (t % 2.0) > 1.7 { 0.0 } else { (std::f32::consts::PI * 4.0 * t).sin().abs() };
                0.3 * env * (std::f32::consts::TAU * f0 * t).sin()
            })
            .collect()
    }

    fn voice(voices: &Path) -> Profile {
        let p = natural_profile();
        save(voices, &p).unwrap();
        crate::wav::write(&voices.join("my-voice.wav"), &talk(14.0, 16_000, 130.0), 16_000).unwrap();
        p
    }

    #[test]
    fn promoting_keeps_the_old_reference_and_rolls_back() {
        let (models, voices) = data("promote");
        let p = voice(&voices);
        let old_sample = std::fs::read(voices.join("my-voice.wav")).unwrap();
        // a Malayalam reference tied to the sample
        let clip = talk(9.0, 24_000, 130.0);
        crate::wav::write(&voices.join("my-voice.indicf5.wav"), &clip, 24_000).unwrap();
        let c = CrossReference {
            version: CROSS_REFERENCE_VERSION,
            sample: sample_hash(&voices, &p).unwrap(),
            source: "IMG_3337.MOV".into(),
            clip: "my-voice.indicf5.wav".into(),
            clip_sha: sha_hex(&std::fs::read(voices.join("my-voice.indicf5.wav")).unwrap()),
            from: 1.0,
            to: 10.0,
            transcript: "ഇത് ഒരു റോബോട്ട് ആണ്.".into(),
            given: true,
        };
        std::fs::write(voices.join("my-voice.indicf5.json"), serde_json::to_vec(&c).unwrap()).unwrap();
        assert!(cross_reference(&models, "clone:my-voice").unwrap().is_some());
        let id_before = crate::cache::voice_identity("clone:my-voice", Some(&models));
        // a new English sample
        let new = talk(18.0, 44_100, 125.0);
        let r = NewReference {
            part: Part::Sample,
            audio: &new,
            rate: 44_100,
            source: "IMG_9.MOV [3.0–21.0 s]",
            from: 3.0,
            to: 21.0,
            transcript: None,
            note: "test",
        };
        let v = promote(&models, "clone:my-voice", &r).unwrap();
        assert_eq!((v.revision, v.dir.as_str()), (0, "r0"));
        assert!(v.files.contains(&"my-voice.json".to_string()) && v.files.contains(&"my-voice.wav".to_string()));
        let q = load_profile(&voices, "clone:my-voice").unwrap();
        assert_eq!(q.revision, 1);
        assert!(q.consent && q.created == p.created, "consent carried over");
        assert_eq!(q.source, "IMG_9.MOV [3.0–21.0 s]");
        let nq = q.natural.as_ref().unwrap();
        assert!(nq.from >= 0.0 && nq.to <= 18.5 && nq.to > nq.from, "seconds into the stretch: {nq:?}");
        assert_ne!(std::fs::read(voices.join("my-voice.wav")).unwrap(), old_sample);
        assert!(voices.join("my-voice.full.wav").is_file(), "the full band kept");
        // the narration cache will not serve old readings, and Malayalam still works
        assert_ne!(crate::cache::voice_identity("clone:my-voice", Some(&models)), id_before);
        assert!(cross_reference(&models, "clone:my-voice").unwrap().is_some());
        // a new Malayalam reference
        let ml = talk(9.5, 44_100, 125.0);
        let r = NewReference {
            part: Part::Malayalam,
            audio: &ml,
            rate: 44_100,
            source: "IMG_9.MOV [40–49.5 s]",
            from: 40.0,
            to: 49.5,
            transcript: Some("ഇതാണ് സ്പോട്ട്."),
            note: "test",
        };
        let v1 = promote(&models, "clone:my-voice", &r).unwrap();
        assert_eq!(v1.revision, 1);
        let cr = cross_reference(&models, "clone:my-voice").unwrap().unwrap();
        assert_eq!((cr.transcript.as_str(), cr.from, cr.to), ("ഇതാണ് സ്പോട്ട്.", 0.0, 9.5));
        assert_eq!(load_profile(&voices, "clone:my-voice").unwrap().revision, 2);
        let vs = versions(&models, "clone:my-voice").unwrap();
        assert_eq!(vs.iter().map(|v| v.dir.as_str()).collect::<Vec<_>>(), ["r0", "r1"]);
        // back to the voice as made
        let back = rollback(&models, "clone:my-voice", "r0").unwrap();
        assert_eq!(back.dir, "r0");
        assert_eq!(std::fs::read(voices.join("my-voice.wav")).unwrap(), old_sample);
        assert!(!voices.join("my-voice.full.wav").exists(), "a file the version did not have is gone");
        assert_eq!(load_profile(&voices, "clone:my-voice").unwrap(), p);
        assert_eq!(cross_reference(&models, "clone:my-voice").unwrap().unwrap(), c);
        assert_eq!(crate::cache::voice_identity("clone:my-voice", Some(&models)), id_before, "the old readings are served again");
        // and the rollback itself was kept
        assert_eq!(versions(&models, "clone:my-voice").unwrap().len(), 3);
        // deleting the voice takes its versions
        delete(&voices, "clone:my-voice").unwrap();
        assert!(!voices.join("my-voice.versions").exists());
    }

    #[test]
    fn bad_promotions_change_nothing() {
        let (models, voices) = data("bad");
        voice(&voices);
        let before = std::fs::read(voices.join("my-voice.wav")).unwrap();
        let ok = talk(18.0, 44_100, 125.0);
        let nan = vec![f32::NAN; 44_100 * 18];
        let short = talk(4.0, 44_100, 125.0);
        let r =
            |part, audio: &'static [f32], rate, from: f64, to: f64, transcript| NewReference { part, audio, rate, source: "x", from, to, transcript, note: "" };
        let ok: &'static [f32] = Box::leak(ok.into_boxed_slice());
        let nan: &'static [f32] = Box::leak(nan.into_boxed_slice());
        let short: &'static [f32] = Box::leak(short.into_boxed_slice());
        for bad in [
            r(Part::Sample, nan, 44_100, 0.0, 18.0, None),
            r(Part::Sample, ok, 0, 0.0, 18.0, None),
            r(Part::Sample, ok, 44_100, f64::NAN, 18.0, None),
            r(Part::Sample, ok, 44_100, 5.0, 1.0, None),
            r(Part::Sample, short, 44_100, 0.0, 4.0, None),
            r(Part::Malayalam, ok, 44_100, 0.0, 18.0, Some("ഇത്.")),
            r(Part::Malayalam, short, 44_100, 0.0, 4.0, Some("ഇത്.")),
            r(Part::Malayalam, Box::leak(talk(9.0, 44_100, 125.0).into_boxed_slice()), 44_100, 0.0, 9.0, None),
        ] {
            assert!(promote(&models, "clone:my-voice", &bad).is_err(), "{:?} {} {}", bad.part, bad.rate, bad.audio.len());
        }
        assert_eq!(std::fs::read(voices.join("my-voice.wav")).unwrap(), before);
        assert!(versions(&models, "clone:my-voice").unwrap().is_empty(), "nothing archived for a refused promotion");
        assert!(promote(&models, "clone:nobody", &r(Part::Sample, ok, 44_100, 0.0, 18.0, None)).is_err());
        assert!(rollback(&models, "clone:my-voice", "r9").is_err());
        assert!(rollback(&models, "clone:my-voice", "../../etc").is_err());
    }
}
