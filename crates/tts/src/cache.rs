//! The narration cache: synthesized speech kept on disk, so a script read by a voice is
//! synthesized once — across sessions, rebuilds and projects. A cloned voice runs at about 2.6×
//! real time, so rebuilding a 25-line show would otherwise re-read every line for minutes.
//!
//! - **Key**: SHA-256 of [`FORMAT_VERSION`], the voice's [`voice_identity`] (its id, the pinned
//!   model files it speaks with and, for a cloned voice, its profile: id, version, creation time,
//!   base voice and a hash of its tone colours), the pitch and pace (exact bits) and the script.
//!   Anything that changes the samples changes the key; an old entry is never served for it.
//! - **Entry**: `<key>.fcnc` — a 28-byte header (magic `FCNC`, format version, sample rate,
//!   sample count, FNV-1a checksum of the samples) and the mono samples as little-endian `f32`,
//!   exactly as the voice made them (nothing is re-quantized).
//! - **Writes are atomic**: a temporary file in the same directory, renamed into place.
//! - **Size cap** ([`DEFAULT_CAP_BYTES`], 2 GB): after each write the oldest entries by last use
//!   (the file's modification time, refreshed on every hit) are removed until the total fits.
//! - **Never in the way**: an unreadable, truncated or corrupt entry is removed and treated as a
//!   miss; a cache that cannot be read or written (full disk, read-only directory, the web build)
//!   only means the voice synthesizes. Nothing here fails a synthesis.

#![cfg_attr(not(test), deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable))]

use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use sha2::{Digest, Sha256};

use crate::{Audio, CLONE_PREFIX, MAX_SECONDS, Params, TtsError, catalog, formant};

/// Bump when the entry layout, or what any voice makes of the same inputs, changes: every older
/// entry then misses (and ages out under the cap).
pub const FORMAT_VERSION: u32 = 1;
/// Default size cap of the cache directory.
pub const DEFAULT_CAP_BYTES: u64 = 2 * 1024 * 1024 * 1024;
/// Directory name under the app data directory.
pub const DIR_NAME: &str = "narration-cache";

const MAGIC: &[u8; 4] = b"FCNC";
const HEADER: usize = 28;
const EXT: &str = "fcnc";
/// Highest sample rate an entry may claim.
const MAX_RATE: u32 = 384_000;
/// Files looked at when trimming to the cap (a cache holds hundreds, not millions).
const MAX_SCAN: usize = 100_000;
/// A temporary file older than this was left by a writer that died: removed when trimming.
const STALE_TMP: Duration = Duration::from_secs(3600);

/// What, besides the script, pitch and pace, decides a voice's samples: its id, the pinned model
/// files it speaks with, and for a cloned voice its profile (id, version, creation time, base
/// voice, a hash of its tone colours). `None` when there is no such voice, or (a cloned voice
/// without the `kokoro` feature or its profile) nothing stable to key on: such narrations are
/// not cached.
pub fn voice_identity(id: &str, models_dir: Option<&Path>) -> Option<String> {
    if id.starts_with(CLONE_PREFIX) {
        return clone_identity(id, models_dir?);
    }
    if catalog::find(id).is_some() {
        return Some(format!("{id}|neural|{}", pins(catalog::FILES)));
    }
    formant::VOICES.iter().any(|v| v.info.id == id).then(|| format!("{id}|formant"))
}

/// Version of the Malayalam front end (`crate::malayalam`) and voice: bumped when either reads
/// the same script differently, so old Malayalam entries miss.
pub const MALAYALAM_VERSION: u32 = 1;

/// [`voice_identity`] of the voice `id` speaking `language`: English is exactly
/// [`voice_identity`] (so English entries keep their keys); Malayalam is the cloned voice's
/// profile and voice sample, its Malayalam reference (clip and transcript), the pinned IndicF5
/// files and [`MALAYALAM_VERSION`]. `None` when nothing stable can be keyed on (no such clone,
/// or its Malayalam reference not made yet).
pub fn voice_identity_for(id: &str, language: &str, models_dir: Option<&Path>) -> Option<String> {
    if !crate::malayalam::is_tag(language) {
        return voice_identity(id, models_dir);
    }
    malayalam_identity(id, models_dir?)
}

#[cfg(feature = "kokoro")]
fn malayalam_identity(id: &str, models_dir: &Path) -> Option<String> {
    let voices = crate::clone::voices_dir(models_dir);
    let p = crate::clone::load_profile(&voices, id).ok()?;
    let r = crate::clone::cross_reference(models_dir, id).ok()??;
    let mut h = Sha256::new();
    h.update(r.transcript.as_bytes());
    Some(format!(
        "{}|clone v{}|created {}|sample {}|indicf5 ref {}–{} {}|{}|{} v{MALAYALAM_VERSION}",
        p.id,
        p.version,
        p.created,
        r.sample,
        r.from.to_bits(),
        r.to.to_bits(),
        hex(&h.finalize()),
        pins(catalog::INDICF5.files),
        crate::malayalam::TAG,
    ))
}

#[cfg(not(feature = "kokoro"))]
fn malayalam_identity(_id: &str, _models_dir: &Path) -> Option<String> {
    None
}

#[cfg(feature = "kokoro")]
fn clone_identity(id: &str, models_dir: &Path) -> Option<String> {
    let voices = crate::clone::voices_dir(models_dir);
    let p = crate::clone::load_profile(&voices, id).ok()?;
    if let Some(n) = &p.natural {
        // a natural clone speaks from its voice sample: the sample's bytes, its transcript and the
        // generation settings decide the samples
        let pkg = catalog::voxcpm(&n.model)?;
        let path = voices.join(&n.clip);
        if std::fs::metadata(&path).ok()?.len() > crate::wav::MAX_BYTES {
            return None;
        }
        let mut h = Sha256::new();
        h.update(std::fs::read(&path).ok()?);
        h.update([0]);
        h.update(n.transcript.as_bytes());
        return Some(format!(
            "{}|clone v{}|created {}|{}|{:?}|cfg {}|steps {}|denoise {}|sample {}|{}",
            p.id,
            p.version,
            p.created,
            n.model,
            n.mode,
            n.cfg.to_bits(),
            n.timesteps,
            n.denoise_db.to_bits(),
            hex(&h.finalize()),
            pins(pkg.files)
        ));
    }
    let mut h = Sha256::new();
    for v in p.source_colour.iter().chain(&p.target_colour) {
        h.update(v.to_le_bytes());
    }
    Some(format!(
        "{}|clone v{}|created {}|base {}|colour {}|{}|{}",
        p.id,
        p.version,
        p.created,
        p.base_voice,
        hex(&h.finalize()),
        pins(catalog::FILES),
        pins(catalog::CLONE_FILES)
    ))
}

#[cfg(not(feature = "kokoro"))]
fn clone_identity(_id: &str, _models_dir: &Path) -> Option<String> {
    None
}

/// A short digest of a package's pinned file hashes: a new model revision is a new identity.
fn pins(files: &[catalog::ModelFile]) -> String {
    let mut h = Sha256::new();
    for f in files {
        h.update(f.name.as_bytes());
        h.update([0]);
        h.update(f.sha256.as_bytes());
        h.update([0]);
    }
    hex(&h.finalize()).chars().take(16).collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// FNV-1a (64-bit) over the sample bytes: catches flipped bits and torn writes.
fn checksum(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(*b)).wrapping_mul(0x0000_0100_0000_01b3))
}

/// Synthesized narrations on disk under one directory (see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskCache {
    dir: PathBuf,
    cap: u64,
}

/// Unique temporary names within this process.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

impl DiskCache {
    /// A cache in `dir` (made on the first write) holding at most `cap_bytes`.
    pub fn new(dir: impl Into<PathBuf>, cap_bytes: u64) -> Self {
        Self { dir: dir.into(), cap: cap_bytes }
    }

    /// The cache in `<data_dir>/narration-cache` with the default cap.
    pub fn in_data_dir(data_dir: &Path) -> Self {
        Self::new(data_dir.join(DIR_NAME), DEFAULT_CAP_BYTES)
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    pub fn cap(&self) -> u64 {
        self.cap
    }

    /// The key of `text` spoken by the voice with `identity` ([`voice_identity`]) at `params`.
    pub fn key(identity: &str, text: &str, params: &Params) -> String {
        let mut h = Sha256::new();
        h.update(FORMAT_VERSION.to_le_bytes());
        // length-prefixed, so no two different inputs read the same
        for part in [identity.as_bytes(), text.as_bytes()] {
            h.update((part.len() as u64).to_le_bytes());
            h.update(part);
        }
        h.update(params.semitones.to_bits().to_le_bytes());
        h.update(params.pace.to_bits().to_le_bytes());
        // only a styled reading adds to the key, so the plain ones keep their entries
        if let Some(style) = params.style {
            h.update(b"style");
            h.update((style.len() as u64).to_le_bytes());
            h.update(style.as_bytes());
        }
        if params.take != 0 {
            h.update(b"take");
            h.update(params.take.to_le_bytes());
        }
        hex(&h.finalize())
    }

    /// The entry file of `key`; `None` for anything but a key this module made (64 hex digits).
    fn path(&self, key: &str) -> Option<PathBuf> {
        (key.len() == 64 && key.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))).then(|| self.dir.join(format!("{key}.{EXT}")))
    }

    /// The audio cached under `key`, its last use refreshed. A damaged entry is removed.
    pub fn get(&self, key: &str) -> Option<Audio> {
        if cfg!(target_arch = "wasm32") {
            return None;
        }
        let path = self.path(key)?;
        let mut f = fs::File::open(&path).ok()?;
        let len = f.metadata().ok()?.len();
        match read_entry(&mut f, len) {
            Some(a) => {
                drop(f);
                touch(&path);
                Some(a)
            }
            None => {
                drop(f);
                let _ = fs::remove_file(&path);
                None
            }
        }
    }

    /// Keep `audio` under `key` (atomically), then trim the cache to its cap. Failures are
    /// ignored: the cache is an optimisation.
    pub fn put(&self, key: &str, audio: &Audio) {
        if cfg!(target_arch = "wasm32") {
            return;
        }
        let Some(path) = self.path(key) else { return };
        let Some(bytes) = encode(audio) else { return };
        if bytes.len() as u64 > self.cap || fs::create_dir_all(&self.dir).is_err() {
            return;
        }
        let tmp = self.dir.join(format!("{key}.{}-{}.tmp", std::process::id(), TMP_SEQ.fetch_add(1, Ordering::Relaxed)));
        let written = fs::File::create(&tmp).and_then(|mut f| f.write_all(&bytes));
        if written.is_err() || fs::rename(&tmp, &path).is_err() {
            let _ = fs::remove_file(&tmp);
            return;
        }
        self.trim();
    }

    /// The cached audio for `key`, else `synthesize()`'s (then cached). The flag is true on a
    /// hit. Only the synthesis can fail.
    pub fn get_or_synthesize(&self, key: &str, synthesize: impl FnOnce() -> Result<Audio, TtsError>) -> Result<(Audio, bool), TtsError> {
        if let Some(a) = self.get(key) {
            return Ok((a, true));
        }
        let a = synthesize()?;
        self.put(key, &a);
        Ok((a, false))
    }

    /// Entries and their total size in bytes.
    pub fn usage(&self) -> (usize, u64) {
        let entries = self.entries();
        (entries.len(), entries.iter().map(|e| e.1).sum())
    }

    /// `(path, bytes, last use)` of every entry; stale temporary files are removed on the way.
    fn entries(&self) -> Vec<(PathBuf, u64, SystemTime)> {
        let Ok(rd) = fs::read_dir(&self.dir) else { return Vec::new() };
        let now = SystemTime::now();
        let mut out = Vec::new();
        for e in rd.filter_map(|e| e.ok()).take(MAX_SCAN) {
            let path = e.path();
            let Ok(meta) = e.metadata() else { continue };
            let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
            match path.extension().and_then(|x| x.to_str()) {
                Some(EXT) if meta.is_file() => out.push((path, meta.len(), modified)),
                Some("tmp") if now.duration_since(modified).is_ok_and(|age| age > STALE_TMP) => {
                    let _ = fs::remove_file(&path);
                }
                _ => {}
            }
        }
        out
    }

    /// Remove the least recently used entries until the cache fits its cap.
    fn trim(&self) {
        let mut entries = self.entries();
        let mut total: u64 = entries.iter().map(|e| e.1).fold(0u64, u64::saturating_add);
        if total <= self.cap {
            return;
        }
        entries.sort_by(|a, b| a.2.cmp(&b.2).then_with(|| a.0.cmp(&b.0)));
        for (path, len, _) in entries {
            if total <= self.cap {
                break;
            }
            if fs::remove_file(&path).is_ok() {
                total = total.saturating_sub(len);
            }
        }
    }
}

/// Mark an entry as just used (LRU order; access times are unreliable, so the modification time).
fn touch(path: &Path) {
    if let Ok(f) = fs::File::options().write(true).open(path) {
        let _ = f.set_modified(SystemTime::now());
    }
}

/// Most samples an entry may hold at `rate` (twice the longest narration, as the engine allows).
fn max_samples(rate: u32) -> u64 {
    (MAX_SECONDS * 2.0) as u64 * u64::from(rate)
}

fn encode(a: &Audio) -> Option<Vec<u8>> {
    if a.sample_rate == 0 || a.sample_rate > MAX_RATE || a.samples.len() as u64 > max_samples(a.sample_rate) {
        return None;
    }
    let payload = a.samples.len().checked_mul(4)?;
    let mut out = Vec::new();
    out.try_reserve_exact(payload.checked_add(HEADER)?).ok()?;
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&a.sample_rate.to_le_bytes());
    out.extend_from_slice(&(a.samples.len() as u64).to_le_bytes());
    out.extend_from_slice(&[0; 8]); // checksum, below
    for s in &a.samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    let sum = checksum(out.get(HEADER..)?);
    out.get_mut(20..HEADER)?.copy_from_slice(&sum.to_le_bytes());
    Some(out)
}

/// Read and check an entry of `len` bytes; `None` if anything about it is off.
fn read_entry(f: &mut fs::File, len: u64) -> Option<Audio> {
    let mut head = [0u8; HEADER];
    f.read_exact(&mut head).ok()?;
    let u32_at = |o: usize| head.get(o..o + 4).and_then(|b| b.try_into().ok()).map(u32::from_le_bytes);
    let u64_at = |o: usize| head.get(o..o + 8).and_then(|b| b.try_into().ok()).map(u64::from_le_bytes);
    if head.get(..4)? != MAGIC || u32_at(4)? != FORMAT_VERSION {
        return None;
    }
    let rate = u32_at(8)?;
    let n = u64_at(12)?;
    let sum = u64_at(20)?;
    if rate == 0 || rate > MAX_RATE || n > max_samples(rate) || n.checked_mul(4)?.checked_add(HEADER as u64)? != len {
        return None;
    }
    let mut bytes = Vec::new();
    bytes.try_reserve_exact(usize::try_from(n.checked_mul(4)?).ok()?).ok()?;
    f.take(n.checked_mul(4)?).read_to_end(&mut bytes).ok()?;
    if bytes.len() as u64 != n.checked_mul(4)? || checksum(&bytes) != sum {
        return None;
    }
    let (chunks, _) = bytes.as_chunks::<4>();
    let samples: Vec<f32> = chunks.iter().map(|c| f32::from_le_bytes(*c)).collect();
    if samples.iter().any(|s| !s.is_finite()) {
        return None;
    }
    Some(Audio { samples, sample_rate: rate })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Voice, VoiceInfo};
    use std::sync::atomic::AtomicUsize;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-tts-cache-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    /// A voice that counts how often it is asked to speak.
    struct Counting {
        calls: AtomicUsize,
    }

    impl Voice for Counting {
        fn info(&self) -> VoiceInfo {
            crate::voices()[0]
        }
        fn synthesize(&self, text: &str, p: &Params) -> Result<Audio, TtsError> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let n = text.len() * 100;
            Ok(Audio { samples: (0..n).map(|i| (i as f32 * 0.01 * p.pace as f32).sin() * 0.5).collect(), sample_rate: crate::SAMPLE_RATE })
        }
    }

    fn speak(c: &DiskCache, v: &Counting, id: &str, text: &str, p: Params) -> (Audio, bool) {
        let key = DiskCache::key(&voice_identity(id, None).unwrap(), text, &p);
        c.get_or_synthesize(&key, || v.synthesize(text, &p)).unwrap()
    }

    #[test]
    fn a_second_reading_is_served_from_disk_and_changes_miss() {
        let dir = tmp("hit");
        let v = Counting { calls: AtomicUsize::new(0) };
        let p = Params::default();
        let (a, hit) = speak(&DiskCache::new(&dir, DEFAULT_CAP_BYTES), &v, "basic-male", "Hello there.", p);
        assert!(!hit);
        // a new cache over the same directory (a later session): no synthesis, the same samples
        let c = DiskCache::new(&dir, DEFAULT_CAP_BYTES);
        let (b, hit) = speak(&c, &v, "basic-male", "Hello there.", p);
        assert!(hit);
        assert_eq!(v.calls.load(Ordering::Relaxed), 1);
        assert_eq!(a, b, "bit-exact");
        // another pace, pitch, voice or script misses
        speak(&c, &v, "basic-male", "Hello there.", Params { pace: 1.1, ..p });
        speak(&c, &v, "basic-male", "Hello there.", Params { semitones: 2.0, ..p });
        speak(&c, &v, "basic-female", "Hello there.", p);
        speak(&c, &v, "basic-male", "Hello there!", p);
        assert_eq!(v.calls.load(Ordering::Relaxed), 5);
        assert_eq!(c.usage().0, 5);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn identities_cover_the_model_and_refuse_unknown_voices() {
        assert_eq!(voice_identity("basic-male", None).as_deref(), Some("basic-male|formant"));
        let k = voice_identity("kokoro-heart", None).unwrap();
        assert!(k.starts_with("kokoro-heart|neural|") && k.len() > 30, "{k}");
        assert_ne!(voice_identity("kokoro-bella", None), voice_identity("kokoro-heart", None));
        assert_eq!(voice_identity("nope", None), None);
        // a cloned voice needs its profile
        assert_eq!(voice_identity("clone:me", None), None);
        assert_eq!(voice_identity("clone:me", Some(&tmp("noprofile"))), None);
        // keys are fixed-length hex and differ per input
        let p = Params::default();
        let a = DiskCache::key("x", "ab", &p);
        assert_eq!(a.len(), 64);
        assert_ne!(a, DiskCache::key("xa", "b", &p), "length-prefixed parts");
        assert_ne!(a, DiskCache::key("x", "ab", &Params { pace: 1.0 + f64::EPSILON, ..p }));
        // a style is part of the key; two styles differ
        let warm = DiskCache::key("x", "ab", &Params { style: Some("warm"), ..p });
        assert_ne!(a, warm);
        assert_ne!(warm, DiskCache::key("x", "ab", &Params { style: Some("calm"), ..p }));
        assert_ne!(a, DiskCache::key("x", "ab", &Params { take: 1, ..p }), "another take is another reading");
    }

    #[test]
    fn damaged_entries_are_removed_and_resynthesized() {
        let dir = tmp("corrupt");
        let c = DiskCache::new(&dir, DEFAULT_CAP_BYTES);
        let v = Counting { calls: AtomicUsize::new(0) };
        let p = Params::default();
        let (good, _) = speak(&c, &v, "basic-male", "Damage me.", p);
        let key = DiskCache::key(&voice_identity("basic-male", None).unwrap(), "Damage me.", &p);
        let path = c.path(&key).unwrap();
        let bytes = fs::read(&path).unwrap();
        let mut flipped = bytes.clone();
        let last = flipped.len() - 1;
        flipped[last] ^= 0x40;
        let mut nan = bytes.clone();
        nan[HEADER..HEADER + 4].copy_from_slice(&f32::NAN.to_le_bytes());
        let mut huge = bytes.clone();
        huge[12..20].copy_from_slice(&u64::MAX.to_le_bytes());
        let mut rate0 = bytes.clone();
        rate0[8..12].copy_from_slice(&0u32.to_le_bytes());
        let damaged: Vec<Vec<u8>> = vec![
            Vec::new(),
            b"FCNC".to_vec(),
            bytes[..bytes.len() / 2].to_vec(),
            flipped,
            nan,
            huge,
            rate0,
            [b"XXXX", &bytes[4..]].concat(),
            [&bytes[..], &[0u8; 4]].concat(),
        ];
        for (i, d) in damaged.iter().enumerate() {
            fs::write(&path, d).unwrap();
            assert_eq!(c.get(&key), None, "case {i}");
            assert!(!path.exists(), "case {i}: the damaged entry is removed");
        }
        // and the next reading synthesizes again, then hits
        let calls = v.calls.load(Ordering::Relaxed);
        let (again, hit) = speak(&c, &v, "basic-male", "Damage me.", p);
        assert!(!hit && again == good);
        assert_eq!(v.calls.load(Ordering::Relaxed), calls + 1);
        assert!(speak(&c, &v, "basic-male", "Damage me.", p).1);
        // keys that are not ours never touch the file system
        assert_eq!(c.get("../../etc/passwd"), None);
        c.put("../x", &good);
        assert_eq!(c.usage().0, 1);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_cap_evicts_the_least_recently_used() {
        let dir = tmp("cap");
        let v = Counting { calls: AtomicUsize::new(0) };
        let p = Params::default();
        // each entry is 28 + 4 × 1000 bytes (ten-character scripts)
        let entry = (HEADER + 4000) as u64;
        let c = DiskCache::new(&dir, entry * 3);
        let keys: Vec<String> = ["0123456789", "abcdefghij", "klmnopqrst"]
            .iter()
            .map(|t| {
                speak(&c, &v, "basic-male", t, p);
                DiskCache::key(&voice_identity("basic-male", None).unwrap(), t, &p)
            })
            .collect();
        assert_eq!(c.usage(), (3, entry * 3));
        // the first was written longest ago, then used most recently: the second is the oldest
        let past = |s: u64| SystemTime::now() - Duration::from_secs(s);
        for (k, age) in keys.iter().zip([300, 200, 100]) {
            fs::File::options().write(true).open(c.path(k).unwrap()).unwrap().set_modified(past(age)).unwrap();
        }
        assert!(c.get(&keys[0]).is_some());
        speak(&c, &v, "basic-male", "uvwxyzABCD", p);
        assert_eq!(c.usage(), (3, entry * 3));
        assert!(c.path(&keys[0]).unwrap().exists(), "recently used: kept");
        assert!(!c.path(&keys[1]).unwrap().exists(), "least recently used: evicted");
        assert!(c.path(&keys[2]).unwrap().exists());
        // an entry larger than the whole cap is never written
        let small = DiskCache::new(tmp("tiny"), 100);
        speak(&small, &v, "basic-male", "too long for the cap", p);
        assert_eq!(small.usage().0, 0);
        // stale temporary files from a writer that died are cleared when trimming
        let stale = dir.join("deadbeef.1-1.tmp");
        fs::write(&stale, b"x").unwrap();
        fs::File::options().write(true).open(&stale).unwrap().set_modified(past(7200)).unwrap();
        speak(&c, &v, "basic-male", "EFGHIJKLMN", p);
        assert!(!stale.exists());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn an_unwritable_cache_never_fails_a_synthesis() {
        // the cache "directory" is a file: nothing can be written there
        let file = tmp("blocked");
        fs::write(&file, b"not a directory").unwrap();
        let c = DiskCache::new(file.join("sub"), DEFAULT_CAP_BYTES);
        let v = Counting { calls: AtomicUsize::new(0) };
        let (a, hit) = speak(&c, &v, "basic-male", "Still speaks.", Params::default());
        assert!(!hit && !a.samples.is_empty());
        assert_eq!(c.usage(), (0, 0));
        // a failing voice fails as itself, not because of the cache
        let e = c.get_or_synthesize(&"0".repeat(64), || Err(TtsError::Empty)).unwrap_err();
        assert_eq!(e, TtsError::Empty);
        let _ = fs::remove_file(&file);
    }
}
