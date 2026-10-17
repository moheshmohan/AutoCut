//! A structural check of a SoundFont 2 file, run before rustysynth reads it.
//!
//! rustysynth trusts its input. Reading a file it walks the RIFF chunks by their declared sizes,
//! allocates what a size says before it reads, casts the sample chunk to 16-bit samples (sound
//! only when its size is even), slices zone lists by the file's indices and adds sample offsets in
//! 32-bit arithmetic. This module walks the file exactly as rustysynth will, from the bytes in
//! memory, and refuses anything that would take it out of bounds:
//!
//! - `RIFF`/`sfbk` with the `INFO`, `sdta` and `pdta` lists in that order, each inside the file;
//! - every sub-chunk inside its list, the sub-chunks tiling the list exactly, every size even (the
//!   SoundFont 2 rule; rustysynth also skips no pad bytes, so an odd size would desynchronise it);
//! - `ifil`/`iver` exactly 4 bytes (rustysynth reads 4 whatever the size says), texts at most
//!   64 KiB, only the chunk ids rustysynth knows;
//! - a sample chunk of at least two samples; every record list (`phdr` 38, `pbag`/`pgen`/`ibag`/
//!   `igen` 4, `inst` 22, `shdr` 46 bytes) whole, with its terminal record;
//! - preset and instrument zone ranges increasing and inside the bag lists, bag generator indices
//!   increasing and inside the generator lists;
//! - every sample's start, end and loop points inside the sample data (so offsets added to them
//!   cannot overflow).
//!
//! Generator values (sample and instrument ids, address offsets) are left to rustysynth, which
//! refuses out-of-range ones with an error.

use crate::Error;

/// Largest SoundFont read, bytes (GeneralUser GS is 32 MB; the largest General MIDI banks are
/// around 1 GB).
pub const MAX_FILE_BYTES: u64 = 1 << 30;
/// Longest `INFO` text (the specification allows 65 536 bytes for comments, plus padding).
const MAX_TEXT: usize = 65_538;
/// Most sub-chunks in a list.
const MAX_CHUNKS: usize = 4096;

/// What the check read from a file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    /// `INAM`.
    pub name: String,
    /// `ifil` (major, minor).
    pub version: (u16, u16),
    /// 16-bit samples in the sample chunk.
    pub samples: usize,
    /// (bank, program, name) of every preset.
    pub presets: Vec<(u16, u16, String)>,
    pub instruments: usize,
    pub sample_headers: usize,
}

fn bad(msg: impl Into<String>) -> Error {
    Error(format!("not a valid SoundFont: {}", msg.into()))
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    let s = b.get(at..at.checked_add(2)?)?;
    Some(u16::from_le_bytes([*s.first()?, *s.get(1)?]))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    let s = b.get(at..at.checked_add(4)?)?;
    Some(u32::from_le_bytes([*s.first()?, *s.get(1)?, *s.get(2)?, *s.get(3)?]))
}

fn i32_at(b: &[u8], at: usize) -> Option<i32> {
    u32_at(b, at).map(|v| i32::from_le_bytes(v.to_le_bytes()))
}

fn id_at(b: &[u8], at: usize) -> Option<[u8; 4]> {
    let s = b.get(at..at.checked_add(4)?)?;
    Some([*s.first()?, *s.get(1)?, *s.get(2)?, *s.get(3)?])
}

fn show(id: [u8; 4]) -> String {
    id.iter().map(|b| if b.is_ascii_graphic() || *b == b' ' { char::from(*b) } else { '?' }).collect()
}

/// A text field, as rustysynth reads it: up to the first NUL, anything but printable ASCII as `?`.
fn text(b: &[u8]) -> String {
    b.iter().take_while(|c| **c != 0).map(|c| if (9..=126).contains(c) { char::from(*c) } else { '?' }).collect()
}

/// The list at `at` (`LIST`, size, type): its type and its body after the type, and where the
/// next top-level chunk starts.
fn list<'a>(b: &'a [u8], at: usize, want: &[u8; 4]) -> Result<(&'a [u8], usize), Error> {
    let what = show(*want);
    if id_at(b, at) != Some(*b"LIST") {
        return Err(bad(format!("the {what} list is missing")));
    }
    let size = u32_at(b, at.saturating_add(4)).ok_or_else(|| bad(format!("the {what} list is cut short")))? as usize;
    if size < 4 {
        return Err(bad(format!("the {what} list is empty")));
    }
    let start = at.checked_add(8).ok_or_else(|| bad("too large"))?;
    let end = start.checked_add(size).ok_or_else(|| bad("too large"))?;
    let body = b.get(start..end).ok_or_else(|| bad(format!("the {what} list runs past the end of the file")))?;
    if id_at(body, 0) != Some(*want) {
        return Err(bad(format!("expected the {what} list, found {}", id_at(body, 0).map(show).unwrap_or_default())));
    }
    Ok((body.get(4..).unwrap_or(&[]), end))
}

/// The sub-chunks of a list body: they tile it exactly and every size is even.
fn chunks<'a>(body: &'a [u8], what: &str) -> Result<Vec<([u8; 4], &'a [u8])>, Error> {
    let mut out = Vec::new();
    let mut at = 0usize;
    while at < body.len() {
        if out.len() >= MAX_CHUNKS {
            return Err(bad(format!("the {what} list has too many chunks")));
        }
        let id = id_at(body, at).ok_or_else(|| bad(format!("a chunk header in the {what} list is cut short")))?;
        let size = u32_at(body, at.saturating_add(4)).ok_or_else(|| bad(format!("a chunk header in the {what} list is cut short")))? as usize;
        if !size.is_multiple_of(2) {
            return Err(bad(format!("the {} chunk has an odd size ({size} bytes)", show(id))));
        }
        let start = at.checked_add(8).ok_or_else(|| bad("too large"))?;
        let end = start.checked_add(size).ok_or_else(|| bad("too large"))?;
        let data = body.get(start..end).ok_or_else(|| bad(format!("the {} chunk runs past the end of its list", show(id))))?;
        out.push((id, data));
        at = end;
    }
    Ok(out)
}

/// A record list of `rec`-byte records with its terminal record: the number of records (with the
/// terminal one), at least `min`.
fn records(data: &[u8], rec: usize, min: usize, what: &str) -> Result<usize, Error> {
    if data.is_empty() || !data.len().is_multiple_of(rec) {
        return Err(bad(format!("the {what} list is not whole {rec}-byte records")));
    }
    let n = data.len() / rec;
    if n < min {
        return Err(bad(format!("the {what} list has too few records")));
    }
    Ok(n)
}

/// Zone ranges of presets or instruments (`starts`, one per record with the terminal one) over
/// `zones` bags: each a non-empty range inside the bag list.
fn zone_ranges(starts: &[u16], zones: usize, what: &str) -> Result<(), Error> {
    for (i, w) in starts.windows(2).enumerate() {
        let (a, b) = match w {
            [a, b] => (usize::from(*a), usize::from(*b)),
            _ => continue,
        };
        if b <= a {
            return Err(bad(format!("{what} {i} has no zones")));
        }
        if b > zones {
            return Err(bad(format!("the zones of {what} {i} run past the zone list")));
        }
    }
    Ok(())
}

/// Bag generator indices: increasing and inside the generator list (`gens` generators without the
/// terminal one).
fn bag_generators(bags: &[u8], gens: usize, what: &str) -> Result<(), Error> {
    let idx: Vec<usize> = (0..bags.len() / 4).filter_map(|j| u16_at(bags, j * 4).map(usize::from)).collect();
    for w in idx.windows(2) {
        if let [a, b] = w
            && (b < a || *b > gens)
        {
            return Err(bad(format!("the {what} zones point outside their generators")));
        }
    }
    Ok(())
}

/// Check `b` as rustysynth will read it. On success, what the file holds.
pub fn check(b: &[u8]) -> Result<Summary, Error> {
    if b.len() as u64 > MAX_FILE_BYTES {
        return Err(bad("the file is too large"));
    }
    if id_at(b, 0) != Some(*b"RIFF") || id_at(b, 8) != Some(*b"sfbk") {
        return Err(bad("it does not start with RIFF/sfbk"));
    }
    let mut s = Summary::default();

    // INFO
    let (info, next) = list(b, 12, b"INFO")?;
    let mut have_ifil = false;
    for (id, data) in chunks(info, "INFO")? {
        match &id {
            b"ifil" | b"iver" => {
                if data.len() != 4 {
                    return Err(bad(format!("the {} chunk must be 4 bytes", show(id))));
                }
                if &id == b"ifil" {
                    have_ifil = true;
                    s.version = (u16_at(data, 0).unwrap_or(0), u16_at(data, 2).unwrap_or(0));
                }
            }
            b"isng" | b"INAM" | b"irom" | b"ICRD" | b"IENG" | b"IPRD" | b"ICOP" | b"ICMT" | b"ISFT" => {
                if data.len() > MAX_TEXT {
                    return Err(bad(format!("the {} text is too long", show(id))));
                }
                if &id == b"INAM" {
                    s.name = text(data);
                }
            }
            _ => return Err(bad(format!("unknown INFO chunk {}", show(id)))),
        }
    }
    if !have_ifil {
        return Err(bad("the version (ifil) is missing"));
    }

    // sdta
    let (sdta, next) = list(b, next, b"sdta")?;
    let mut smpl: Option<&[u8]> = None;
    for (id, data) in chunks(sdta, "sdta")? {
        match &id {
            b"smpl" => smpl = Some(data),
            b"sm24" => {}
            _ => return Err(bad(format!("unknown sample chunk {}", show(id)))),
        }
    }
    let smpl = smpl.ok_or_else(|| bad("there are no samples"))?;
    if smpl.len() < 4 {
        return Err(bad("there are no samples"));
    }
    if smpl.get(..4) == Some(b"OggS") {
        return Err(Error("compressed SoundFonts (SF3) are not supported; use an .sf2 file".into()));
    }
    s.samples = smpl.len() / 2;

    // pdta
    let (pdta, _) = list(b, next, b"pdta")?;
    let mut get = std::collections::HashMap::new();
    for (id, data) in chunks(pdta, "pdta")? {
        match &id {
            b"phdr" | b"pbag" | b"pmod" | b"pgen" | b"inst" | b"ibag" | b"imod" | b"igen" | b"shdr" => {
                // as rustysynth does, a repeated chunk replaces the earlier one
                get.insert(id, data);
            }
            _ => return Err(bad(format!("unknown preset-data chunk {}", show(id)))),
        }
    }
    let need = |id: &[u8; 4]| get.get(id).copied().ok_or_else(|| bad(format!("the {} chunk is missing", show(*id))));
    let phdr = need(b"phdr")?;
    let pbag = need(b"pbag")?;
    let pgen = need(b"pgen")?;
    let inst = need(b"inst")?;
    let ibag = need(b"ibag")?;
    let igen = need(b"igen")?;
    let shdr = need(b"shdr")?;
    let n_presets = records(phdr, 38, 2, "preset")?;
    let n_pbag = records(pbag, 4, 2, "preset zone")?;
    let n_pgen = records(pgen, 4, 1, "preset generator")?;
    let n_inst = records(inst, 22, 2, "instrument")?;
    let n_ibag = records(ibag, 4, 2, "instrument zone")?;
    let n_igen = records(igen, 4, 1, "instrument generator")?;
    let n_shdr = records(shdr, 46, 1, "sample header")?;

    let preset_starts: Vec<u16> = (0..n_presets).map(|i| u16_at(phdr, i * 38 + 24).unwrap_or(0)).collect();
    zone_ranges(&preset_starts, n_pbag - 1, "preset")?;
    bag_generators(pbag, n_pgen - 1, "preset")?;
    let inst_starts: Vec<u16> = (0..n_inst).map(|i| u16_at(inst, i * 22 + 20).unwrap_or(0)).collect();
    zone_ranges(&inst_starts, n_ibag - 1, "instrument")?;
    bag_generators(ibag, n_igen - 1, "instrument")?;

    let wave = i64::try_from(s.samples).unwrap_or(i64::MAX);
    for h in 0..n_shdr - 1 {
        let at = h * 46 + 20;
        let field = |o: usize| i32_at(shdr, at + o).map(i64::from).unwrap_or(-1);
        let (start, end, loop_start, loop_end) = (field(0), field(4), field(8), field(12));
        let inside = |x: i64| (0..=wave).contains(&x);
        if !(inside(start) && inside(end) && inside(loop_start) && inside(loop_end)) || end < start {
            return Err(bad(format!("sample {} (\"{}\") points outside the sample data", h, text(shdr.get(h * 46..h * 46 + 20).unwrap_or(&[])))));
        }
    }

    s.presets = (0..n_presets - 1)
        .map(|i| {
            let r = i * 38;
            (u16_at(phdr, r + 22).unwrap_or(0), u16_at(phdr, r + 20).unwrap_or(0), text(phdr.get(r..r + 20).unwrap_or(&[])))
        })
        .collect();
    s.instruments = n_inst - 1;
    s.sample_headers = n_shdr - 1;
    Ok(s)
}
