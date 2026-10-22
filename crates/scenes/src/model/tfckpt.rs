//! A strict reader of TensorFlow 2 checkpoints (`variables.index` + `variables.data-*`), enough
//! for TransNetV2's published weights. From the public format descriptions:
//!
//! - the index is a LevelDB-style sorted table: a 48-byte footer (the metaindex and index block
//!   handles, then the magic `0xdb4775248b80fb57`), an index block of handles to data blocks,
//!   data blocks of prefix-compressed key/value entries with a restart array, each block followed
//!   by a one-byte compression type (only "none" is read) and a CRC;
//! - each key names a tensor, each value is a `BundleEntryProto` (dtype, shape, shard, offset,
//!   size) locating its bytes in the data file;
//! - the key `_CHECKPOINTABLE_OBJECT_GRAPH` holds a string tensor: the `TrackableObjectGraph`
//!   whose nodes carry, for each variable, its full TensorFlow name and its checkpoint key — so a
//!   tensor is found by its layer's name (`TransNet/SDDCNN_1/DDCNN_1/bn/gamma`), never by
//!   position.
//!
//! Everything is bounds-checked; sizes and counts are capped; damaged files are errors.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use super::{R, err};

/// Largest index file read.
const MAX_INDEX: u64 = 16 << 20;
/// Most entries in the index, and nodes in the object graph.
const MAX_ENTRIES: usize = 100_000;
/// Largest object graph read.
const MAX_GRAPH: u64 = 64 << 20;
/// Largest tensor read (elements).
const MAX_ELEMENTS: usize = 1 << 28;
const MAGIC: u64 = 0xdb47_7524_8b80_fb57;
const DT_FLOAT: u64 = 1;
const DT_STRING: u64 = 7;

/// Where a tensor's bytes are.
#[derive(Clone, Debug, PartialEq)]
struct Entry {
    dtype: u64,
    shape: Vec<usize>,
    offset: u64,
    size: u64,
}

/// An open checkpoint: every float tensor by its full TensorFlow name.
pub struct Checkpoint {
    data: PathBuf,
    data_len: u64,
    by_name: HashMap<String, Entry>,
}

// ---------------------------------------------------------------------------------------------
// protobuf wire format
// ---------------------------------------------------------------------------------------------

fn varint(b: &[u8], pos: &mut usize) -> Option<u64> {
    let mut v = 0u64;
    for shift in (0..64).step_by(7) {
        let c = *b.get(*pos)?;
        *pos += 1;
        v |= u64::from(c & 0x7f) << shift;
        if c < 0x80 {
            return Some(v);
        }
    }
    None
}

/// A protobuf field: its number and value (a varint, or the bytes of a length-delimited one).
enum Field<'a> {
    Int(u64),
    Bytes(&'a [u8]),
    Fixed,
}

/// The fields of a protobuf message, in order; `None` for a malformed one.
fn fields(b: &[u8]) -> Option<Vec<(u64, Field<'_>)>> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < b.len() {
        let key = varint(b, &mut pos)?;
        let (num, wire) = (key >> 3, key & 7);
        let f = match wire {
            0 => Field::Int(varint(b, &mut pos)?),
            1 => {
                pos = pos.checked_add(8).filter(|p| *p <= b.len())?;
                Field::Fixed
            }
            2 => {
                let n = usize::try_from(varint(b, &mut pos)?).ok()?;
                let end = pos.checked_add(n).filter(|e| *e <= b.len())?;
                let v = b.get(pos..end)?;
                pos = end;
                Field::Bytes(v)
            }
            5 => {
                pos = pos.checked_add(4).filter(|p| *p <= b.len())?;
                Field::Fixed
            }
            _ => return None,
        };
        out.push((num, f));
        if out.len() > MAX_ENTRIES {
            return None;
        }
    }
    Some(out)
}

fn int(f: &[(u64, Field<'_>)], num: u64) -> Option<u64> {
    f.iter().rev().find_map(|(n, v)| match v {
        Field::Int(x) if *n == num => Some(*x),
        _ => None,
    })
}

fn bytes<'a>(f: &[(u64, Field<'a>)], num: u64) -> Option<&'a [u8]> {
    f.iter().rev().find_map(|(n, v)| match v {
        Field::Bytes(x) if *n == num => Some(*x),
        _ => None,
    })
}

// ---------------------------------------------------------------------------------------------
// the table
// ---------------------------------------------------------------------------------------------

/// A block handle at `pos`: (offset, size).
fn handle(b: &[u8], pos: &mut usize) -> Option<(usize, usize)> {
    let off = usize::try_from(varint(b, pos)?).ok()?;
    let size = usize::try_from(varint(b, pos)?).ok()?;
    Some((off, size))
}

/// The contents of block `(off, size)`: checked to be inside the file, uncompressed.
fn block(file: &[u8], (off, size): (usize, usize)) -> Option<&[u8]> {
    let end = off.checked_add(size)?;
    let kind = *file.get(end)?;
    // the type byte and a 4-byte CRC follow every block
    file.get(end..end.checked_add(5)?)?;
    if kind != 0 {
        return None;
    }
    file.get(off..end)
}

/// The key/value entries of a block.
fn entries(b: &[u8]) -> Option<Vec<(Vec<u8>, &[u8])>> {
    let n = b.len();
    let restarts = u32::from_le_bytes(b.get(n.checked_sub(4)?..n)?.try_into().ok()?) as usize;
    let end = n.checked_sub(4)?.checked_sub(restarts.checked_mul(4)?)?;
    let mut pos = 0;
    let mut key: Vec<u8> = Vec::new();
    let mut out = Vec::new();
    while pos < end {
        let shared = usize::try_from(varint(b, &mut pos)?).ok()?;
        let unshared = usize::try_from(varint(b, &mut pos)?).ok()?;
        let vlen = usize::try_from(varint(b, &mut pos)?).ok()?;
        if shared > key.len() {
            return None;
        }
        key.truncate(shared);
        let kend = pos.checked_add(unshared).filter(|e| *e <= end)?;
        key.extend_from_slice(b.get(pos..kend)?);
        let vend = kend.checked_add(vlen).filter(|e| *e <= end)?;
        out.push((key.clone(), b.get(kend..vend)?));
        pos = vend;
        if out.len() > MAX_ENTRIES {
            return None;
        }
    }
    Some(out)
}

fn entry_of(v: &[u8]) -> Option<Entry> {
    let f = fields(v)?;
    let dtype = int(&f, 1).unwrap_or(0);
    let mut shape = Vec::new();
    if let Some(s) = bytes(&f, 2) {
        for (n, d) in fields(s)? {
            if let (2, Field::Bytes(dim)) = (n, d) {
                let size = int(&fields(dim)?, 1).unwrap_or(0);
                shape.push(usize::try_from(size).ok()?);
            }
        }
    }
    if shape.len() > 8 || int(&f, 3).unwrap_or(0) != 0 {
        // more than one shard is not something TransNetV2's checkpoint has
        return None;
    }
    Some(Entry { dtype, shape, offset: int(&f, 4).unwrap_or(0), size: int(&f, 5).unwrap_or(0) })
}

impl Checkpoint {
    /// Open the checkpoint `index` with its single data shard `data`.
    pub fn open(index: &Path, data: &Path) -> R<Checkpoint> {
        let bad = |what: &str| err(format!("{}: {what}", index.display()));
        let len = std::fs::metadata(index).map_err(|e| err(format!("{}: {e}", index.display())))?.len();
        if !(48..=MAX_INDEX).contains(&len) {
            return Err(bad("not a checkpoint index"));
        }
        let file = std::fs::read(index).map_err(|e| err(format!("{}: {e}", index.display())))?;
        let foot = file.len().checked_sub(48).and_then(|at| file.get(at..)).ok_or_else(|| bad("too short"))?;
        let magic = u64::from_le_bytes(foot.get(40..48).and_then(|m| m.try_into().ok()).ok_or_else(|| bad("too short"))?);
        if magic != MAGIC {
            return Err(bad("not a checkpoint index (bad magic)"));
        }
        let mut pos = 0;
        let _meta = handle(foot, &mut pos).ok_or_else(|| bad("bad footer"))?;
        let idx = handle(foot, &mut pos).ok_or_else(|| bad("bad footer"))?;
        let index_block = block(&file, idx).ok_or_else(|| bad("bad index block"))?;
        let mut raw: HashMap<String, Entry> = HashMap::new();
        for (_, h) in entries(index_block).ok_or_else(|| bad("bad index block"))? {
            let mut p = 0;
            let bh = handle(h, &mut p).ok_or_else(|| bad("bad block handle"))?;
            let b = block(&file, bh).ok_or_else(|| bad("bad or compressed data block"))?;
            for (k, v) in entries(b).ok_or_else(|| bad("bad data block"))? {
                if k.is_empty() {
                    continue; // the bundle header
                }
                let key = String::from_utf8(k).map_err(|_| bad("a key is not text"))?;
                let e = entry_of(v).ok_or_else(|| bad(&format!("bad entry for {key}")))?;
                raw.insert(key, e);
                if raw.len() > MAX_ENTRIES {
                    return Err(bad("too many entries"));
                }
            }
        }
        let data_len = std::fs::metadata(data).map_err(|e| err(format!("{}: {e}", data.display())))?.len();
        // the object graph: full variable names → checkpoint keys
        let og = raw.get("_CHECKPOINTABLE_OBJECT_GRAPH").ok_or_else(|| bad("no object graph"))?.clone();
        if og.dtype != DT_STRING || og.size > MAX_GRAPH || og.offset.saturating_add(og.size) > data_len {
            return Err(bad("bad object graph entry"));
        }
        let blob = read_range(data, og.offset, og.size)?;
        let mut p = 0;
        let n = usize::try_from(varint(&blob, &mut p).ok_or_else(|| bad("bad object graph"))?).map_err(|_| bad("bad object graph"))?;
        let start = p.checked_add(4).ok_or_else(|| bad("bad object graph"))?;
        let graph = blob.get(start..start.saturating_add(n)).filter(|g| g.len() == n).ok_or_else(|| bad("bad object graph"))?;
        let mut by_name = HashMap::new();
        for (num, node) in fields(graph).ok_or_else(|| bad("bad object graph"))? {
            let (1, Field::Bytes(node)) = (num, node) else { continue };
            for (an, attr) in fields(node).ok_or_else(|| bad("bad object graph node"))? {
                let (2, Field::Bytes(attr)) = (an, attr) else { continue };
                let af = fields(attr).ok_or_else(|| bad("bad object graph attribute"))?;
                let (Some(full), Some(key)) = (bytes(&af, 2), bytes(&af, 3)) else { continue };
                let (Ok(full), Ok(key)) = (std::str::from_utf8(full), std::str::from_utf8(key)) else { continue };
                if let Some(e) = raw.get(key).filter(|e| e.dtype == DT_FLOAT) {
                    by_name.insert(full.to_string(), e.clone());
                }
            }
        }
        if by_name.is_empty() {
            return Err(bad("no float variables"));
        }
        Ok(Checkpoint { data: data.to_path_buf(), data_len, by_name })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.by_name.keys().map(String::as_str)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.by_name.get(name).map(|e| e.shape.as_slice())
    }

    /// The float tensor `name` (row-major), checked against `shape` and for finite values.
    pub fn load(&self, name: &str, shape: &[usize]) -> R<Vec<f32>> {
        let e = self.by_name.get(name).ok_or_else(|| err(format!("TransNetV2 checkpoint: missing `{name}`")))?;
        if e.shape != shape {
            return Err(err(format!("TransNetV2 checkpoint: `{name}` is {:?}, expected {shape:?}", e.shape)));
        }
        let n = shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d)).filter(|n| *n <= MAX_ELEMENTS).ok_or_else(|| err(format!("`{name}` is too large")))?;
        if e.size != (n as u64).saturating_mul(4) || e.offset.saturating_add(e.size) > self.data_len {
            return Err(err(format!("TransNetV2 checkpoint: `{name}` has a bad size or offset")));
        }
        let b = read_range(&self.data, e.offset, e.size)?;
        let v: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        if v.len() != n || v.iter().any(|x| !x.is_finite()) {
            return Err(err(format!("TransNetV2 checkpoint: `{name}` is damaged")));
        }
        Ok(v)
    }
}

fn read_range(path: &Path, offset: u64, size: u64) -> R<Vec<u8>> {
    let mut f = File::open(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
    f.seek(SeekFrom::Start(offset)).map_err(|e| err(format!("{}: {e}", path.display())))?;
    let n = usize::try_from(size).map_err(|_| err("tensor too large"))?;
    let mut b = Vec::new();
    b.try_reserve_exact(n).map_err(|_| err("out of memory"))?;
    b.resize(n, 0);
    f.read_exact(&mut b).map_err(|e| err(format!("{}: {e}", path.display())))?;
    Ok(b)
}

/// Write a minimal checkpoint (tests): float tensors under the given full names.
#[cfg(test)]
pub(crate) fn write_test_checkpoint(dir: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> (PathBuf, PathBuf) {
    fn put_varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }
    fn put_field(out: &mut Vec<u8>, num: u64, b: &[u8]) {
        put_varint(out, num << 3 | 2);
        put_varint(out, b.len() as u64);
        out.extend_from_slice(b);
    }
    fn put_int(out: &mut Vec<u8>, num: u64, v: u64) {
        put_varint(out, num << 3);
        put_varint(out, v);
    }
    let mut data = Vec::new();
    let mut table: Vec<(String, Vec<u8>)> = Vec::new();
    let mut graph = Vec::new();
    for (i, (name, shape, v)) in tensors.iter().enumerate() {
        let key = format!("variables/{i}/.ATTRIBUTES/VARIABLE_VALUE");
        let mut e = Vec::new();
        put_int(&mut e, 1, DT_FLOAT);
        let mut sh = Vec::new();
        for d in shape {
            let mut dim = Vec::new();
            put_int(&mut dim, 1, *d as u64);
            put_field(&mut sh, 2, &dim);
        }
        put_field(&mut e, 2, &sh);
        put_int(&mut e, 4, data.len() as u64);
        put_int(&mut e, 5, (v.len() * 4) as u64);
        for x in v {
            data.extend_from_slice(&x.to_le_bytes());
        }
        table.push((key.clone(), e));
        let mut attr = Vec::new();
        put_field(&mut attr, 1, b"VARIABLE_VALUE");
        put_field(&mut attr, 2, name.as_bytes());
        put_field(&mut attr, 3, key.as_bytes());
        let mut node = Vec::new();
        put_field(&mut node, 2, &attr);
        put_field(&mut graph, 1, &node);
    }
    // the object graph as a scalar string tensor
    let og_off = data.len() as u64;
    let mut og = Vec::new();
    put_varint(&mut og, graph.len() as u64);
    og.extend_from_slice(&[0, 0, 0, 0]);
    og.extend_from_slice(&graph);
    data.extend_from_slice(&og);
    let mut e = Vec::new();
    put_int(&mut e, 1, DT_STRING);
    put_int(&mut e, 4, og_off);
    put_int(&mut e, 5, og.len() as u64);
    table.push(("_CHECKPOINTABLE_OBJECT_GRAPH".into(), e));
    table.push((String::new(), vec![8, 1]));
    table.sort();
    // one data block, no prefix compression, one restart
    let mut file = Vec::new();
    let mut blk = Vec::new();
    for (k, v) in &table {
        put_varint(&mut blk, 0);
        put_varint(&mut blk, k.len() as u64);
        put_varint(&mut blk, v.len() as u64);
        blk.extend_from_slice(k.as_bytes());
        blk.extend_from_slice(v);
    }
    blk.extend_from_slice(&0u32.to_le_bytes());
    blk.extend_from_slice(&1u32.to_le_bytes());
    let data_handle = (file.len(), blk.len());
    file.extend_from_slice(&blk);
    file.extend_from_slice(&[0, 0, 0, 0, 0]);
    let mut iblk = Vec::new();
    let mut h = Vec::new();
    put_varint(&mut h, data_handle.0 as u64);
    put_varint(&mut h, data_handle.1 as u64);
    put_varint(&mut iblk, 0);
    put_varint(&mut iblk, 1);
    put_varint(&mut iblk, h.len() as u64);
    iblk.push(b'~');
    iblk.extend_from_slice(&h);
    iblk.extend_from_slice(&0u32.to_le_bytes());
    iblk.extend_from_slice(&1u32.to_le_bytes());
    let index_handle = (file.len(), iblk.len());
    file.extend_from_slice(&iblk);
    file.extend_from_slice(&[0, 0, 0, 0, 0]);
    let mut foot = Vec::new();
    put_varint(&mut foot, 0);
    put_varint(&mut foot, 0);
    put_varint(&mut foot, index_handle.0 as u64);
    put_varint(&mut foot, index_handle.1 as u64);
    foot.resize(40, 0);
    foot.extend_from_slice(&MAGIC.to_le_bytes());
    file.extend_from_slice(&foot);
    let (ip, dp) = (dir.join("variables.index"), dir.join("variables.data-00000-of-00001"));
    std::fs::write(&ip, &file).unwrap();
    std::fs::write(&dp, &data).unwrap();
    (ip, dp)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-scenes-ckpt-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn reads_tensors_by_full_name() {
        let d = tmp("ok");
        let (ip, dp) = write_test_checkpoint(
            &d,
            &[("Net/dense/kernel", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), ("Net/dense/bias", vec![3], vec![0.5, -0.5, 0.0])],
        );
        let c = Checkpoint::open(&ip, &dp).unwrap();
        assert_eq!(c.names().count(), 2);
        assert_eq!(c.shape("Net/dense/kernel"), Some(&[2usize, 3][..]));
        assert_eq!(c.load("Net/dense/bias", &[3]).unwrap(), vec![0.5, -0.5, 0.0]);
        assert_eq!(c.load("Net/dense/kernel", &[2, 3]).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(c.load("Net/dense/kernel", &[3, 2]).is_err(), "shape is checked");
        assert!(c.load("nope", &[1]).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn damaged_checkpoints_are_errors_not_panics() {
        let d = tmp("bad");
        let (ip, dp) = write_test_checkpoint(&d, &[("a", vec![4], vec![1.0, 2.0, 3.0, 4.0])]);
        let good = std::fs::read(&ip).unwrap();
        let data = std::fs::read(&dp).unwrap();
        for n in [0, 10, 47, 48, good.len() - 1] {
            std::fs::write(&ip, &good[..n]).unwrap();
            assert!(Checkpoint::open(&ip, &dp).is_err(), "truncated to {n}");
        }
        for i in 0..good.len() {
            for x in [0x01u8, 0x80, 0xff] {
                let mut b = good.clone();
                b[i] ^= x;
                std::fs::write(&ip, &b).unwrap();
                std::panic::catch_unwind(|| {
                    if let Ok(c) = Checkpoint::open(&ip, &dp) {
                        let _ = c.load("a", &[4]);
                    }
                })
                .expect("no panic");
            }
        }
        std::fs::write(&ip, &good).unwrap();
        for i in 0..data.len() {
            let mut b = data.clone();
            b[i] ^= 0xa5;
            std::fs::write(&dp, &b).unwrap();
            std::panic::catch_unwind(|| {
                if let Ok(c) = Checkpoint::open(&ip, &dp) {
                    let _ = c.load("a", &[4]);
                }
            })
            .expect("no panic");
        }
        // a short data file
        std::fs::write(&dp, &data[..8]).unwrap();
        assert!(Checkpoint::open(&ip, &dp).is_err() || Checkpoint::open(&ip, &dp).unwrap().load("a", &[4]).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
