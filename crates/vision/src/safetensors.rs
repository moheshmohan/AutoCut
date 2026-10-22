//! A strict reader of safetensors files (the person detector's weights).
//!
//! The file is read whole (it is capped, [`MAX_BYTES`]), its JSON header parsed and checked
//! before any value is used: every entry's offsets inside the data, its byte length equal to its
//! shape's element count times the type's size, the shape's rank and element count capped, and
//! the float32 values finite. Entries of other types (a batch norm's `num_batches_tracked`
//! counter) are skipped; only float32 tensors are kept.

use std::collections::HashMap;
use std::path::Path;

use crate::VisionError;

/// Largest file read (the small D-FINE models are 15 and 41 MB).
pub const MAX_BYTES: u64 = 256 << 20;
/// Largest header.
const MAX_HEADER: u64 = 16 << 20;
/// Most entries in one file.
const MAX_TENSORS: usize = 20_000;
/// Most elements of one tensor.
const MAX_ELEMENTS: usize = 64 << 20;
/// Most dimensions of one tensor.
const MAX_RANK: usize = 8;

fn bad(what: impl std::fmt::Display) -> VisionError {
    VisionError::Model(format!("safetensors: {what}"))
}

/// One float32 tensor: its shape and values, row-major.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// The float32 tensors of a safetensors file, by name.
#[derive(Clone, Debug, Default)]
pub struct SafeTensors {
    tensors: HashMap<String, Tensor>,
}

impl SafeTensors {
    /// Read a file.
    pub fn read(path: &Path) -> Result<Self, VisionError> {
        let len = std::fs::metadata(path).map_err(|e| bad(format!("{}: {e}", path.display())))?.len();
        if len > MAX_BYTES {
            return Err(bad(format!("{} is larger than {MAX_BYTES} bytes", path.display())));
        }
        let bytes = std::fs::read(path).map_err(|e| bad(format!("{}: {e}", path.display())))?;
        Self::parse(&bytes)
    }

    /// Parse a file's bytes.
    pub fn parse(bytes: &[u8]) -> Result<Self, VisionError> {
        if bytes.len() as u64 > MAX_BYTES {
            return Err(bad("the file is too large"));
        }
        let n = bytes.get(..8).ok_or_else(|| bad("too short for a header"))?;
        let mut le = [0u8; 8];
        le.copy_from_slice(n);
        let header_len = u64::from_le_bytes(le);
        if header_len == 0 || header_len > MAX_HEADER || header_len > bytes.len() as u64 - 8 {
            return Err(bad("damaged header length"));
        }
        let hl = usize::try_from(header_len).map_err(|_| bad("damaged header length"))?;
        let header = bytes.get(8..8 + hl).ok_or_else(|| bad("truncated header"))?;
        let data = bytes.get(8 + hl..).unwrap_or(&[]);
        let v: serde_json::Value = serde_json::from_slice(header).map_err(|e| bad(format!("header: {e}")))?;
        let obj = v.as_object().ok_or_else(|| bad("the header is not an object"))?;
        if obj.len() > MAX_TENSORS {
            return Err(bad("too many tensors"));
        }
        let mut tensors = HashMap::new();
        for (name, e) in obj {
            if name == "__metadata__" {
                continue;
            }
            let what = |m: &str| bad(format!("`{}`: {m}", name.chars().take(120).collect::<String>()));
            let dtype = e.get("dtype").and_then(serde_json::Value::as_str).ok_or_else(|| what("no dtype"))?;
            let shape: Vec<usize> = e
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| what("no shape"))?
                .iter()
                .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
                .collect::<Option<_>>()
                .ok_or_else(|| what("a dimension is not a size"))?;
            if shape.len() > MAX_RANK {
                return Err(what("too many dimensions"));
            }
            let count = shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d)).filter(|n| *n <= MAX_ELEMENTS).ok_or_else(|| what("too many elements"))?;
            let offs = e.get("data_offsets").and_then(serde_json::Value::as_array).ok_or_else(|| what("no data offsets"))?;
            let (Some(b), Some(en), 2) = (offs.first().and_then(serde_json::Value::as_u64), offs.get(1).and_then(serde_json::Value::as_u64), offs.len()) else {
                return Err(what("damaged data offsets"));
            };
            if b > en || en > data.len() as u64 {
                return Err(what("its data lies outside the file"));
            }
            let size = match dtype {
                "F32" | "I32" | "U32" => 4u64,
                "F64" | "I64" | "U64" => 8,
                "F16" | "BF16" | "I16" | "U16" => 2,
                "I8" | "U8" | "BOOL" => 1,
                other => return Err(what(&format!("unknown type {}", other.chars().take(16).collect::<String>()))),
            };
            if (count as u64).checked_mul(size) != Some(en - b) {
                return Err(what("its byte length does not match its shape"));
            }
            if dtype != "F32" {
                continue;
            }
            let raw = data.get(b as usize..en as usize).ok_or_else(|| what("its data lies outside the file"))?;
            let vals: Vec<f32> = raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
            if vals.iter().any(|v| !v.is_finite()) {
                return Err(what("values are not finite"));
            }
            tensors.insert(name.clone(), Tensor { shape, data: vals });
        }
        Ok(Self { tensors })
    }

    /// Whether the file has a float32 tensor `name`.
    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    /// Tensor `name`, whatever its shape.
    pub fn get(&self, name: &str) -> Result<&Tensor, VisionError> {
        self.tensors.get(name).ok_or_else(|| bad(format!("missing weight `{name}`")))
    }

    /// The values of tensor `name`, which must have shape `shape`.
    pub fn take(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>, VisionError> {
        let t = self.get(name)?;
        if t.shape != shape {
            return Err(bad(format!("weight `{name}` is {:?}, expected {shape:?}", t.shape)));
        }
        Ok(t.data.clone())
    }

    /// The one value of a single-element tensor `name` (any rank).
    pub fn scalar(&self, name: &str) -> Result<f32, VisionError> {
        let t = self.get(name)?;
        match t.data.as_slice() {
            [v] => Ok(*v),
            _ => Err(bad(format!("weight `{name}` is {:?}, expected one value", t.shape))),
        }
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }
}

/// A safetensors file's bytes, for tests: `(name, dtype, shape, raw bytes)`.
#[cfg(test)]
pub(crate) fn write(entries: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, raw) in entries {
        let b = data.len();
        data.extend_from_slice(raw);
        header.insert((*name).to_string(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [b, data.len()]}));
    }
    let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = (h.len() as u64).to_le_bytes().to_vec();
    out.extend_from_slice(&h);
    out.extend_from_slice(&data);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn reads_float_tensors_and_skips_others() {
        let w = f32s(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let n = 7i64.to_le_bytes();
        let file = write(&[("a.weight", "F32", &[2, 3], &w), ("a.num_batches_tracked", "I64", &[], &n), ("s", "F32", &[1], &f32s(&[0.5]))]);
        let st = SafeTensors::parse(&file).unwrap();
        assert_eq!(st.len(), 2);
        assert_eq!(st.take("a.weight", &[2, 3]).unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(st.take("a.weight", &[3, 2]).is_err());
        assert!(st.get("a.num_batches_tracked").is_err());
        assert_eq!(st.scalar("s").unwrap(), 0.5);
        assert!(st.scalar("a.weight").is_err());
        assert!(st.get("missing").is_err());
    }

    #[test]
    fn damaged_files_are_errors_not_panics() {
        let good = write(&[("a", "F32", &[2, 2], &f32s(&[1.0, 2.0, 3.0, 4.0])), ("b", "F32", &[3], &f32s(&[1.0, 2.0, 3.0]))]);
        assert!(SafeTensors::parse(&good).is_ok());
        // truncation at every length, bit flips everywhere
        for cut in 0..good.len() {
            let r = std::panic::catch_unwind(|| SafeTensors::parse(&good[..cut]).map(|s| s.len()));
            assert!(r.is_ok(), "panicked at length {cut}");
        }
        for i in 0..good.len() {
            for bit in 0..8 {
                let mut m = good.clone();
                m[i] ^= 1 << bit;
                let r = std::panic::catch_unwind(|| SafeTensors::parse(&m).map(|s| s.len()));
                assert!(r.is_ok(), "panicked flipping bit {bit} of byte {i}");
            }
        }
        // hostile headers
        let hostile = [
            write(&[("a", "F32", &[usize::MAX, 2], &[])]),
            write(&[("a", "F32", &[3], &f32s(&[1.0, 2.0]))]),
            write(&[("a", "Q4", &[1], &[0])]),
            write(&[("a", "F32", &[1], &f32s(&[f32::NAN]))]),
            write(&[("a", "F32", &[1, 1, 1, 1, 1, 1, 1, 1, 1], &f32s(&[1.0]))]),
        ];
        for h in hostile {
            assert!(SafeTensors::parse(&h).is_err());
        }
        let mut huge_len = u64::MAX.to_le_bytes().to_vec();
        huge_len.extend_from_slice(b"{}");
        assert!(SafeTensors::parse(&huge_len).is_err());
        assert!(SafeTensors::parse(b"").is_err());
        let mut offsets = b"{\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[8,4]}}".to_vec();
        let mut f = (offsets.len() as u64).to_le_bytes().to_vec();
        f.append(&mut offsets);
        f.extend_from_slice(&[0; 8]);
        assert!(SafeTensors::parse(&f).is_err());
    }
}
