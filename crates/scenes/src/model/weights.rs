//! Reading safetensors weights without memory-mapping: the header is parsed and checked before
//! anything else is read (offsets inside the file, sizes that match the shapes, a known dtype,
//! counts and sizes capped), then each tensor is read on its own with a seek, so a
//! multi-gigabyte file is never held in memory whole. Values are checked to be finite.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};

use super::{R, err};

/// Largest header accepted.
const MAX_HEADER: u64 = 32 << 20;
/// Most tensors in one file.
const MAX_TENSORS: usize = 100_000;
/// Largest single tensor (elements): Qwen3-VL's 151 936 × 4096 embedding is 622 M.
const MAX_ELEMENTS: usize = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dt {
    Bf16,
    F16,
    F32,
    /// Another type (integer buffers such as CLIP's `position_ids`): its element size, so the
    /// header is still checked, but it is never loaded.
    Other(usize),
}

impl Dt {
    fn size(self) -> usize {
        match self {
            Dt::Bf16 | Dt::F16 => 2,
            Dt::F32 => 4,
            Dt::Other(n) => n,
        }
    }
}

#[derive(Clone, Debug)]
struct Entry {
    dt: Dt,
    shape: Vec<usize>,
    begin: u64,
    len: u64,
}

/// An open safetensors file whose header has been checked.
pub struct SafeTensors {
    path: PathBuf,
    file: File,
    data_start: u64,
    entries: HashMap<String, Entry>,
}

impl SafeTensors {
    pub fn open(path: &Path) -> R<SafeTensors> {
        let mut file = File::open(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
        let file_len = file.metadata().map_err(|e| err(format!("{}: {e}", path.display())))?.len();
        let mut n = [0u8; 8];
        file.read_exact(&mut n).map_err(|_| err(format!("{}: not a safetensors file (too short)", path.display())))?;
        let header_len = u64::from_le_bytes(n);
        if header_len == 0 || header_len > MAX_HEADER || header_len.saturating_add(8) > file_len {
            return Err(err(format!("{}: damaged safetensors header", path.display())));
        }
        let mut header = Vec::new();
        (&mut file).take(header_len).read_to_end(&mut header).map_err(|e| err(format!("{}: {e}", path.display())))?;
        if header.len() as u64 != header_len {
            return Err(err(format!("{}: truncated safetensors header", path.display())));
        }
        let data_start = 8 + header_len;
        let entries = parse_header(&header, file_len - data_start).map_err(|e| err(format!("{}: {e}", path.display())))?;
        Ok(SafeTensors { path: path.to_path_buf(), file, data_start, entries })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    fn bytes(&mut self, name: &str) -> R<(Dt, Vec<usize>, Vec<u8>)> {
        let e = self.entries.get(name).ok_or_else(|| err(format!("{}: missing weight `{name}`", self.path.display())))?.clone();
        self.file.seek(SeekFrom::Start(self.data_start.saturating_add(e.begin))).map_err(|er| err(format!("{}: {er}", self.path.display())))?;
        let len = usize::try_from(e.len).map_err(|_| err(format!("weight `{name}` is too large")))?;
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(len).map_err(|_| err(format!("weight `{name}`: out of memory")))?;
        bytes.resize(len, 0);
        self.file.read_exact(&mut bytes).map_err(|er| err(format!("{}: `{name}`: {er}", self.path.display())))?;
        Ok((e.dt, e.shape, bytes))
    }

    /// The tensor `name` on `dev` in `dtype` (f32, f16 or bf16), checked for finite values. A
    /// file's own bf16 kept as bf16 (or f16 as f16) is not converted.
    pub fn load(&mut self, name: &str, dev: &Device, dtype: DType) -> R<Tensor> {
        let (dt, shape, bytes) = self.bytes(name)?;
        if let Dt::Other(_) = dt {
            return Err(err(format!("weight `{name}` is not a float tensor")));
        }
        let same = matches!((dt, dtype), (Dt::Bf16, DType::BF16) | (Dt::F16, DType::F16));
        if same {
            // exponent all ones: infinity or NaN
            let exp = if dt == Dt::F16 { 0x7c00 } else { 0x7f80 };
            if bytes.as_chunks::<2>().0.iter().any(|b| u16::from_le_bytes(*b) & exp == exp) {
                return Err(err(format!("weight `{name}` is not finite")));
            }
            return Tensor::from_raw_buffer(&bytes, dtype, &shape, dev).map_err(err);
        }
        let data: Vec<f32> = match dt {
            Dt::Bf16 => bytes.as_chunks::<2>().0.iter().map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16)).collect(),
            Dt::F16 => bytes.as_chunks::<2>().0.iter().map(|b| filmcraft_project::scenes::f16_to_f32(u16::from_le_bytes(*b))).collect(),
            Dt::F32 => bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
            Dt::Other(_) => Vec::new(),
        };
        drop(bytes);
        if data.iter().any(|v| !v.is_finite()) {
            return Err(err(format!("weight `{name}` is not finite")));
        }
        let t = Tensor::from_vec(data, shape.as_slice(), dev).map_err(err)?;
        if dtype == DType::F32 { Ok(t) } else { t.to_dtype(dtype).map_err(err) }
    }

    /// Every tensor whose name starts with one of `prefixes`, into `out` (by its full name).
    pub fn load_prefixed(&mut self, prefixes: &[&str], dev: &Device, dtype: DType, out: &mut HashMap<String, Tensor>) -> R<usize> {
        let mut names: Vec<String> =
            self.entries.iter().filter(|(n, e)| !matches!(e.dt, Dt::Other(_)) && prefixes.iter().any(|p| n.starts_with(p))).map(|(n, _)| n.clone()).collect();
        names.sort();
        for n in &names {
            let t = self.load(n, dev, dtype)?;
            out.insert(n.clone(), t);
        }
        Ok(names.len())
    }
}

fn parse_header(header: &[u8], data_len: u64) -> Result<HashMap<String, Entry>, String> {
    let v: serde_json::Value = serde_json::from_slice(header).map_err(|e| format!("bad safetensors header: {e}"))?;
    let obj = v.as_object().ok_or("the safetensors header is not an object")?;
    if obj.len() > MAX_TENSORS + 1 {
        return Err("too many tensors".into());
    }
    let mut out = HashMap::with_capacity(obj.len());
    for (name, t) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dt = match t.get("dtype").and_then(|d| d.as_str()) {
            Some("BF16") => Dt::Bf16,
            Some("F16") => Dt::F16,
            Some("F32") => Dt::F32,
            Some("F64" | "I64" | "U64") => Dt::Other(8),
            Some("I32" | "U32") => Dt::Other(4),
            Some("I16" | "U16") => Dt::Other(2),
            Some("I8" | "U8" | "BOOL" | "F8_E4M3" | "F8_E5M2") => Dt::Other(1),
            other => return Err(format!("`{name}`: dtype {other:?} is not supported")),
        };
        let shape: Vec<usize> = t
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| format!("`{name}`: no shape"))?
            .iter()
            .map(|d| d.as_u64().and_then(|d| usize::try_from(d).ok()))
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| format!("`{name}`: bad shape"))?;
        if shape.len() > 8 {
            return Err(format!("`{name}`: too many dimensions"));
        }
        let elements = shape.iter().try_fold(1usize, |a, d| a.checked_mul(*d)).filter(|n| *n <= MAX_ELEMENTS).ok_or_else(|| format!("`{name}`: too large"))?;
        let offs = t.get("data_offsets").and_then(|o| o.as_array()).ok_or_else(|| format!("`{name}`: no data offsets"))?;
        let (Some(begin), Some(end)) = (offs.first().and_then(|x| x.as_u64()), offs.get(1).and_then(|x| x.as_u64())) else {
            return Err(format!("`{name}`: bad data offsets"));
        };
        let len = end.checked_sub(begin).ok_or_else(|| format!("`{name}`: data offsets run backwards"))?;
        if end > data_len {
            return Err(format!("`{name}`: data past the end of the file"));
        }
        if (elements as u64).checked_mul(dt.size() as u64) != Some(len) {
            return Err(format!("`{name}`: {len} bytes do not hold {shape:?}"));
        }
        out.insert(name.clone(), Entry { dt, shape, begin, len });
    }
    Ok(out)
}

/// Write an f32 safetensors file (tests: small fake models and damaged files).
#[cfg(test)]
pub(crate) fn write_f32(path: &Path, tensors: &[(&str, Vec<usize>, Vec<f32>)]) {
    use std::io::Write;
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, shape, v) in tensors {
        let begin = data.len();
        for x in v {
            data.extend_from_slice(&x.to_le_bytes());
        }
        header.insert(name.to_string(), serde_json::json!({"dtype": "F32", "shape": shape, "data_offsets": [begin, data.len()]}));
    }
    let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut f = File::create(path).unwrap();
    f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
    f.write_all(&h).unwrap();
    f.write_all(&data).unwrap();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("filmcraft-scenes-st-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn reads_tensors_by_name_and_prefix() {
        let d = tmp("ok");
        let p = d.join("m.safetensors");
        write_f32(&p, &[("a.weight", vec![2, 3], vec![1.0, -2.5, 0.5, 3.0, 0.0, -0.0]), ("b.bias", vec![2], vec![7.0, 8.0])]);
        let mut st = SafeTensors::open(&p).unwrap();
        assert!(st.contains("a.weight") && !st.contains("c"));
        assert_eq!(st.shape("a.weight"), Some(&[2usize, 3][..]));
        let t = st.load("a.weight", &Device::Cpu, DType::F32).unwrap();
        assert_eq!(t.to_vec2::<f32>().unwrap(), vec![vec![1.0, -2.5, 0.5], vec![3.0, 0.0, -0.0]]);
        let b = st.load("b.bias", &Device::Cpu, DType::BF16).unwrap();
        assert_eq!(b.dtype(), DType::BF16);
        let mut all = HashMap::new();
        assert_eq!(st.load_prefixed(&["a."], &Device::Cpu, DType::F32, &mut all).unwrap(), 1);
        assert!(all.contains_key("a.weight"));
        assert!(st.load("missing", &Device::Cpu, DType::F32).is_err());
        assert_eq!(st.names().count(), 2);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn hostile_files_are_errors_not_panics() {
        let d = tmp("bad");
        let p = d.join("m.safetensors");
        write_f32(&p, &[("a", vec![2, 3], vec![1.0; 6])]);
        let good = std::fs::read(&p).unwrap();
        for n in [0, 4, 8, 9, 20, good.len() - 1] {
            std::fs::write(&p, &good[..n]).unwrap();
            assert!(SafeTensors::open(&p).and_then(|mut s| s.load("a", &Device::Cpu, DType::F32)).is_err(), "truncated to {n}");
        }
        for i in 0..good.len() {
            let mut b = good.clone();
            b[i] ^= 0x5a;
            std::fs::write(&p, &b).unwrap();
            std::panic::catch_unwind(|| {
                let _ = SafeTensors::open(&p).and_then(|mut s| s.load("a", &Device::Cpu, DType::F32));
                let _ = SafeTensors::open(&p).and_then(|mut s| s.load("a", &Device::Cpu, DType::BF16));
            })
            .expect("no panic");
        }
        for h in [
            r#"{"a":{"dtype":"C64","shape":[1],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"I64","shape":[2],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1000000,1000000],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[4294967296,4294967296,16],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,999999]}}"#,
            r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1,1,1,1,1,1,1,1,1],"data_offsets":[0,4]}}"#,
            r#"[1,2]"#,
        ] {
            let mut b = (h.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(h.as_bytes());
            b.extend_from_slice(&[0u8; 8]);
            std::fs::write(&p, &b).unwrap();
            assert!(SafeTensors::open(&p).is_err(), "{h}");
        }
        write_f32(&p, &[("n", vec![2], vec![f32::NAN, 1.0])]);
        assert!(SafeTensors::open(&p).unwrap().load("n", &Device::Cpu, DType::F32).is_err());
        assert!(SafeTensors::open(&d.join("missing")).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
