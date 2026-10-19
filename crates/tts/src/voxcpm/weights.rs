//! Reading VoxCPM's weights.
//!
//! - `model.safetensors` (bf16): the header is parsed and checked (offsets inside the file, sizes
//!   that match the shapes, a known dtype, no overlap past the end), then each tensor is read on
//!   its own with a seek and converted to f32. Nothing is memory-mapped (that needs `unsafe`) and
//!   the file is never held in memory whole.
//! - `audiovae.pth`: a PyTorch checkpoint with the state dictionary under `state_dict`, read with
//!   candle's restricted pickle reader (tensors only, no code from the file runs) by the Kokoro
//!   loader, which also folds weight norm (`weight_g`, `weight_v`) into plain weights.

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use candle_core::{Device, Tensor};

use super::{R, err};

/// Largest header accepted (the released ones are ≈ 60 KB).
const MAX_HEADER: u64 = 16 << 20;
/// Most tensors in one file.
const MAX_TENSORS: usize = 100_000;
/// Largest single tensor (elements): the biggest released one is the 73 448 × 2048 embedding.
const MAX_ELEMENTS: usize = 1 << 30;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Dt {
    Bf16,
    F16,
    F32,
}

impl Dt {
    fn size(self) -> usize {
        match self {
            Dt::Bf16 | Dt::F16 => 2,
            Dt::F32 => 4,
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
pub(crate) struct SafeTensors {
    path: PathBuf,
    file: File,
    data_start: u64,
    entries: HashMap<String, Entry>,
}

impl SafeTensors {
    pub(crate) fn open(path: &Path) -> R<SafeTensors> {
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

    #[cfg(test)]
    pub(crate) fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// The shape of the tensor `name`, if the file has it (IndicF5 reads its architecture from
    /// the shapes).
    #[cfg(feature = "indicf5")]
    pub(crate) fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    #[cfg(test)]
    pub(crate) fn count(&self) -> usize {
        self.entries.len()
    }

    /// The tensor `name` as f32 on `dev`, checked against `shape`.
    pub(crate) fn load(&mut self, name: &str, shape: &[usize], dev: &Device) -> R<Tensor> {
        let e = self.entries.get(name).ok_or_else(|| err(format!("{}: missing weight `{name}`", self.path.display())))?.clone();
        if e.shape != shape {
            return Err(err(format!("weight `{name}` is {:?}, expected {shape:?}", e.shape)));
        }
        self.file.seek(SeekFrom::Start(self.data_start.saturating_add(e.begin))).map_err(|er| err(format!("{}: {er}", self.path.display())))?;
        // the length was checked against the file size and the element cap when the header was read
        let mut bytes = vec![0u8; e.len as usize];
        self.file.read_exact(&mut bytes).map_err(|er| err(format!("{}: `{name}`: {er}", self.path.display())))?;
        let data: Vec<f32> = match e.dt {
            Dt::Bf16 => bytes.as_chunks::<2>().0.iter().map(|b| f32::from_bits(u32::from(u16::from_le_bytes(*b)) << 16)).collect(),
            Dt::F16 => bytes.as_chunks::<2>().0.iter().map(|b| f16_to_f32(u16::from_le_bytes(*b))).collect(),
            Dt::F32 => bytes.as_chunks::<4>().0.iter().map(|b| f32::from_le_bytes(*b)).collect(),
        };
        if data.iter().any(|v| !v.is_finite()) {
            return Err(err(format!("weight `{name}` is not finite")));
        }
        Tensor::from_vec(data, shape, dev).map_err(err)
    }
}

/// IEEE half → single (subnormals, infinities and NaN included).
fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // subnormal: normalise
            let mut e = 127 - 15 + 1;
            let mut m = m;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
        (0x1f, m) => sign | 0x7f80_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
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

/// The AudioVAE checkpoint (`state_dict` key), weight norm folded.
pub(crate) fn load_vae(path: &Path) -> R<crate::kokoro::weights::Weights> {
    crate::kokoro::weights::Weights::load_key(path, "state_dict")
}

/// Write a safetensors file (tests: the fake models, and the corrupt-file cases).
#[cfg(test)]
pub(crate) fn write(path: &Path, tensors: &[(String, Tensor)]) {
    use std::io::Write;
    let mut header = serde_json::Map::new();
    let mut data: Vec<u8> = Vec::new();
    for (name, t) in tensors {
        let v: Vec<f32> = t.flatten_all().unwrap().to_dtype(candle_core::DType::F32).unwrap().to_vec1().unwrap();
        let begin = data.len();
        for x in v {
            // bf16, round to nearest even like the released files
            let b = x.to_bits();
            let r = ((b >> 16) & 1) + 0x7fff;
            data.extend_from_slice(&(((b + r) >> 16) as u16).to_le_bytes());
        }
        header.insert(name.clone(), serde_json::json!({"dtype": "BF16", "shape": t.dims(), "data_offsets": [begin, data.len()]}));
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
        let d = std::env::temp_dir().join(format!("filmcraft-voxcpm-st-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn round_trips_bf16() {
        let d = tmp("ok");
        let p = d.join("m.safetensors");
        let t = Tensor::from_vec(vec![1.0f32, -2.5, 0.15625, 3.0e5, 0.0, -0.0], (2, 3), &Device::Cpu).unwrap();
        write(&p, &[("a.weight".into(), t)]);
        let mut st = SafeTensors::open(&p).unwrap();
        assert_eq!(st.count(), 1);
        assert!(st.contains("a.weight"));
        let back = st.load("a.weight", &[2, 3], &Device::Cpu).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(back, vec![1.0, -2.5, 0.15625, 299_008.0, 0.0, -0.0]);
        assert!(st.load("a.weight", &[3, 2], &Device::Cpu).is_err(), "shape is checked");
        assert!(st.load("missing", &[1], &Device::Cpu).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn halves_convert() {
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert!(f16_to_f32(0x7c00).is_infinite());
        assert!(f16_to_f32(0x7e00).is_nan());
    }

    #[test]
    fn hostile_files_are_errors_not_panics() {
        let d = tmp("bad");
        let p = d.join("m.safetensors");
        let t = Tensor::from_vec(vec![1.0f32; 6], (2, 3), &Device::Cpu).unwrap();
        write(&p, &[("a".into(), t)]);
        let good = std::fs::read(&p).unwrap();
        // truncations at every length
        for n in [0, 4, 8, 9, 20, good.len() - 1] {
            std::fs::write(&p, &good[..n]).unwrap();
            let r = SafeTensors::open(&p).and_then(|mut s| s.load("a", &[2, 3], &Device::Cpu));
            assert!(r.is_err(), "truncated to {n}");
        }
        // bit flips anywhere: never a panic
        for i in 0..good.len() {
            let mut b = good.clone();
            b[i] ^= 0x5a;
            std::fs::write(&p, &b).unwrap();
            let _ = std::panic::catch_unwind(|| SafeTensors::open(&p).and_then(|mut s| s.load("a", &[2, 3], &Device::Cpu))).expect("no panic");
        }
        // doctored headers
        for h in [
            r#"{"a":{"dtype":"I64","shape":[1],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[1000000,1000000],"data_offsets":[0,4]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[8,0]}}"#,
            r#"{"a":{"dtype":"F32","shape":[2],"data_offsets":[0,999999]}}"#,
            r#"{"a":{"dtype":"F32","shape":[3],"data_offsets":[0,8]}}"#,
            r#"{"a":{"dtype":"F32","shape":[-1],"data_offsets":[0,8]}}"#,
            r#"[1,2]"#,
            r#"{"a":5}"#,
        ] {
            let mut b = (h.len() as u64).to_le_bytes().to_vec();
            b.extend_from_slice(h.as_bytes());
            b.extend_from_slice(&[0u8; 8]);
            std::fs::write(&p, &b).unwrap();
            assert!(SafeTensors::open(&p).is_err(), "{h}");
        }
        // NaN weights are refused
        let t = Tensor::from_vec(vec![f32::NAN; 2], 2, &Device::Cpu).unwrap();
        write(&p, &[("n".into(), t)]);
        assert!(SafeTensors::open(&p).unwrap().load("n", &[2], &Device::Cpu).is_err());
        assert!(SafeTensors::open(&d.join("missing")).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }
}
