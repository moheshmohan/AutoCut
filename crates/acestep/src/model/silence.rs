//! `silence_latent.pt`: the VAE latent of silence (`[1, 64, 15000]` float32, 10 minutes), a
//! PyTorch zip archive holding one bare tensor. The text-to-music path uses it as the source
//! latents (nothing to cover or repaint) and its first 750 frames as the timbre reference (no
//! reference audio).
//!
//! The archive is read without unpickling anything: the tensor's bytes are `<name>/data/0`; the
//! pickle (`<name>/data.pkl`, a few hundred bytes) is only searched for the shape tuple
//! `(1, 64, T)`, which must agree with the data's size; the byte order must be little-endian and
//! every value finite.

use std::io::Read;
use std::path::Path;

use super::{R, err};

/// Most frames accepted (the released file has 15 000).
const MAX_FRAMES: usize = 200_000;
/// Frames the timbre reference takes (the reference's `timbre_fix_frame`).
pub(crate) const TIMBRE_FRAMES: usize = 750;

/// The silence latent, frame-major `[frames, channels]`.
pub(crate) struct Silence {
    pub data: Vec<f32>,
    pub frames: usize,
    pub channels: usize,
}

impl Silence {
    pub(crate) fn load(path: &Path, channels: usize) -> R<Silence> {
        let f = std::fs::File::open(path).map_err(|e| err(format!("{}: {e}", path.display())))?;
        let mut z = zip::ZipArchive::new(f).map_err(|e| err(format!("{}: {e}", path.display())))?;
        let names: Vec<String> = (0..z.len()).filter_map(|i| z.by_index(i).ok().map(|e| e.name().to_string())).collect();
        let find = |suffix: &str| names.iter().find(|n| n.ends_with(suffix)).cloned();
        let bad = |what: &str| err(format!("{}: not a silence latent ({what})", path.display()));
        if let Some(o) = find("/byteorder") {
            let mut s = String::new();
            z.by_name(&o).map_err(err)?.take(16).read_to_string(&mut s).map_err(err)?;
            if s.trim() != "little" {
                return Err(bad("byte order"));
            }
        }
        let pkl = find("/data.pkl").ok_or_else(|| bad("no pickle"))?;
        let mut header = Vec::new();
        {
            let e = z.by_name(&pkl).map_err(err)?;
            if e.size() > 4096 {
                return Err(bad("pickle too large"));
            }
            e.take(4096).read_to_end(&mut header).map_err(err)?;
        }
        let frames = shape_frames(&header, channels).ok_or_else(|| bad("shape"))?;
        if !(TIMBRE_FRAMES..=MAX_FRAMES).contains(&frames) {
            return Err(bad("length"));
        }
        let want = frames * channels * 4;
        let data_name = find("/data/0").ok_or_else(|| bad("no data"))?;
        let mut e = z.by_name(&data_name).map_err(err)?;
        if e.size() != want as u64 {
            return Err(bad("size"));
        }
        let mut buf = Vec::with_capacity(want);
        e.by_ref().take(want as u64).read_to_end(&mut buf).map_err(err)?;
        if buf.len() != want {
            return Err(bad("truncated"));
        }
        // stored [1, channels, frames]: transpose to frame-major
        let v: Vec<f32> = buf.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        if v.iter().any(|x| !x.is_finite()) {
            return Err(bad("not finite"));
        }
        let mut data = vec![0f32; want / 4];
        for c in 0..channels {
            for t in 0..frames {
                if let (Some(d), Some(s)) = (data.get_mut(t * channels + c), v.get(c * frames + t)) {
                    *d = *s;
                }
            }
        }
        Ok(Silence { data, frames, channels })
    }

    /// The first `n` frames, tiled when `n` is longer than the latent (as the reference tiles it).
    pub(crate) fn frames(&self, n: usize) -> Vec<f32> {
        let c = self.channels;
        let mut out = Vec::with_capacity(n.saturating_mul(c));
        for t in 0..n {
            let src = t % self.frames.max(1);
            out.extend_from_slice(self.data.get(src * c..(src + 1) * c).unwrap_or(&[]));
        }
        out
    }
}

/// `T` from the pickled shape tuple `(1, channels, T)` (BININT1 1, BININT1/BININT2 channels,
/// then BININT1, BININT2 or BININT for T), followed by contiguous strides.
fn shape_frames(pkl: &[u8], channels: usize) -> Option<usize> {
    let ch = u8::try_from(channels).ok()?;
    // K 0x01 K <channels>: BININT1 1, BININT1 channels
    let pat = [b'K', 1, b'K', ch];
    let at = pkl.windows(4).position(|w| w == pat)? + 4;
    let rest = pkl.get(at..)?;
    let (frames, used) = match rest.first()? {
        b'K' => (usize::from(*rest.get(1)?), 2),
        b'M' => (usize::from(u16::from_le_bytes([*rest.get(1)?, *rest.get(2)?])), 3),
        b'J' => (usize::try_from(i32::from_le_bytes([*rest.get(1)?, *rest.get(2)?, *rest.get(3)?, *rest.get(4)?])).ok()?, 5),
        _ => return None,
    };
    // the tuple closes (TUPLE3 0x87), then the strides tuple must be contiguous: (64·T, T, 1)
    let rest = rest.get(used..)?;
    if *rest.first()? != 0x87 {
        return None;
    }
    let strides = rest.get(1..)?;
    // skip a memo put (q n / r nnnn), then expect the first stride
    let strides = match strides.first()? {
        b'q' => strides.get(2..)?,
        b'r' => strides.get(5..)?,
        _ => strides,
    };
    let first = match strides.first()? {
        b'K' => usize::from(*strides.get(1)?),
        b'M' => usize::from(u16::from_le_bytes([*strides.get(1)?, *strides.get(2)?])),
        b'J' => usize::try_from(i32::from_le_bytes([*strides.get(1)?, *strides.get(2)?, *strides.get(3)?, *strides.get(4)?])).ok()?,
        _ => return None,
    };
    (first == channels.checked_mul(frames)?).then_some(frames)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    /// The pickle torch writes for a contiguous `[1, channels, frames]` float tensor.
    fn pickle(channels: u8, frames: u16) -> Vec<u8> {
        let mut p = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04J".to_vec();
        p.extend_from_slice(&(u32::from(channels) * u32::from(frames)).to_le_bytes());
        p.extend_from_slice(b"tq\x05QK\x00K\x01K");
        p.push(channels);
        p.push(b'M');
        p.extend_from_slice(&frames.to_le_bytes());
        p.extend_from_slice(b"\x87q\x06J");
        p.extend_from_slice(&(u32::from(channels) * u32::from(frames)).to_le_bytes());
        p.push(b'M');
        p.extend_from_slice(&frames.to_le_bytes());
        p.extend_from_slice(b"K\x01\x87q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.");
        p
    }

    /// A silence-latent archive whose value at (channel c, frame t) is `c + t / 1000`.
    pub(crate) fn write(path: &Path, channels: u8, frames: u16, order: &str) {
        let mut z = zip::ZipWriter::new(std::fs::File::create(path).unwrap());
        let o = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        z.start_file("silence_latent/data.pkl", o).unwrap();
        z.write_all(&pickle(channels, frames)).unwrap();
        z.start_file("silence_latent/byteorder", o).unwrap();
        z.write_all(order.as_bytes()).unwrap();
        z.start_file("silence_latent/data/0", o).unwrap();
        for c in 0..u32::from(channels) {
            for t in 0..u32::from(frames) {
                z.write_all(&(c as f32 + t as f32 / 1000.0).to_le_bytes()).unwrap();
            }
        }
        z.finish().unwrap();
    }

    #[test]
    fn reads_the_tensor_frame_major_and_tiles() {
        let d = std::env::temp_dir().join(format!("filmcraft-acestep-sil-{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        let p = d.join("silence_latent.pt");
        write(&p, 4, 800, "little");
        let s = Silence::load(&p, 4).unwrap();
        assert_eq!((s.frames, s.channels), (800, 4));
        assert_eq!(&s.data[..8], &[0.0, 1.0, 2.0, 3.0, 0.001, 1.001, 2.001, 3.001]);
        let f = s.frames(802);
        assert_eq!(f.len(), 802 * 4);
        assert_eq!(&f[800 * 4..801 * 4], &[0.0, 1.0, 2.0, 3.0], "tiled from the start");
        // wrong channel count, byte order, too short, garbage
        assert!(Silence::load(&p, 8).is_err());
        write(&p, 4, 800, "big");
        assert!(Silence::load(&p, 4).is_err());
        write(&p, 4, 100, "little");
        assert!(Silence::load(&p, 4).is_err(), "shorter than the timbre reference");
        std::fs::write(&p, b"PK\x03\x04 not a zip").unwrap();
        assert!(Silence::load(&p, 4).is_err());
        std::fs::write(&p, b"").unwrap();
        assert!(Silence::load(&p, 4).is_err());
        assert!(Silence::load(&d.join("missing.pt"), 4).is_err());
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn the_shape_is_read_from_the_pickle() {
        assert_eq!(shape_frames(&pickle(64, 15000), 64), Some(15000));
        assert_eq!(shape_frames(&pickle(64, 15000), 32), None);
        // the released file's pickle, byte for byte
        let real = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\nq\x00((X\x07\x00\x00\x00storageq\x01ctorch\nFloatStorage\nq\x02X\x01\x00\x00\x000q\x03X\x03\x00\x00\x00cpuq\x04J\x00\xa6\x0e\x00tq\x05QK\x00K\x01K@M\x98:\x87q\x06J\x00\xa6\x0e\x00M\x98:K\x01\x87q\x07\x89ccollections\nOrderedDict\nq\x08)Rq\ttq\nRq\x0b.";
        assert_eq!(shape_frames(real, 64), Some(15000));
        // every truncation and byte flip: an answer or None, never a panic
        for n in 0..real.len() {
            let _ = shape_frames(&real[..n], 64);
        }
        for i in 0..real.len() {
            let mut b = real.to_vec();
            b[i] ^= 0xff;
            let _ = shape_frames(&b, 64);
        }
    }
}
