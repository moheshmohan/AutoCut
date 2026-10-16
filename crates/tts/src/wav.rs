//! Minimal WAV I/O for voice samples kept next to cloned-voice profiles: mono 16-bit PCM out;
//! in, RIFF/WAVE with 16-, 24- or 32-bit integer or 32-bit float PCM, any channel count (mixed
//! to mono). The file is hostile: chunk sizes are checked against the file, the sample count and
//! rate are capped, and anything else is an error, never a panic.

use crate::TtsError;

/// Longest file read (a voice sample is seconds long; this is minutes at 48 kHz stereo).
pub const MAX_BYTES: u64 = 64 << 20;

fn bad(why: &str) -> TtsError {
    TtsError::Invalid(format!("voice sample: {why}"))
}

/// Mono 16-bit PCM WAV bytes of `x` at `rate`, clipped to ±1.
pub fn encode(x: &[f32], rate: u32) -> Vec<u8> {
    let data = (x.len() * 2) as u32;
    let mut v = Vec::with_capacity(44 + x.len() * 2);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&36u32.saturating_add(data).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes());
    v.extend_from_slice(&rate.to_le_bytes());
    v.extend_from_slice(&rate.saturating_mul(2).to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data.to_le_bytes());
    for s in x {
        let s = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
        v.extend_from_slice(&((s * 32767.0).round() as i16).to_le_bytes());
    }
    v
}

/// Write `x` as a mono 16-bit WAV, atomically (a temporary file renamed into place).
pub fn write(path: &std::path::Path, x: &[f32], rate: u32) -> Result<(), TtsError> {
    let tmp = path.with_extension("wav.tmp");
    std::fs::write(&tmp, encode(x, rate)).map_err(|e| TtsError::Model(format!("{}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))
}

/// Read a WAV file: mono samples and the rate.
pub fn read(path: &std::path::Path) -> Result<(Vec<f32>, u32), TtsError> {
    let len = std::fs::metadata(path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?.len();
    if len > MAX_BYTES {
        return Err(bad("the file is too large"));
    }
    let bytes = std::fs::read(path).map_err(|e| TtsError::Model(format!("{}: {e}", path.display())))?;
    decode(&bytes)
}

fn u16_at(b: &[u8], i: usize) -> Option<u16> {
    Some(u16::from_le_bytes([*b.get(i)?, *b.get(i + 1)?]))
}

fn u32_at(b: &[u8], i: usize) -> Option<u32> {
    Some(u32::from_le_bytes([*b.get(i)?, *b.get(i + 1)?, *b.get(i + 2)?, *b.get(i + 3)?]))
}

/// Parse WAV bytes: mono samples and the rate.
pub fn decode(b: &[u8]) -> Result<(Vec<f32>, u32), TtsError> {
    if b.get(0..4) != Some(b"RIFF") || b.get(8..12) != Some(b"WAVE") {
        return Err(bad("not a WAV file"));
    }
    let mut pos = 12usize;
    let mut fmt: Option<(u16, u16, u32, u16)> = None;
    while pos.checked_add(8).is_some_and(|e| e <= b.len()) {
        let id = b.get(pos..pos + 4).ok_or_else(|| bad("truncated"))?;
        let size = u32_at(b, pos + 4).ok_or_else(|| bad("truncated"))? as usize;
        let start = pos + 8;
        let end = start.checked_add(size).filter(|e| *e <= b.len()).ok_or_else(|| bad("a chunk runs past the end of the file"))?;
        let body = b.get(start..end).ok_or_else(|| bad("truncated"))?;
        if id == b"fmt " {
            let tag = u16_at(body, 0).ok_or_else(|| bad("short format chunk"))?;
            let ch = u16_at(body, 2).ok_or_else(|| bad("short format chunk"))?;
            let rate = u32_at(body, 4).ok_or_else(|| bad("short format chunk"))?;
            let bits = u16_at(body, 14).ok_or_else(|| bad("short format chunk"))?;
            // WAVE_FORMAT_EXTENSIBLE: the real tag is the first two bytes of the sub-format GUID
            let tag = if tag == 0xFFFE { u16_at(body, 24).ok_or_else(|| bad("short extensible format"))? } else { tag };
            fmt = Some((tag, ch, rate, bits));
        } else if id == b"data" {
            let (tag, ch, rate, bits) = fmt.ok_or_else(|| bad("the data comes before its format"))?;
            if ch == 0 || ch > 32 || !(1_000..=384_000).contains(&rate) {
                return Err(bad("unsupported channels or rate"));
            }
            let width = match (tag, bits) {
                (1, 16) => 2,
                (1, 24) => 3,
                (1, 32) | (3, 32) => 4,
                _ => return Err(bad("only 16/24/32-bit PCM and 32-bit float are supported")),
            };
            let frame = width * usize::from(ch);
            let frames = body.len() / frame;
            let mut out = Vec::with_capacity(frames);
            for f in body.chunks_exact(frame) {
                let mut acc = 0f32;
                for s in f.chunks_exact(width) {
                    let a = |i: usize| s.get(i).copied().unwrap_or(0);
                    acc += match (tag, width) {
                        (1, 2) => f32::from(i16::from_le_bytes([a(0), a(1)])) / 32768.0,
                        (1, 3) => (i32::from_le_bytes([0, a(0), a(1), a(2)]) >> 8) as f32 / 8_388_608.0,
                        (1, 4) => i32::from_le_bytes([a(0), a(1), a(2), a(3)]) as f32 / 2_147_483_648.0,
                        _ => {
                            let v = f32::from_le_bytes([a(0), a(1), a(2), a(3)]);
                            if v.is_finite() { v } else { 0.0 }
                        }
                    };
                }
                out.push(acc / f32::from(ch));
            }
            return Ok((out, rate));
        }
        // chunks are padded to even sizes
        pos = end + (size & 1);
    }
    Err(bad("no audio data"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_mixes_down() {
        let x: Vec<f32> = (0..1000).map(|i| (i as f32 * 0.01).sin() * 0.5).collect();
        let (y, r) = decode(&encode(&x, 16_000)).unwrap();
        assert_eq!(r, 16_000);
        assert_eq!(y.len(), x.len());
        assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-4));
        // stereo float, opposite channels cancel
        let mut b = encode(&[], 8_000);
        b[22] = 2; // channels
        b[20] = 3; // float
        b[34] = 32; // bits
        let samples: Vec<u8> = [0.5f32, -0.5, 0.25, 0.25].iter().flat_map(|v| v.to_le_bytes()).collect();
        b[40..44].copy_from_slice(&(samples.len() as u32).to_le_bytes());
        b.extend_from_slice(&samples);
        let (y, _) = decode(&b).unwrap();
        assert_eq!(y, vec![0.0, 0.25]);
    }

    #[test]
    fn hostile_files_are_errors() {
        let good = encode(&[0.1, 0.2, 0.3], 16_000);
        for n in 0..good.len() {
            let _ = decode(&good[..n]);
        }
        for i in 0..good.len() {
            let mut b = good.clone();
            b[i] ^= 0xff;
            let _ = decode(&b);
        }
        assert!(decode(b"RIFF\0\0\0\0WAVE").is_err());
        let mut b = good.clone();
        b[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(decode(&b).is_err(), "data chunk past the end");
        let mut b = good;
        b[24..28].copy_from_slice(&0u32.to_le_bytes());
        assert!(decode(&b).is_err(), "zero rate");
    }
}
