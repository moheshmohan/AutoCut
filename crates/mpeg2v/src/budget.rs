//! Memory budget for decoded pictures.
//!
//! A per-axis or area cap does not bound memory: a decoder holds the current picture, its
//! reference pictures and the pictures it outputs. The decoder checks the whole picture store
//! against a byte budget ([`DECODE_MEMORY_BUDGET`] by default) with checked arithmetic before it
//! allocates for a new sequence, so a hostile header is an error, not an out-of-memory abort.

/// Bytes the decoded pictures of one decoder may take by default: 4 GiB on 64-bit targets, 1 GiB
/// on 32-bit ones (wasm32 has a 4 GiB address space in all).
pub const DECODE_MEMORY_BUDGET: u64 = if cfg!(target_pointer_width = "64") { 4 << 30 } else { 1 << 30 };

/// The shape of one decoded picture as the decoder allocates it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PictureShape {
    /// Allocated luma samples per row and rows (padding included).
    pub width: u64,
    pub height: u64,
    /// Chroma subsampling shifts (0 or 1); ignored when `chroma_planes` is 0.
    pub shift_x: u32,
    pub shift_y: u32,
    /// Chroma planes (0 for monochrome, 2 otherwise).
    pub chroma_planes: u64,
    /// Bytes per sample (1 for 8-bit buffers, 2 for 16-bit ones).
    pub bytes_per_sample: u64,
}

impl PictureShape {
    /// Bytes of one picture, `None` if it does not fit in a `u64`.
    pub fn bytes(&self) -> Option<u64> {
        let luma = self.width.checked_mul(self.height)?;
        let cw = self.width.div_ceil(1 << self.shift_x.min(1));
        let ch = self.height.div_ceil(1 << self.shift_y.min(1));
        let chroma = cw.checked_mul(ch)?.checked_mul(self.chroma_planes)?;
        luma.checked_add(chroma)?.checked_mul(self.bytes_per_sample)
    }

    /// Whether `pictures` such pictures fit in `budget` bytes.
    pub fn fits(&self, pictures: u64, budget: u64) -> bool {
        self.bytes().and_then(|b| b.checked_mul(pictures)).is_some_and(|b| b <= budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(w: u64, h: u64) -> PictureShape {
        PictureShape { width: w, height: h, shift_x: 1, shift_y: 1, chroma_planes: 2, bytes_per_sample: 1 }
    }

    #[test]
    fn bytes_count_luma_and_subsampled_chroma() {
        assert_eq!(shape(4, 4).bytes(), Some(16 + 2 * 4));
        // odd sizes round chroma up
        assert_eq!(shape(3, 3).bytes(), Some(9 + 2 * 4));
        let mono = PictureShape { chroma_planes: 0, ..shape(10, 10) };
        assert_eq!(mono.bytes(), Some(100));
        let full = PictureShape { shift_x: 0, shift_y: 0, bytes_per_sample: 2, ..shape(10, 10) };
        assert_eq!(full.bytes(), Some(600));
        let c422 = PictureShape { shift_y: 0, ..shape(16, 16) };
        assert_eq!(c422.bytes(), Some(512));
    }

    #[test]
    fn exact_budget_boundary() {
        let s = shape(4, 4); // 24 bytes
        assert!(s.fits(10, 240));
        assert!(!s.fits(10, 239));
        assert!(!s.fits(11, 240));
    }

    #[test]
    fn overflow_is_never_within_budget() {
        let huge = shape(u64::MAX / 2, 3);
        assert_eq!(huge.bytes(), None);
        assert!(!huge.fits(1, u64::MAX));
        assert!(!shape(1 << 31, 1 << 31).fits(u64::MAX, u64::MAX));
    }
}
