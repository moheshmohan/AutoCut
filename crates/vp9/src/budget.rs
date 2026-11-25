//! Memory budget for decoded pictures.
//!
//! A size cap does not bound memory: the decoder holds every reference frame, the frames whose
//! post stage (loop filter, output picture) is still running, and the output pictures being built
//! or waiting to be returned. Before it allocates for a frame the decoder adds those up for the
//! frame's size with checked arithmetic ([`Store`]) and compares them with a byte budget, so a
//! hostile header is an error, not an out-of-memory abort.

/// Bytes the decoded pictures of one decoder may take by default.
///
/// On 64-bit targets 12 GiB: more than a decoder can hold for any frame within its size limits
/// (2^26 luma samples, 16384 a side) in any profile and at any thread count (at most about
/// 12.1 GB: 4:4:4 at 12 bits, about 16384x4096, with 18 threads or more; a level 6.2 frame at
/// most about 6.5 GB). The default therefore refuses nothing those limits admit; it binds once a
/// caller raises them, or lowers it. On 32-bit targets 1 GiB (wasm32 has a 4 GiB address space
/// in all, and decodes on one thread).
pub const DECODE_MEMORY_BUDGET: u64 = if cfg!(target_pointer_width = "64") { 12 << 30 } else { 1 << 30 };

/// What a decoder holds at once at one frame size: `frames` frames of `frame_bytes` each (sample
/// bands; reference slots and frames in their post stage) and `pictures` output pictures of
/// `picture_bytes` each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Store {
    pub frames: u64,
    pub frame_bytes: u64,
    pub pictures: u64,
    pub picture_bytes: u64,
}

impl Store {
    /// Bytes of the whole store, `None` if it does not fit in a `u64`.
    pub fn bytes(&self) -> Option<u64> {
        self.frames.checked_mul(self.frame_bytes)?.checked_add(self.pictures.checked_mul(self.picture_bytes)?)
    }

    /// Whether the store fits in `budget` bytes.
    pub fn fits(&self, budget: u64) -> bool {
        self.bytes().is_some_and(|b| b <= budget)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bytes_count_frames_and_pictures() {
        let s = Store { frames: 9, frame_bytes: 100, pictures: 3, picture_bytes: 40 };
        assert_eq!(s.bytes(), Some(1020));
        assert!(s.fits(1020));
        assert!(!s.fits(1019));
        assert!(!Store { frames: 10, ..s }.fits(1020));
    }

    #[test]
    fn overflow_is_never_within_budget() {
        let huge = Store { frames: 14, frame_bytes: u64::MAX / 2, pictures: 0, picture_bytes: 0 };
        assert_eq!(huge.bytes(), None);
        assert!(!huge.fits(u64::MAX));
        let sum = Store { frames: 1, frame_bytes: u64::MAX, pictures: 1, picture_bytes: 1 };
        assert!(!sum.fits(u64::MAX));
    }
}
