//! The pictures decoded ahead of the render (job.rs, "Decoding ahead").
//!
//! The thread decoding the batch after next asks the sources through [`Recording`], which keeps
//! every picture it gets in [`Pictures`]; the render asks through [`Serving`], which answers a
//! request that was decoded ahead from there. Without it the render's request for a picture that
//! is already decoded still had to take the source's GOP-cache lock, which the decoding-ahead
//! thread holds while it decodes the next pictures: the render threads queued behind it.
//!
//! A source gives the same picture for the same request (same media time and scale), so a picture
//! from the table is the one the render's own request would have returned.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use filmcraft_frame::{AudioBuffer, VideoFrame};
use filmcraft_media::{FrameRequest, MediaInfo, MediaSource, SharedSource};
use filmcraft_project::ItemId;
use filmcraft_render::SourceProvider;

/// Most pictures kept (two batches of frames of a timeline of many layers fit many times over); a
/// request beyond it is simply not kept.
const MAX_PICTURES: usize = 4096;

/// A request of one source: the item, the media time (ticks) and the scale's bits.
type Key = (ItemId, i64, u32);

/// Pictures decoded ahead, by request, with the output frame they were decoded for.
#[derive(Default)]
pub(crate) struct Pictures {
    by_request: Mutex<HashMap<Key, (i64, Arc<VideoFrame>)>>,
}

impl Pictures {
    fn get(&self, key: &Key) -> Option<Arc<VideoFrame>> {
        self.by_request.lock().unwrap_or_else(PoisonError::into_inner).get(key).map(|(_, f)| f.clone())
    }

    fn keep(&self, key: Key, frame: i64, pic: &Arc<VideoFrame>) {
        let mut m = self.by_request.lock().unwrap_or_else(PoisonError::into_inner);
        if m.len() < MAX_PICTURES || m.contains_key(&key) {
            m.insert(key, (frame, pic.clone()));
        }
    }

    /// Forget the pictures of output frames before `first` (rendered already).
    pub(crate) fn drop_before(&self, first: i64) {
        self.by_request.lock().unwrap_or_else(PoisonError::into_inner).retain(|_, (f, _)| *f >= first);
    }

    /// Pictures kept now.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.by_request.lock().unwrap_or_else(PoisonError::into_inner).len()
    }
}

/// What a wrapped source does with its pictures.
#[derive(Clone, Copy)]
enum Mode {
    /// Keep every picture it decodes, for output frame `.0`.
    Record(i64),
    /// Answer from the pictures decoded ahead first.
    Serve,
}

/// The sources of `inner`, keeping the pictures they decode for output frame `frame`.
pub(crate) struct Recording<'a> {
    pub inner: &'a dyn SourceProvider,
    pub pictures: &'a Arc<Pictures>,
    pub frame: i64,
}

/// The sources of `inner`, answering from the pictures decoded ahead.
pub(crate) struct Serving<'a> {
    pub inner: &'a dyn SourceProvider,
    pub pictures: &'a Arc<Pictures>,
}

impl SourceProvider for Recording<'_> {
    fn source(&self, item: ItemId) -> Option<SharedSource> {
        let inner = self.inner.source(item)?;
        Some(Arc::new(Wrapped { inner, item, pictures: self.pictures.clone(), mode: Mode::Record(self.frame) }))
    }
}

impl SourceProvider for Serving<'_> {
    fn source(&self, item: ItemId) -> Option<SharedSource> {
        let inner = self.inner.source(item)?;
        Some(Arc::new(Wrapped { inner, item, pictures: self.pictures.clone(), mode: Mode::Serve }))
    }
}

struct Wrapped {
    inner: SharedSource,
    item: ItemId,
    pictures: Arc<Pictures>,
    mode: Mode,
}

impl MediaSource for Wrapped {
    fn info(&self) -> &MediaInfo {
        self.inner.info()
    }

    fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
        let key = (self.item, req.time.0, req.scale.to_bits());
        match self.mode {
            Mode::Serve => match self.pictures.get(&key) {
                Some(pic) => Ok(pic),
                None => self.inner.video_frame(req),
            },
            Mode::Record(frame) => {
                let pic = self.inner.video_frame(req)?;
                self.pictures.keep(key, frame, &pic);
                Ok(pic)
            }
        }
    }

    fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
        self.inner.audio(start, frames, sample_rate)
    }

    fn audio_stream(&self, stream: usize, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
        self.inner.audio_stream(stream, start, frames, sample_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_media::generators::GeneratorSource;
    use filmcraft_media::{Generator, MediaSource};
    use filmcraft_render::SourceMap;
    use filmcraft_time::{FrameRate, TICKS_PER_SECOND, Tick};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts the pictures asked of the source underneath.
    struct Counting {
        inner: GeneratorSource,
        asked: AtomicUsize,
    }

    impl MediaSource for Counting {
        fn info(&self) -> &MediaInfo {
            self.inner.info()
        }
        fn video_frame(&self, req: FrameRequest) -> filmcraft_media::Result<Arc<VideoFrame>> {
            self.asked.fetch_add(1, Ordering::SeqCst);
            self.inner.video_frame(req)
        }
        fn audio(&self, start: i64, frames: usize, sample_rate: u32) -> filmcraft_media::Result<AudioBuffer> {
            self.inner.audio(start, frames, sample_rate)
        }
    }

    #[test]
    fn pictures_decoded_ahead_answer_the_same_requests_only() {
        let g = GeneratorSource::new(Generator::ColorMatte { color: [0.0, 1.0, 0.0, 1.0] }, 64, 36, FrameRate::FPS_24, Tick(TICKS_PER_SECOND));
        let counting = Arc::new(Counting { inner: g, asked: AtomicUsize::new(0) });
        let id = ItemId(7);
        let map = SourceMap([(id, counting.clone() as SharedSource)].into_iter().collect());
        let pictures = Arc::new(Pictures::default());
        let req = FrameRequest { time: Tick(1000), scale: 0.5 };
        let ahead = Recording { inner: &map, pictures: &pictures, frame: 3 }.source(id).unwrap().video_frame(req).unwrap();
        assert_eq!(counting.asked.load(Ordering::SeqCst), 1);
        let serving = Serving { inner: &map, pictures: &pictures };
        let src = serving.source(id).unwrap();
        // the same request: the very picture, without asking the source again
        assert!(Arc::ptr_eq(&src.video_frame(req).unwrap(), &ahead));
        assert_eq!(counting.asked.load(Ordering::SeqCst), 1);
        // another time, another scale or another item go to the source
        src.video_frame(FrameRequest { time: Tick(2000), scale: 0.5 }).unwrap();
        src.video_frame(FrameRequest { time: Tick(1000), scale: 1.0 }).unwrap();
        assert_eq!(counting.asked.load(Ordering::SeqCst), 3);
        assert!(serving.source(ItemId(8)).is_none());
        assert_eq!(src.info().name, counting.info().name);
        // frames rendered are forgotten
        pictures.drop_before(3);
        assert_eq!(pictures.len(), 1);
        pictures.drop_before(4);
        assert_eq!(pictures.len(), 0);
        src.video_frame(req).unwrap();
        assert_eq!(counting.asked.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn the_table_is_capped() {
        let g = GeneratorSource::new(Generator::ColorMatte { color: [0.0, 0.0, 1.0, 1.0] }, 8, 8, FrameRate::FPS_24, Tick(TICKS_PER_SECOND));
        let id = ItemId(1);
        let map = SourceMap([(id, Arc::new(g) as SharedSource)].into_iter().collect());
        let pictures = Arc::new(Pictures::default());
        let src = Recording { inner: &map, pictures: &pictures, frame: 0 }.source(id).unwrap();
        for t in 0..(MAX_PICTURES as i64 + 50) {
            src.video_frame(FrameRequest { time: Tick(t), scale: 1.0 }).unwrap();
        }
        assert_eq!(pictures.len(), MAX_PICTURES);
    }
}
