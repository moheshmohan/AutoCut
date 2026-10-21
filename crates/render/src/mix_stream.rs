//! Render streams: whose stateful audio a request continues.
//!
//! Most audio DSP is stateful (compressor and limiter envelopes, filter memories, delay lines,
//! reverb tails), while the mixer is pull-based: any range, in any order. Between requests the
//! mixer graph ([`crate::mixer`]), the clip effect chains ([`crate::audio_fx`]) and the
//! time-stretch memos ([`crate::audio`]) keep their state, keyed by the stream that made it and
//! the next sample it will produce, so a sequential consumer continues its own state exactly and
//! any other request starts fresh with a pre-roll.
//!
//! A [`MixStream`] is one sequential consumer: playback, an export, an audio preview render, one
//! measuring pass of a loudness analysis, the waveform of a nest. Each owns its stream and passes it
//! with every chunk, so two consumers rendering the same structure from the same position (two
//! sessions or two sequences alike, playback and a loudness measurement, parallel exports) never
//! continue each other's state: a limiter never starts from another signal's envelope, and what
//! a measurement hears is what the export plays. A stream renders one sequence of one project;
//! inside it, every nested sequence and its clips get their own sub-stream (by the clip that
//! holds them), so two clips of the same nest never share either.
//!
//! Dropping a stream drops the state it left behind.

use std::sync::atomic::{AtomicU64, Ordering};

/// One sequential consumer of rendered audio (see the module docs). Not `Clone`: a second consumer
/// makes its own.
#[derive(Debug)]
pub struct MixStream {
    id: u64,
}

/// Next stream id (0 is never handed out).
static NEXT: AtomicU64 = AtomicU64::new(1);

impl MixStream {
    /// A new stream, distinct from every other stream in the process.
    pub fn new() -> Self {
        MixStream { id: NEXT.fetch_add(1, Ordering::Relaxed) }
    }

    /// The key of the stream's top-level sequence.
    pub(crate) fn key(&self) -> StreamKey {
        StreamKey { id: self.id, path: 0 }
    }
}

impl Default for MixStream {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for MixStream {
    fn drop(&mut self) {
        crate::mixer::forget_stream(self.id);
        crate::audio_fx::forget_stream(self.id);
        crate::audio::forget_stream(self.id);
    }
}

/// Where in a stream a render runs: the stream, and the chain of nesting clips down to the sequence
/// being mixed (0 for the stream's own sequence). Part of every stateful cache key.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct StreamKey {
    pub(crate) id: u64,
    path: u64,
}

impl StreamKey {
    /// The sub-stream of the nested sequence that clip `clip` (a [`filmcraft_project::ClipId`])
    /// plays.
    pub(crate) fn nested(self, clip: u64) -> StreamKey {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (self.path, clip).hash(&mut h);
        // 0 stays the top level's
        StreamKey { id: self.id, path: h.finish().max(1) }
    }
}

/// A state that renders a stream from a known sample on.
pub(crate) trait Continues {
    /// The next sample this state will produce.
    fn next_out(&self) -> i64;
}

/// Stateful renders waiting for their stream's next request, by key (which holds a
/// [`StreamKey`]): at most `per_key` states per key (a stream that jumps leaves its old state
/// behind, and a jump back can continue it), and at most `keys` keys, the least recently used
/// going first.
pub(crate) struct StateCache<K, S> {
    map: std::collections::HashMap<K, Slot<S>>,
    clock: u64,
    keys: usize,
    per_key: usize,
}

struct Slot<S> {
    used: u64,
    /// Oldest first.
    states: Vec<S>,
}

impl<K: Eq + std::hash::Hash + Clone, S: Continues> StateCache<K, S> {
    pub(crate) fn new(keys: usize, per_key: usize) -> Self {
        StateCache { map: Default::default(), clock: 0, keys: keys.max(1), per_key: per_key.max(1) }
    }

    /// Take out the state of `key` that produces sample `next_out` next.
    pub(crate) fn take(&mut self, key: &K, next_out: i64) -> Option<S> {
        let slot = self.map.get_mut(key)?;
        let i = slot.states.iter().position(|s| s.next_out() == next_out)?;
        let st = slot.states.remove(i);
        if slot.states.is_empty() {
            self.map.remove(key);
        }
        Some(st)
    }

    /// Keep `state` for `key`'s next request.
    pub(crate) fn put(&mut self, key: K, state: S) {
        self.clock = self.clock.wrapping_add(1);
        let used = self.clock;
        if !self.map.contains_key(&key) && self.map.len() >= self.keys {
            let oldest = self.map.iter().min_by_key(|(_, s)| s.used).map(|(k, _)| k.clone());
            if let Some(k) = oldest {
                self.map.remove(&k);
            }
        }
        let slot = self.map.entry(key).or_insert_with(|| Slot { used, states: Vec::new() });
        slot.used = used;
        slot.states.push(state);
        if slot.states.len() > self.per_key {
            slot.states.remove(0);
        }
    }

    /// Drop every state whose key `gone` picks.
    pub(crate) fn forget(&mut self, gone: impl Fn(&K) -> bool) {
        self.map.retain(|k, _| !gone(k));
    }

    /// Keys held.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.map.len()
    }

    /// States held under the keys `pick` picks.
    #[cfg(test)]
    pub(crate) fn count(&self, pick: impl Fn(&K) -> bool) -> usize {
        self.map.iter().filter(|(k, _)| pick(k)).map(|(_, s)| s.states.len()).sum()
    }
}
