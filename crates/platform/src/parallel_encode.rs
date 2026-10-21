//! Parallel hardware encoding: one picture stream cut into segments of whole closed GOPs, the
//! segments encoded by several sessions ("lanes") at once, and the compressed frames joined back
//! into one stream in presentation order. Safe code, generic over the session ([`LaneEncoder`]),
//! so the joining is tested on every system with a stand-in encoder.
//!
//! Chips with two media engines (Apple's M-series Max and Ultra) encode two streams at once, while
//! one session runs on one engine at a time (`hardware_encode` decides where it pays:
//! `filmcraft_export::EncodeSessions`). How a stream is cut and joined:
//!
//! - **Segments.** Pictures `k·S .. (k+1)·S` form segment `k` (`S` = the keyframe interval, so the
//!   GOP structure is the one a single session writes) and go to lane `k mod n`. The first
//!   picture of every segment is coded as a keyframe (an IDR picture) and nothing is reordered,
//!   so every segment is a closed GOP that decodes on its own: there is nothing to mend at a join.
//! - **One timeline per lane.** A lane numbers its own pictures `0, 1, 2…` one frame apart, so its
//!   rate control sees a stream at the export's frame rate and bitrate (with gaps it would see
//!   half the frames over the same time). Every lane has the same configuration, so the joined
//!   stream averages the same bitrate.
//! - **The same sample description.** The container stores one set of parameter sets (`avcC` /
//!   `hvcC`): the first picture each lane gets waits for its parameter sets, which must equal the
//!   first lane's. A lane that disagrees is retired before it gave a frame to the stream, its
//!   picture goes to the first lane, and the export continues with fewer lanes.
//! - **Presentation order.** Lanes return their frames in their own order; the joiner hands them
//!   out segment after segment, checking that each segment starts with a keyframe and that each lane
//!   returned exactly one frame per picture. Frames of a later segment wait (compressed) until the
//!   earlier ones are out, so the muxer receives them up to about a segment late.
//! - **A warm start.** A session's rate control starts cold: its first pictures get few bits (4.6
//!   dB below a running session at the same point of a 4K H.264 stream). A lane that has not
//!   encoded anything yet therefore also encodes the end of the segment before its first one (the
//!   last third, at least 20 pictures: as good as all of it on 4K footage), with its frames dropped,
//!   in what is otherwise its idle time: its first segment then starts as if it had been running all
//!   along (the same PSNR at the join as one session), and its parameter sets are checked before it
//!   gives a frame to the stream.
//! - **Memory.** Pictures are produced one after the other (the export renders in order), so while one
//!   lane takes the segment being produced, the other works through the pictures queued for its
//!   own: about half a segment. [`Joiner::new`]'s `budget` caps the pictures handed to the lanes
//!   and not encoded yet (the caller sizes it in bytes; a picture a lane warms up on is shared, not
//!   copied); the producer waits when it is reached.
//! - **Threads.** Each lane runs on a thread of its own under `catch_unwind`; a panic or an encoder
//!   error becomes an error of the export, which stops (an encoder cannot hand a half-written stream
//!   to another one). Dropping the joiner closes the lanes and joins their threads.

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError, channel};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::thread::JoinHandle;
use std::time::Duration;

/// One compressed frame from a lane, in its decoding order (= presentation order).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LanePacket {
    pub data: Vec<u8>,
    pub key: bool,
}

/// A session the joiner drives from a thread of its own.
pub trait LaneEncoder: Send + 'static {
    /// The codec configuration the container stores for the stream (its parameter sets): every
    /// lane must produce the same.
    type Params: Clone + PartialEq + Send + 'static;

    /// Submit picture `local` of this lane (Y', then Cb, then Cr, tightly packed), coded as a
    /// keyframe when `keyframe`, and return the frames that came out meanwhile, in order. The
    /// first call returns once the parameter sets are known ([`Self::params`]).
    fn encode(&mut self, local: u64, planes: &[u8], keyframe: bool) -> Result<Vec<LanePacket>, String>;
    /// Finish every pending frame and return them.
    fn flush(&mut self) -> Result<Vec<LanePacket>, String>;
    /// The parameter sets, once the first frame is out.
    fn params(&self) -> Option<Self::Params>;
}

/// A picture on its way through the lanes (shared by a lane and one warming up on it). Counted
/// against the budget while it lives; its buffer is recycled when the last holder drops it.
struct Picture {
    planes: Vec<u8>,
    gate: Arc<Gate>,
    spare: Sender<Vec<u8>>,
}

impl Drop for Picture {
    fn drop(&mut self) {
        let _ = self.spare.send(std::mem::take(&mut self.planes));
        self.gate.take();
    }
}

/// Work for a lane.
enum Job {
    /// `keep`: the frame belongs to the stream (else it is a warm-up picture, see [`Joiner::push`]).
    Picture {
        local: u64,
        picture: Arc<Picture>,
        keyframe: bool,
        keep: bool,
    },
    Flush,
}

/// What a lane reports, in order.
enum Event<P> {
    /// After the first picture: its parameter sets, and the picture back (in case the lane is
    /// retired and the picture must go to another one).
    First {
        params: P,
        picture: Arc<Picture>,
        keep: bool,
    },
    Packets(Vec<LanePacket>),
    Flushed(Vec<LanePacket>),
    Failed(String),
}

/// Pictures handed to the lanes and not encoded yet, against the budget.
#[derive(Default)]
struct Gate {
    queued: Mutex<usize>,
    changed: Condvar,
}

impl Gate {
    fn add(&self) {
        let mut q = self.queued.lock().unwrap_or_else(PoisonError::into_inner);
        *q = q.saturating_add(1);
    }

    fn take(&self) {
        let mut q = self.queued.lock().unwrap_or_else(PoisonError::into_inner);
        *q = q.saturating_sub(1);
        drop(q);
        self.changed.notify_all();
    }

    /// Wait until fewer than `budget` pictures are alive, or `timeout` passed; whether there is room.
    fn room(&self, budget: usize, timeout: Duration) -> bool {
        let q = self.queued.lock().unwrap_or_else(PoisonError::into_inner);
        if *q < budget {
            return true;
        }
        let (q, _) = self.changed.wait_timeout(q, timeout).unwrap_or_else(PoisonError::into_inner);
        *q < budget
    }
}

struct Lane<P> {
    /// `None` once retired (or after the joiner finished).
    jobs: Option<Sender<Job>>,
    events: Receiver<Event<P>>,
    thread: Option<JoinHandle<()>>,
    /// Pictures sent so far: the next one's number on this lane's timeline.
    sent: u64,
    /// Frames received and not handed out yet.
    ready: VecDeque<LanePacket>,
    /// The parameter sets were checked (after its first picture).
    verified: bool,
    /// Gave nothing to the stream and takes nothing more (its parameter sets differed).
    retired: bool,
    flushed: bool,
}

/// A segment of the stream: which lane encodes it and how far it got.
struct Segment {
    lane: usize,
    pushed: u64,
    emitted: u64,
    /// No more pictures will be added.
    closed: bool,
}

/// How long the producer waits for the lanes between looks at what they reported.
const POLL: Duration = Duration::from_millis(20);

/// The longest a lane may take for its first picture or to finish its frames before it counts as
/// stuck (an error, not a hung export).
const FINISH: Duration = Duration::from_secs(60);

/// Splits a picture stream over lanes and joins their frames (see the module docs).
pub struct Joiner<E: LaneEncoder> {
    lanes: Vec<Lane<E::Params>>,
    segment: u64,
    budget: usize,
    gate: Arc<Gate>,
    /// Set when the joiner is dropped: the lanes leave what is queued (a cancelled export).
    stop: Arc<AtomicBool>,
    /// Picture buffers the lanes are done with.
    spare: Receiver<Vec<u8>>,
    spare_tx: Sender<Vec<u8>>,
    order: VecDeque<Segment>,
    /// The lane warming up on the current segment's pictures (see [`Joiner::push`]).
    warming: Option<usize>,
    /// Pictures at the end of a segment the next segment's lane warms up on, if it has not started.
    warm: u64,
    /// The next picture's number in the stream.
    next: u64,
    /// The first lane's parameter sets: the stream's.
    params: Option<E::Params>,
    /// The first error; every later call returns it.
    failed: Option<String>,
    finished: bool,
}

/// The frames of `packets` that belong to the stream: each frame answers the oldest picture in
/// `keep` (a lane returns one frame per picture, in order); frames beyond the pictures are kept, for
/// the joiner to report.
fn kept(packets: Vec<LanePacket>, keep: &mut VecDeque<bool>) -> Vec<LanePacket> {
    packets.into_iter().filter(|_| keep.pop_front().unwrap_or(true)).collect()
}

fn run_lane<E: LaneEncoder>(mut enc: E, jobs: &Receiver<Job>, events: &Sender<Event<E::Params>>, stop: &AtomicBool) {
    let mut first = true;
    let mut pending = VecDeque::new();
    while let Ok(job) = jobs.recv() {
        if stop.load(Ordering::Relaxed) {
            return;
        }
        match job {
            Job::Picture { local, picture, keyframe, keep } => {
                pending.push_back(keep);
                let packets = match enc.encode(local, &picture.planes, keyframe) {
                    Ok(p) => kept(p, &mut pending),
                    Err(why) => {
                        let _ = events.send(Event::Failed(why));
                        return;
                    }
                };
                if first {
                    first = false;
                    let Some(params) = enc.params() else {
                        let _ = events.send(Event::Failed("the encoder gave no parameter sets with its first frame".into()));
                        return;
                    };
                    let _ = events.send(Event::First { params, picture, keep });
                }
                if !packets.is_empty() {
                    let _ = events.send(Event::Packets(packets));
                }
            }
            Job::Flush => match enc.flush() {
                Ok(p) => {
                    let _ = events.send(Event::Flushed(kept(p, &mut pending)));
                }
                Err(why) => {
                    let _ = events.send(Event::Failed(why));
                    return;
                }
            },
        }
    }
}

impl<E: LaneEncoder> Joiner<E> {
    /// Start a lane thread for each encoder. `segment`: pictures per segment (at least 2, so two
    /// IDR pictures never follow each other across a join); `budget`: pictures in the lanes at most.
    pub fn new(encoders: Vec<E>, segment: u64, budget: usize) -> Result<Self, String> {
        if encoders.is_empty() {
            return Err("no encoder sessions".into());
        }
        if segment < 2 {
            return Err(format!("segments of {segment} pictures"));
        }
        let gate = Arc::new(Gate::default());
        let stop = Arc::new(AtomicBool::new(false));
        let (spare_tx, spare) = channel();
        let mut lanes = Vec::with_capacity(encoders.len());
        for (k, enc) in encoders.into_iter().enumerate() {
            let (jobs_tx, jobs_rx) = channel::<Job>();
            let (events_tx, events_rx) = channel::<Event<E::Params>>();
            let stopped = stop.clone();
            let thread = std::thread::Builder::new()
                .name(format!("filmcraft-encode-lane-{k}"))
                .spawn(move || {
                    let ran = catch_unwind(AssertUnwindSafe(|| run_lane(enc, &jobs_rx, &events_tx, &stopped)));
                    if ran.is_err() {
                        // (the panic hook has logged the panic itself)
                        let _ = events_tx.send(Event::Failed("internal error in a hardware encoding thread".into()));
                    }
                })
                .map_err(|e| format!("cannot start an encoding thread: {e}"))?;
            lanes.push(Lane {
                jobs: Some(jobs_tx),
                events: events_rx,
                thread: Some(thread),
                sent: 0,
                ready: VecDeque::new(),
                verified: false,
                retired: false,
                flushed: false,
            });
        }
        Ok(Joiner {
            lanes,
            segment,
            budget: budget.max(1),
            gate,
            stop,
            spare,
            spare_tx,
            order: VecDeque::new(),
            warming: None,
            warm: segment.min((segment / 3).max(20)),
            next: 0,
            params: None,
            failed: None,
            finished: false,
        })
    }

    /// The stream's parameter sets (the first lane's), known once its first picture is encoded.
    pub fn params(&self) -> Option<&E::Params> {
        self.params.as_ref()
    }

    /// Lanes taking pictures (not retired).
    pub fn active_lanes(&self) -> usize {
        self.lanes.iter().filter(|l| !l.retired).count()
    }

    fn fail<T>(&mut self, why: String) -> Result<T, String> {
        if self.failed.is_none() {
            self.failed = Some(why.clone());
        }
        Err(why)
    }

    /// The lane for segment `k`: `k mod n` over the lanes still taking pictures.
    fn lane_for(&self, k: u64) -> usize {
        let active: Vec<usize> = self.lanes.iter().enumerate().filter(|(_, l)| !l.retired).map(|(i, _)| i).collect();
        let n = u64::try_from(active.len()).unwrap_or(1).max(1);
        usize::try_from(k % n).ok().and_then(|i| active.get(i).copied()).unwrap_or(0)
    }

    /// Picture `index` of the stream (pictures come in order, from 0): Y' (`y`), Cb and Cr planes.
    /// Returns the frames of the joined stream that are ready, in order. The lane of the next
    /// segment warms up on this one when it has not encoded anything yet (module docs).
    pub fn push(&mut self, index: u64, y: &[u8], u: &[u8], v: &[u8]) -> Result<Vec<LanePacket>, String> {
        if let Some(why) = &self.failed {
            return Err(why.clone());
        }
        if self.finished {
            return self.fail("internal: a picture after the end of the stream".into());
        }
        if index != self.next {
            return self.fail(format!("internal: picture {index} came when picture {} was due", self.next));
        }
        let (segment, start) = (index / self.segment, index.is_multiple_of(self.segment));
        if start {
            if let Some(last) = self.order.back_mut() {
                last.closed = true;
            }
            let lane = self.lane_for(segment);
            self.order.push_back(Segment { lane, pushed: 0, emitted: 0, closed: false });
            self.warming = None;
        }
        let Some(lane) = self.order.back().map(|s| s.lane) else {
            return self.fail("internal: no segment for the picture".into());
        };
        self.wait_for_room()?;
        let mut planes = self.spare.try_recv().unwrap_or_default();
        planes.clear();
        planes.reserve_exact(y.len().saturating_add(u.len()).saturating_add(v.len()));
        planes.extend_from_slice(y);
        planes.extend_from_slice(u);
        planes.extend_from_slice(v);
        self.gate.add();
        let picture = Arc::new(Picture { planes, gate: self.gate.clone(), spare: self.spare_tx.clone() });
        let lane = self.send(lane, picture.clone(), start, true)?;
        if let Some(seg) = self.order.back_mut() {
            seg.lane = lane;
            seg.pushed = seg.pushed.saturating_add(1);
        }
        // the lane of the next segment warms up on this one when it has not started yet
        let next = self.lane_for(segment.saturating_add(1));
        let fresh = self.lanes.get(next).is_some_and(|l| !l.retired && (l.sent == 0 || self.warming == Some(next)));
        let window = index % self.segment >= self.segment.saturating_sub(self.warm);
        if next != lane && fresh && window {
            self.warming = Some(next);
            self.send(next, picture, start, false)?;
        }
        self.next = self.next.saturating_add(1);
        self.poll()?;
        self.emit()
    }

    /// Send a picture to `lane` (its first one waits for, and checks, its parameter sets; a lane
    /// that disagrees is retired, and a picture of the stream goes to the first lane instead).
    /// Returns the lane that took it. `keep`: a picture of the stream, not a warm-up.
    fn send(&mut self, lane: usize, picture: Arc<Picture>, keyframe: bool, keep: bool) -> Result<usize, String> {
        let Some(l) = self.lanes.get_mut(lane) else {
            return self.fail(format!("internal: no encoding lane {lane}"));
        };
        let local = l.sent;
        let sent = l.jobs.as_ref().is_some_and(|jobs| jobs.send(Job::Picture { local, picture, keyframe, keep }).is_ok());
        if !sent {
            return self.fail(self.stopped(lane));
        }
        l.sent = local.saturating_add(1);
        if l.verified {
            return Ok(lane);
        }
        // the lane's first picture: wait for its parameter sets
        loop {
            let event = self.lanes.get(lane).map(|l| l.events.recv_timeout(FINISH));
            match event {
                Some(Ok(Event::First { params, picture, keep })) => {
                    if let Some(l) = self.lanes.get_mut(lane) {
                        l.verified = true;
                    }
                    match &self.params {
                        // (the first lane's first picture is one of the stream)
                        None if keep => self.params = Some(params),
                        Some(p) if *p == params => {}
                        _ => return self.retire(lane, picture, keep),
                    }
                    return Ok(lane);
                }
                Some(Ok(Event::Packets(p))) => {
                    if let Some(l) = self.lanes.get_mut(lane) {
                        l.ready.extend(p);
                    }
                }
                Some(Ok(Event::Flushed(_))) => return self.fail("internal: an encoding lane finished early".into()),
                Some(Ok(Event::Failed(why))) => return self.fail(why),
                Some(Err(RecvTimeoutError::Timeout)) => {
                    return self.fail(format!("hardware encoding lane {lane} did not encode its first picture within a minute"));
                }
                Some(Err(RecvTimeoutError::Disconnected)) | None => return self.fail(self.stopped(lane)),
            }
        }
    }

    /// Why `lane` stopped: the error it reported, if it did.
    fn stopped(&self, lane: usize) -> String {
        if let Some(l) = self.lanes.get(lane) {
            while let Ok(event) = l.events.try_recv() {
                if let Event::Failed(why) = event {
                    return why;
                }
            }
        }
        format!("hardware encoding lane {lane} stopped")
    }

    /// Lane `lane` made other parameter sets than the first lane: it gave nothing to the stream
    /// yet (this was its first picture), so it is retired, and a picture of the stream (`keep`)
    /// goes to the first lane.
    fn retire(&mut self, lane: usize, picture: Arc<Picture>, keep: bool) -> Result<usize, String> {
        if lane == 0 {
            return self.fail("internal: the first encoding lane cannot be retired".into());
        }
        if let Some(l) = self.lanes.get_mut(lane) {
            log::warn!("hardware encoding lane {lane} made other parameter sets than lane 0: it is not used");
            l.retired = true;
            l.ready.clear();
            // closing its queue ends its thread
            l.jobs = None;
        }
        if self.warming == Some(lane) {
            self.warming = None;
        }
        if keep { self.send(0, picture, true, true) } else { Ok(0) }
    }

    /// Wait until the pictures in the lanes are within the budget, collecting what the lanes report.
    fn wait_for_room(&mut self) -> Result<(), String> {
        while !self.gate.room(self.budget, POLL) {
            self.poll()?;
        }
        Ok(())
    }

    /// Take what the lanes reported so far (without waiting).
    fn poll(&mut self) -> Result<(), String> {
        for k in 0..self.lanes.len() {
            while let Some(l) = self.lanes.get_mut(k) {
                if l.retired {
                    // a retired lane's frames are dropped, and its thread ends on its own
                    while l.events.try_recv().is_ok() {}
                    break;
                }
                match l.events.try_recv() {
                    Ok(Event::Packets(p)) => l.ready.extend(p),
                    Ok(Event::Flushed(p)) => {
                        l.ready.extend(p);
                        l.flushed = true;
                    }
                    Ok(Event::First { .. }) => return self.fail("internal: an encoding lane reported its parameter sets twice".into()),
                    Ok(Event::Failed(why)) => return self.fail(why),
                    Err(TryRecvError::Empty) => break,
                    // the thread ended without saying why, though it was still given work
                    Err(TryRecvError::Disconnected) if l.jobs.is_some() => return self.fail(self.stopped(k)),
                    Err(TryRecvError::Disconnected) => break,
                }
            }
        }
        Ok(())
    }

    /// Hand out the frames that are next in presentation order.
    fn emit(&mut self) -> Result<Vec<LanePacket>, String> {
        let mut out = Vec::new();
        while let Some(seg) = self.order.front_mut() {
            let Some(lane) = self.lanes.get_mut(seg.lane) else {
                let why = format!("internal: no encoding lane {}", seg.lane);
                return self.fail(why);
            };
            while seg.emitted < seg.pushed {
                let Some(p) = lane.ready.pop_front() else { break };
                if seg.emitted == 0 && !p.key {
                    return self.fail("the hardware encoder did not start a segment with a keyframe".into());
                }
                seg.emitted += 1;
                out.push(p);
            }
            if seg.closed && seg.emitted == seg.pushed {
                self.order.pop_front();
            } else {
                break;
            }
        }
        Ok(out)
    }

    /// Finish the stream: every lane completes its frames, and the rest of the joined stream comes
    /// out. Checks that every lane returned one frame per picture.
    pub fn finish(&mut self) -> Result<Vec<LanePacket>, String> {
        if let Some(why) = &self.failed {
            return Err(why.clone());
        }
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        if let Some(last) = self.order.back_mut() {
            last.closed = true;
        }
        for k in 0..self.lanes.len() {
            let Some(l) = self.lanes.get_mut(k) else { continue };
            if l.retired || l.sent == 0 {
                l.flushed = true;
                continue;
            }
            if !l.jobs.as_ref().is_some_and(|jobs| jobs.send(Job::Flush).is_ok()) {
                return self.fail(self.stopped(k));
            }
        }
        for k in 0..self.lanes.len() {
            while let Some(l) = self.lanes.get_mut(k) {
                if l.flushed {
                    break;
                }
                match l.events.recv_timeout(FINISH) {
                    Ok(Event::Packets(p)) => l.ready.extend(p),
                    Ok(Event::Flushed(p)) => {
                        l.ready.extend(p);
                        l.flushed = true;
                    }
                    Ok(Event::First { .. }) => return self.fail("internal: an encoding lane reported its parameter sets twice".into()),
                    Ok(Event::Failed(why)) => return self.fail(why),
                    Err(RecvTimeoutError::Timeout) => return self.fail(format!("hardware encoding lane {k} did not finish its frames within a minute")),
                    Err(RecvTimeoutError::Disconnected) => return self.fail(self.stopped(k)),
                }
            }
        }
        let out = self.emit()?;
        if !self.order.is_empty() {
            return self.fail("a hardware encoding lane returned fewer frames than it was given".into());
        }
        if self.lanes.iter().any(|l| !l.retired && !l.ready.is_empty()) {
            return self.fail("a hardware encoding lane returned more frames than it was given".into());
        }
        Ok(out)
    }
}

impl<E: LaneEncoder> Drop for Joiner<E> {
    /// Ends the lanes: they leave the pictures still queued (the stream is not finished, or it was
    /// and nothing is queued), close their sessions, and their threads are joined.
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        for l in &mut self.lanes {
            l.jobs = None;
        }
        for l in &mut self.lanes {
            if let Some(t) = l.thread.take() {
                let _ = t.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A stand-in session: each picture becomes a frame holding (lane, local number, first byte
    /// of the picture); a keyframe when asked, or every `keyint` pictures of its own timeline.
    struct Fake {
        lane: u8,
        params: u32,
        keyint: u64,
        /// Frames held back until the next picture or the flush (an encoder's delay).
        held: Vec<LanePacket>,
        delay: usize,
        fail_at: Option<u64>,
        panic_at: Option<u64>,
        extra_on_flush: bool,
        no_key: bool,
        seen: Arc<AtomicUsize>,
    }

    fn fake(lane: u8) -> Fake {
        Fake {
            lane,
            params: 7,
            keyint: 1000,
            held: Vec::new(),
            delay: 0,
            fail_at: None,
            panic_at: None,
            extra_on_flush: false,
            no_key: false,
            seen: Arc::default(),
        }
    }

    impl LaneEncoder for Fake {
        type Params = u32;
        fn encode(&mut self, local: u64, planes: &[u8], keyframe: bool) -> Result<Vec<LanePacket>, String> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            if self.fail_at == Some(local) {
                return Err(format!("lane {} failed", self.lane));
            }
            if self.panic_at == Some(local) {
                panic!("lane {} panicked", self.lane);
            }
            let key = !self.no_key && (keyframe || local.is_multiple_of(self.keyint));
            let mut data = vec![self.lane];
            data.extend_from_slice(&local.to_le_bytes());
            data.push(planes.first().copied().unwrap_or(0));
            self.held.push(LanePacket { data, key });
            let keep = self.held.len().saturating_sub(self.delay);
            Ok(self.held.drain(..keep).collect())
        }
        fn flush(&mut self) -> Result<Vec<LanePacket>, String> {
            let mut out = std::mem::take(&mut self.held);
            if self.extra_on_flush {
                out.push(LanePacket { data: vec![0xff], key: false });
            }
            Ok(out)
        }
        fn params(&self) -> Option<u32> {
            Some(self.params)
        }
    }

    fn picture(i: u64) -> [u8; 3] {
        [i as u8, 0, 0]
    }

    fn run(j: &mut Joiner<Fake>, n: u64) -> Result<Vec<LanePacket>, String> {
        let mut out = Vec::new();
        for i in 0..n {
            let p = picture(i);
            out.extend(j.push(i, &p[..1], &p[1..2], &p[2..])?);
        }
        out.extend(j.finish()?);
        Ok(out)
    }

    /// (lane, local number, picture byte) of each frame.
    fn frames(out: &[LanePacket]) -> Vec<(u8, u64, u8)> {
        out.iter()
            .map(|p| {
                let local = u64::from_le_bytes(p.data[1..9].try_into().unwrap());
                (p.data[0], local, p.data[9])
            })
            .collect()
    }

    #[test]
    fn segments_alternate_and_join_in_order() {
        for (lanes, n, segment, delay, budget) in
            [(2usize, 100u64, 10u64, 0usize, 4usize), (2, 37, 6, 3, 1), (3, 95, 8, 2, 50), (1, 20, 4, 1, 2), (2, 5, 10, 0, 3), (2, 200, 45, 1, 8)]
        {
            let encs = (0..lanes).map(|k| Fake { delay, ..fake(k as u8) }).collect();
            let mut j = Joiner::new(encs, segment, budget).unwrap();
            let out = run(&mut j, n).unwrap();
            let f = frames(&out);
            assert_eq!(f.len() as u64, n, "one frame per picture");
            for (i, (lane, local, byte)) in f.iter().enumerate() {
                let i = i as u64;
                let seg = i / segment;
                assert_eq!(*byte, i as u8, "presentation order");
                assert_eq!(u64::from(*lane), seg % lanes as u64, "segment {seg} on its lane");
                // the lane's own timeline: the segment before its first (a warm-up, dropped) unless
                // it is the first lane, its earlier segments, then the place in this one
                let before = (0..seg).filter(|s| s % lanes as u64 == seg % lanes as u64).count() as u64;
                let warm = if *lane == 0 { 0 } else { segment.min((segment / 3).max(20)) };
                assert_eq!(*local, warm + before * segment + i % segment, "picture {i}");
                assert_eq!(out[i as usize].key, i.is_multiple_of(segment), "keyframes exactly at segment starts (picture {i})");
            }
            assert_eq!(j.params(), Some(&7));
        }
    }

    #[test]
    fn an_empty_stream_and_a_single_picture() {
        let mut j = Joiner::new(vec![fake(0), fake(1)], 4, 4).unwrap();
        assert!(run(&mut j, 0).unwrap().is_empty());
        assert_eq!(j.params(), None);
        let mut j = Joiner::new(vec![fake(0), fake(1)], 4, 4).unwrap();
        assert_eq!(frames(&run(&mut j, 1).unwrap()), vec![(0, 0, 0)]);
    }

    #[test]
    fn a_lane_with_other_parameter_sets_is_retired_before_it_gives_a_frame() {
        let mut j = Joiner::new(vec![fake(0), Fake { params: 8, ..fake(1) }], 5, 3).unwrap();
        let out = run(&mut j, 23).unwrap();
        let f = frames(&out);
        assert_eq!(f.len(), 23);
        assert!(f.iter().all(|(lane, _, _)| *lane == 0), "only the first lane's frames: {f:?}");
        assert_eq!(f.iter().map(|(_, l, _)| *l).collect::<Vec<_>>(), (0..23).collect::<Vec<_>>());
        assert_eq!(f.iter().map(|(_, _, b)| *b).collect::<Vec<_>>(), (0..23).map(|i| i as u8).collect::<Vec<_>>());
        assert!(out.iter().enumerate().all(|(i, p)| p.key == (i % 5 == 0)), "segments still start with keyframes");
        assert_eq!(j.active_lanes(), 1);
        // three lanes, the middle one disagreeing: segments go to lanes 0 and 2
        let mut j = Joiner::new(vec![fake(0), Fake { params: 9, ..fake(1) }, fake(2)], 4, 8).unwrap();
        let f = frames(&run(&mut j, 40).unwrap());
        assert_eq!(f.len(), 40);
        assert!(f.iter().all(|(lane, _, _)| *lane != 1));
        assert_eq!(f.iter().map(|(_, _, b)| *b).collect::<Vec<_>>(), (0..40).map(|i| i as u8).collect::<Vec<_>>());
    }

    #[test]
    fn failures_stop_the_stream_with_an_error() {
        // an encoder error on either lane, at the start or later
        for (lane, at) in [(0usize, 0u64), (0, 7), (1, 0), (1, 3)] {
            let mut encs = vec![fake(0), fake(1)];
            encs[lane].fail_at = Some(at);
            let mut j = Joiner::new(encs, 4, 2).unwrap();
            let err = run(&mut j, 40).unwrap_err();
            assert!(err.contains("failed"), "{err}");
            // and stays failed
            assert!(j.push(41, &[0], &[0], &[0]).is_err());
            assert!(j.finish().is_err());
        }
        // a panic in a lane is an error, not a crash
        let mut encs = vec![fake(0), fake(1)];
        encs[1].panic_at = Some(2);
        let mut j = Joiner::new(encs, 4, 2).unwrap();
        assert!(run(&mut j, 40).unwrap_err().contains("internal error"));
    }

    #[test]
    fn broken_encoder_output_is_an_error() {
        // more frames than pictures
        let mut j = Joiner::new(vec![fake(0), Fake { extra_on_flush: true, ..fake(1) }], 4, 4).unwrap();
        assert!(run(&mut j, 12).unwrap_err().contains("more frames"));
        // a segment that does not start with a keyframe
        let mut j = Joiner::new(vec![fake(0), Fake { no_key: true, ..fake(1) }], 4, 4).unwrap();
        assert!(run(&mut j, 12).unwrap_err().contains("keyframe"));
    }

    #[test]
    fn hostile_calls_give_errors() {
        assert!(Joiner::<Fake>::new(Vec::new(), 4, 4).is_err());
        assert!(Joiner::new(vec![fake(0)], 1, 4).is_err());
        assert!(Joiner::new(vec![fake(0)], 0, 4).is_err());
        // out of order
        let mut j = Joiner::new(vec![fake(0), fake(1)], 4, 4).unwrap();
        assert!(j.push(1, &[0], &[], &[]).is_err());
        let mut j = Joiner::new(vec![fake(0), fake(1)], 4, 4).unwrap();
        j.push(0, &[0], &[], &[]).unwrap();
        assert!(j.push(0, &[0], &[], &[]).is_err());
        // after the end
        let mut j = Joiner::new(vec![fake(0), fake(1)], 4, 0).unwrap();
        run(&mut j, 9).unwrap();
        assert!(j.finish().unwrap().is_empty(), "finishing twice gives nothing more");
        assert!(j.push(9, &[0], &[], &[]).is_err());
        // a huge segment and budget
        let mut j = Joiner::new(vec![fake(0), fake(1)], u64::MAX, usize::MAX).unwrap();
        assert_eq!(run(&mut j, 10).unwrap().len(), 10);
    }

    #[test]
    fn the_budget_bounds_the_queued_pictures() {
        // a slow second lane: the producer waits instead of queueing everything
        struct Slow(Fake);
        impl LaneEncoder for Slow {
            type Params = u32;
            fn encode(&mut self, local: u64, planes: &[u8], keyframe: bool) -> Result<Vec<LanePacket>, String> {
                std::thread::sleep(Duration::from_millis(2));
                self.0.encode(local, planes, keyframe)
            }
            fn flush(&mut self) -> Result<Vec<LanePacket>, String> {
                self.0.flush()
            }
            fn params(&self) -> Option<u32> {
                self.0.params()
            }
        }
        let mut j = Joiner::new(vec![Slow(fake(0)), Slow(fake(1))], 8, 3).unwrap();
        let mut most = 0;
        for i in 0..64u64 {
            j.push(i, &[i as u8], &[], &[]).unwrap();
            most = most.max(*j.gate.queued.lock().unwrap());
        }
        assert!(most <= 3, "{most} pictures alive with a budget of 3");
        assert_eq!(j.finish().map(|_| ()), Ok(()));
    }

    #[test]
    fn dropping_mid_stream_ends_the_lanes() {
        let seen = Arc::new(AtomicUsize::new(0));
        let encs = (0..2).map(|k| Fake { seen: seen.clone(), delay: 2, ..fake(k) }).collect();
        let mut j = Joiner::new(encs, 4, 16).unwrap();
        for i in 0..10u64 {
            j.push(i, &[0], &[], &[]).unwrap();
        }
        let at_drop = seen.load(Ordering::SeqCst);
        drop(j);
        // 10 pictures, and the first segment's 4 again to warm the second lane up (all of a short
        // segment): the lanes leave what is still queued, and the threads are joined
        let after = seen.load(Ordering::SeqCst);
        assert!(after >= at_drop && after <= 14, "{at_drop} encoded at the drop, {after} after");
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(seen.load(Ordering::SeqCst), after, "nothing runs after the drop returned");
    }
}
