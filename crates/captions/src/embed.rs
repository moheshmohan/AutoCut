//! Closed captions embedded in H.264 / H.265 video: CEA-608 field-1 data (CC1) and CEA-708
//! (DTVCC) service 1 carried frame by frame as `cc_data()` in `user_data_registered_itu_t_t35`
//! SEI messages, the way ATSC A/53 Part 4, ATSC A/72 Part 1 and ANSI/SCTE 128-1 carry them (and
//! YouTube, broadcast playout and most players read them).
//!
//! ```text
//! SEI message (ITU-T H.264 §7.3.2.3 / D.1.6, H.265 §7.3.5 / D.2.6):
//!   payloadType 4 (user_data_registered_itu_t_t35), payloadSize (0xFF-extended), payload
//! payload (ATSC1_data, SCTE 128-1 §8.1):
//!   itu_t_t35_country_code   0xB5           (United States)
//!   itu_t_t35_provider_code  0x0031         (ATSC)
//!   user_identifier          'G' 'A' '9' '4'
//!   user_data_type_code      0x03           (cc_data)
//!   cc_data() (A/53 Part 4 §6.2.3.1):
//!     1 | process_cc_data_flag 1 | 0 | cc_count (5 bits)
//!     em_data                0xFF
//!     cc_count × [ 1 1 1 1 1 | cc_valid | cc_type (2 bits) ][ cc_data_1 ][ cc_data_2 ]
//!     marker_bits            0xFF
//! ```
//!
//! `cc_type` 0 is CEA-608 field 1 (odd parity bytes, CTA-608-E), 1 field 2, 2 DTVCC packet data
//! and 3 the start of a DTVCC packet (CTA-708-E §4.4); invalid triplets `FA 00 00` pad a frame to
//! its `cc_count`. The caption channel is 9600 bit/s, 600 triplets a second, so `cc_count` is
//! 600 / frame rate: 20 at 29.97 / 30 fps, 25 at 23.976 / 24, 24 at 25, 10 at 59.94 / 60
//! ([`cc_count`]).
//!
//! **Writing** ([`Schedule`]): the pop-on CEA-608 schedule of the [SCC writer](crate::scc) and the
//! CEA-708 two-window pop-on schedule of the [MCC writer](crate::mcc) (both on the 29.97 fps grid
//! of line 21) are moved onto the video's frames: each 29.97 fps slot goes to the nearest video
//! frame, the CEA-608 pairs first (so a caption's End of Caption lands on its first frame, at
//! most one 29.97 fps slot away at other rates), then DTVCC data in order, as much as the frame
//! holds; what does not fit moves to the next frame (DTVCC packets may span frames). Every frame
//! carries one SEI, padded, so decoders see a continuous caption channel. The SEI goes into each
//! access unit before its first slice ([`insert_sei`]); SEI NAL units are H.264 type 6 and H.265
//! prefix SEI type 39 ([`sei_nal`]), with emulation prevention.
//!
//! **Reading** ([`sample_cc`], [`decode`]): the cc_data triplets of every access unit, put in
//! display order by the caller, decode with the SCC decoder (CEA-608) or the MCC decoder's
//! CEA-708 service 1 reader.
//!
//! Caption text and timing are hostile: characters 608 / 708 cannot show are dropped or replaced
//! (control characters never reach the stream), rows wrap at 32 columns and stop at 15, cues with
//! no duration are skipped and overlapping cues are sent one after the other. Every parse is
//! bounds-checked.

use std::collections::{BTreeMap, VecDeque};

use filmcraft_time::FrameRate;

use crate::cea608::with_parity;
use crate::{Cue, Document, mcc, scc};

/// `itu_t_t35_country_code` of ATSC user data (United States).
pub const COUNTRY_US: u8 = 0xb5;
/// `itu_t_t35_provider_code` of ATSC.
pub const PROVIDER_ATSC: u16 = 0x0031;
/// `user_identifier` of ATSC1_data().
pub const GA94: [u8; 4] = *b"GA94";
/// `user_data_type_code` of cc_data().
pub const CC_DATA: u8 = 0x03;
/// SEI payloadType of `user_data_registered_itu_t_t35`.
pub const PAYLOAD_T35: u32 = 4;
/// An invalid DTVCC triplet: padding.
pub const PADDING: [u8; 3] = [0xfa, 0x00, 0x00];
/// The most triplets a cc_data() holds (`cc_count` has 5 bits).
pub const MAX_CC_COUNT: usize = 31;

/// The video codec of the stream the captions go into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Codec {
    H264,
    Hevc,
}

/// Which caption streams to embed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Streams {
    /// CEA-608 field 1, channel 1 (CC1).
    pub cea608: bool,
    /// CEA-708 service 1.
    pub cea708: bool,
}

impl Default for Streams {
    fn default() -> Self {
        Streams { cea608: true, cea708: true }
    }
}

/// Which stream [`decode`] reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stream {
    Cea608,
    Cea708,
}

/// cc_data triplets per frame at `rate`: 600 a second (A/53 Part 4 Table 6.4), at least 1 and at
/// most 31.
pub fn cc_count(rate: FrameRate) -> usize {
    let fps = rate.as_f64();
    if !fps.is_finite() || fps <= 0.0 {
        return 20;
    }
    ((600.0 / fps + 1e-6).floor() as usize).clamp(1, MAX_CC_COUNT)
}

/// The cc_data() of one frame (`triplets` past [`MAX_CC_COUNT`] are dropped).
pub fn cc_data(triplets: &[[u8; 3]]) -> Vec<u8> {
    let n = triplets.len().min(MAX_CC_COUNT);
    let mut out = Vec::with_capacity(3 + 3 * n);
    // reserved '1', process_cc_data_flag '1', zero_bit '0', cc_count
    out.push(0xc0 | n as u8);
    out.push(0xff); // em_data
    for t in &triplets[..n] {
        // marker bits '11111', then cc_valid and cc_type as given
        out.push(0xf8 | (t[0] & 0x07));
        out.push(t[1]);
        out.push(t[2]);
    }
    out.push(0xff); // marker_bits
    out
}

/// The `user_data_registered_itu_t_t35` payload (ATSC1_data with cc_data) of one frame.
pub fn t35_payload(triplets: &[[u8; 3]]) -> Vec<u8> {
    let mut out = vec![COUNTRY_US];
    out.extend_from_slice(&PROVIDER_ATSC.to_be_bytes());
    out.extend_from_slice(&GA94);
    out.push(CC_DATA);
    out.extend(cc_data(triplets));
    out
}

/// The triplets of a `user_data_registered_itu_t_t35` payload, if it is ATSC cc_data.
pub fn parse_t35(payload: &[u8]) -> Option<Vec<[u8; 3]>> {
    let rest = payload.strip_prefix(&[COUNTRY_US])?;
    let rest = rest.strip_prefix(&PROVIDER_ATSC.to_be_bytes())?;
    let rest = rest.strip_prefix(&GA94)?;
    let rest = rest.strip_prefix(&[CC_DATA])?;
    let head = *rest.first()?;
    if head & 0x40 == 0 {
        return Some(Vec::new()); // process_cc_data_flag off: nothing to process
    }
    let n = (head & 0x1f) as usize;
    let body = rest.get(2..2 + 3 * n)?;
    Some(body.as_chunks::<3>().0.to_vec())
}

/// An SEI RBSP holding one message (payloadType and size are 0xFF-extended).
pub fn sei_rbsp(payload_type: u32, payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 8);
    let ext = |mut v: usize, out: &mut Vec<u8>| {
        while v >= 255 {
            out.push(0xff);
            v -= 255;
        }
        out.push(v as u8);
    };
    ext(payload_type as usize, &mut out);
    ext(payload.len(), &mut out);
    out.extend_from_slice(payload);
    out.push(0x80); // rbsp_trailing_bits
    out
}

/// The messages `(payloadType, payload)` of an SEI RBSP (emulation prevention already removed).
/// Stops at the trailing bits or at the first message that does not fit.
pub fn parse_sei_rbsp(rbsp: &[u8]) -> Vec<(u32, &[u8])> {
    let mut out = Vec::new();
    let mut i = 0usize;
    // a message is at least 2 bytes, so the lone 0x80 trailing byte ends the list
    while i + 2 <= rbsp.len() {
        let read = |i: &mut usize| -> Option<usize> {
            let mut v = 0usize;
            loop {
                let b = *rbsp.get(*i)?;
                *i += 1;
                v = v.checked_add(b as usize)?;
                if b != 0xff {
                    return Some(v);
                }
            }
        };
        let (Some(t), Some(size)) = (read(&mut i), read(&mut i)) else { break };
        let Some(p) = i.checked_add(size).and_then(|e| rbsp.get(i..e)) else { break };
        out.push((t as u32, p));
        i += size;
    }
    out
}

/// A complete SEI NAL unit (header and escaped RBSP, no start code or length) with one
/// `user_data_registered_itu_t_t35` message.
pub fn sei_nal(payload: &[u8], codec: Codec) -> Vec<u8> {
    let rbsp = filmcraft_bitstream::escape_rbsp(&sei_rbsp(PAYLOAD_T35, payload));
    let mut out = match codec {
        // forbidden_zero_bit 0, nal_ref_idc 0, nal_unit_type 6
        Codec::H264 => vec![0x06],
        // forbidden_zero_bit 0, nal_unit_type 39 (PREFIX_SEI_NUT), nuh_layer_id 0, nuh_temporal_id_plus1 1
        Codec::Hevc => vec![39 << 1, 0x01],
    };
    out.extend(rbsp);
    out
}

fn nal_type(nal: &[u8], codec: Codec) -> Option<u8> {
    let h = *nal.first()?;
    Some(match codec {
        Codec::H264 => h & 0x1f,
        Codec::Hevc => (h >> 1) & 0x3f,
    })
}

fn is_vcl(t: u8, codec: Codec) -> bool {
    match codec {
        Codec::H264 => (1..=5).contains(&t),
        Codec::Hevc => t <= 31,
    }
}

fn is_sei(t: u8, codec: Codec) -> bool {
    match codec {
        Codec::H264 => t == 6,
        Codec::Hevc => t == 39 || t == 40,
    }
}

/// The NAL units of a sample with `length_size`-byte big-endian lengths, as (offset of the length
/// field, NAL unit); `None` if the lengths do not add up to the sample.
fn nal_units(sample: &[u8], length_size: usize) -> Option<Vec<(usize, &[u8])>> {
    if !(1..=4).contains(&length_size) {
        return None;
    }
    let mut out = Vec::new();
    let mut i = 0usize;
    while i < sample.len() {
        let len_bytes = sample.get(i..i.checked_add(length_size)?)?;
        let len = len_bytes.iter().fold(0usize, |a, &b| (a << 8) | b as usize);
        let start = i + length_size;
        let nal = sample.get(start..start.checked_add(len)?)?;
        out.push((i, nal));
        i = start + len;
    }
    Some(out)
}

/// `sample` (an access unit of `length_size`-prefixed NAL units, as MP4 / QuickTime store them)
/// with `sei` inserted before its first slice. `None` (leave the sample as it is) when the sample
/// does not parse, has no slice, or the SEI is too long for the length field.
pub fn insert_sei(sample: &[u8], sei: &[u8], codec: Codec, length_size: usize) -> Option<Vec<u8>> {
    let nals = nal_units(sample, length_size)?;
    let at = nals.iter().find(|(_, n)| nal_type(n, codec).is_some_and(|t| is_vcl(t, codec)))?.0;
    if (sei.len() as u64) >> (8 * length_size as u64) != 0 {
        return None;
    }
    let len = (sei.len() as u64).to_be_bytes();
    let mut out = Vec::with_capacity(sample.len() + sei.len() + length_size);
    out.extend_from_slice(sample.get(..at)?);
    out.extend_from_slice(len.get(8 - length_size..)?);
    out.extend_from_slice(sei);
    out.extend_from_slice(sample.get(at..)?);
    Some(out)
}

/// The cc_data triplets of every ATSC caption SEI in an access unit (empty when there is none or
/// the sample does not parse).
pub fn sample_cc(sample: &[u8], codec: Codec, length_size: usize) -> Vec<[u8; 3]> {
    let Some(nals) = nal_units(sample, length_size) else { return Vec::new() };
    let header = if codec == Codec::H264 { 1 } else { 2 };
    let mut out = Vec::new();
    for (_, nal) in nals {
        if !nal_type(nal, codec).is_some_and(|t| is_sei(t, codec)) {
            continue;
        }
        let rbsp = filmcraft_bitstream::unescape_rbsp(nal.get(header..).unwrap_or(&[]));
        for (t, payload) in parse_sei_rbsp(&rbsp) {
            if t == PAYLOAD_T35
                && let Some(tr) = parse_t35(payload)
            {
                out.extend(tr);
            }
        }
    }
    out
}

/// Decode the captions of frames `(display index, triplets)` (in display order) at `rate`. A
/// caption still showing at the last frame ends with the stream.
pub fn decode(frames: &[(i64, Vec<[u8; 3]>)], rate: FrameRate, stream: Stream) -> Vec<Cue> {
    let valid = |t: &[u8; 3]| t[0] & 0x04 != 0;
    let end = frames.iter().map(|f| f.0.saturating_add(1)).max();
    match stream {
        Stream::Cea608 => {
            let pairs = frames.iter().flat_map(|(f, ts)| ts.iter().filter(|t| valid(t) && t[0] & 0x03 == 0).map(move |t| (*f, t[1], t[2])));
            scc::decode_pairs_to(pairs, rate, end)
        }
        Stream::Cea708 => {
            let pairs = frames.iter().flat_map(|(f, ts)| ts.iter().filter(|t| valid(t) && t[0] & 0x02 != 0).map(move |t| (*f, t[0] & 0x03 == 3, t[1], t[2])));
            mcc::decode_708(pairs, rate, end)
        }
    }
}

/// The caption data of every frame of an export, built once from the caption document.
#[derive(Clone, Debug)]
pub struct Schedule {
    cc_count: usize,
    /// Frames with caption data: valid triplets (CEA-608 first), without padding.
    frames: BTreeMap<i64, Vec<[u8; 3]>>,
    /// Send a null CEA-608 pair on frames without CEA-608 data (rates up to 30 fps, where one
    /// pair a frame stays within the line 21 rate).
    null_608: bool,
}

impl Schedule {
    /// Schedule `doc` (cue times from the first frame of the video) on the frames of `rate`.
    pub fn new(doc: &Document, rate: FrameRate, streams: Streams) -> Schedule {
        let count = cc_count(rate);
        let line21 = scc::RATE;
        // nearest video frame of a 29.97 fps slot
        let frame_of = |f: i64| rate.frame_at(rate.snap_nearest(line21.tick_of(f)));
        let mut f1: VecDeque<(i64, [u8; 3])> = VecDeque::new();
        if streams.cea608 {
            for (f, (a, b)) in scc::schedule(doc) {
                f1.push_back((frame_of(f), [0xfc, with_parity(a), with_parity(b)]));
            }
        }
        let mut dtv: VecDeque<(i64, [u8; 3])> = VecDeque::new();
        if streams.cea708 {
            for (f, pairs) in mcc::schedule_708(doc) {
                let at = frame_of(f);
                for (start, a, b) in pairs {
                    dtv.push_back((at, [if start { 0xff } else { 0xfe }, a, b]));
                }
            }
        }
        let mut frames: BTreeMap<i64, Vec<[u8; 3]>> = BTreeMap::new();
        let mut k = match (f1.front(), dtv.front()) {
            (Some(a), Some(b)) => a.0.min(b.0),
            (Some(a), None) => a.0,
            (None, Some(b)) => b.0,
            (None, None) => 0,
        };
        // every pass of the loop fills frame `k` (taking at least one triplet when one is due)
        // or jumps to the next frame with data, so it ends
        while !f1.is_empty() || !dtv.is_empty() {
            let due = |q: &VecDeque<(i64, [u8; 3])>| q.front().is_some_and(|x| x.0 <= k);
            if !due(&f1) && !due(&dtv) {
                k = f1.front().map(|x| x.0).into_iter().chain(dtv.front().map(|x| x.0)).min().unwrap_or(k).max(k + 1);
                continue;
            }
            let mut v: Vec<[u8; 3]> = Vec::new();
            while v.len() < count && due(&f1) {
                if let Some((_, t)) = f1.pop_front() {
                    v.push(t);
                }
            }
            while v.len() < count && due(&dtv) {
                if let Some((_, t)) = dtv.pop_front() {
                    v.push(t);
                }
            }
            frames.insert(k, v);
            k = k.saturating_add(1);
        }
        Schedule { cc_count: count, frames, null_608: streams.cea608 && rate.as_f64() <= 30.0 + 1e-6 }
    }

    /// Triplets per frame.
    pub fn cc_count(&self) -> usize {
        self.cc_count
    }

    /// The last frame with caption data.
    pub fn last_frame(&self) -> Option<i64> {
        self.frames.keys().next_back().copied()
    }

    /// Whether any frame carries caption data.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// The `cc_count` triplets of frame `frame`: its caption data, padded.
    pub fn triplets(&self, frame: i64) -> Vec<[u8; 3]> {
        let mut v = self.frames.get(&frame).cloned().unwrap_or_default();
        if self.null_608 && !v.iter().any(|t| t[0] & 0x03 == 0) {
            v.insert(0, [0xfc, 0x80, 0x80]);
        }
        v.truncate(self.cc_count);
        v.resize(self.cc_count, PADDING);
        v
    }

    /// The SEI NAL unit of frame `frame`.
    pub fn sei_nal(&self, frame: i64, codec: Codec) -> Vec<u8> {
        sei_nal(&t35_payload(&self.triplets(frame)), codec)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use filmcraft_time::Tick;

    fn cue(rate: FrameRate, s: i64, e: i64, text: &str) -> Cue {
        Cue { start: rate.tick_of(s), end: rate.tick_of(e), text: text.into(), ..Default::default() }
    }

    /// Frames of a schedule as a decoder sees them.
    fn frames(s: &Schedule, n: i64) -> Vec<(i64, Vec<[u8; 3]>)> {
        (0..n).map(|f| (f, s.triplets(f))).collect()
    }

    fn timed(rate: FrameRate, cues: &[Cue]) -> Vec<(i64, i64, String)> {
        cues.iter().map(|c| (rate.frame_at(c.start), rate.frame_at(c.end), c.text.clone())).collect()
    }

    #[test]
    fn cc_count_follows_the_frame_rate() {
        assert_eq!(cc_count(FrameRate::FPS_29_97), 20);
        assert_eq!(cc_count(FrameRate::FPS_30), 20);
        assert_eq!(cc_count(FrameRate::FPS_23_976), 25);
        assert_eq!(cc_count(FrameRate::FPS_24), 25);
        assert_eq!(cc_count(FrameRate::FPS_25), 24);
        assert_eq!(cc_count(FrameRate::FPS_50), 12);
        assert_eq!(cc_count(FrameRate::FPS_59_94), 10);
        assert_eq!(cc_count(FrameRate::FPS_60), 10);
        assert_eq!(cc_count(FrameRate::new(1000, 1)), 1);
        assert_eq!(cc_count(FrameRate::new(1, 1)), 31);
    }

    /// The bytes of the SEI message match ATSC A/53 Part 4 / SCTE 128-1 field by field.
    #[test]
    fn sei_bytes_follow_the_standard() {
        let t = [[0xfc, 0x94, 0x20], [0xff, 0x02, 0x21], PADDING];
        let p = t35_payload(&t);
        assert_eq!(&p[..8], &[0xb5, 0x00, 0x31, b'G', b'A', b'9', b'4', 0x03], "country, provider, GA94, cc_data type");
        assert_eq!(p[8], 0xc0 | 3, "reserved 1, process_cc_data_flag 1, zero bit, cc_count 3");
        assert_eq!(p[9], 0xff, "em_data");
        assert_eq!(&p[10..19], &[0xfc, 0x94, 0x20, 0xff, 0x02, 0x21, 0xfa, 0x00, 0x00]);
        assert_eq!(p[19], 0xff, "marker_bits");
        assert_eq!(p.len(), 20);
        for c in p[10..19].chunks(3) {
            assert_eq!(c[0] & 0xf8, 0xf8, "marker bits 11111");
        }
        let rbsp = sei_rbsp(PAYLOAD_T35, &p);
        assert_eq!(&rbsp[..2], &[4, 20], "payloadType 4, payloadSize 20");
        assert_eq!(*rbsp.last().unwrap(), 0x80, "rbsp_trailing_bits");
        let h264 = sei_nal(&p, Codec::H264);
        assert_eq!(h264[0], 0x06, "nal_ref_idc 0, nal_unit_type 6");
        let hevc = sei_nal(&p, Codec::Hevc);
        assert_eq!(&hevc[..2], &[0x4e, 0x01], "PREFIX_SEI_NUT, layer 0, temporal id 1");
        assert_eq!(parse_t35(&p).unwrap(), t.to_vec());
        assert_eq!(parse_sei_rbsp(&rbsp), vec![(4, p.as_slice())]);
        // long payloads extend the size with 0xFF bytes
        let long = vec![0u8; 600];
        let r = sei_rbsp(5, &long);
        assert_eq!(&r[..4], &[5, 255, 255, 90]);
        assert_eq!(parse_sei_rbsp(&r)[0].1.len(), 600);
    }

    #[test]
    fn emulation_prevention_and_splicing() {
        // cc bytes that would form start codes are escaped in the NAL and come back unescaped
        let t = [[0xfc, 0x00, 0x00], [0xfe, 0x00, 0x01], [0xfe, 0x00, 0x00]];
        let nal = sei_nal(&t35_payload(&t), Codec::H264);
        assert!(!nal.windows(3).any(|w| w == [0, 0, 1] || w == [0, 0, 0]), "no start code inside the NAL");
        // an access unit: AUD, slice
        let aud = [0x09u8, 0xf0];
        let slice = [0x65u8, 0x88, 0x84, 0x00];
        let mut au = Vec::new();
        for n in [&aud[..], &slice[..]] {
            au.extend_from_slice(&(n.len() as u32).to_be_bytes());
            au.extend_from_slice(n);
        }
        let with = insert_sei(&au, &nal, Codec::H264, 4).unwrap();
        let units = nal_units(&with, 4).unwrap();
        let types: Vec<u8> = units.iter().map(|(_, n)| n[0] & 0x1f).collect();
        assert_eq!(types, vec![9, 6, 5], "the SEI goes before the first slice");
        assert_eq!(sample_cc(&with, Codec::H264, 4), t.to_vec());
        // HEVC: VPS-free sample with an IDR_W_RADL slice (type 19)
        let hnal = sei_nal(&t35_payload(&t), Codec::Hevc);
        let hslice = [19u8 << 1, 0x01, 0xaf, 0x00];
        let mut hau = (hslice.len() as u32).to_be_bytes().to_vec();
        hau.extend_from_slice(&hslice);
        let hwith = insert_sei(&hau, &hnal, Codec::Hevc, 4).unwrap();
        assert_eq!(sample_cc(&hwith, Codec::Hevc, 4), t.to_vec());
        let ht: Vec<u8> = nal_units(&hwith, 4).unwrap().iter().map(|(_, n)| (n[0] >> 1) & 0x3f).collect();
        assert_eq!(ht, vec![39, 19]);
        // no slice, or lengths that do not add up: left alone
        assert_eq!(insert_sei(&au[..6], &nal, Codec::H264, 4), None);
        assert_eq!(insert_sei(&au[..au.len() - 1], &nal, Codec::H264, 4), None);
        assert_eq!(insert_sei(&au, &nal, Codec::H264, 0), None);
    }

    fn doc(rate: FrameRate) -> Document {
        Document {
            cues: vec![cue(rate, 60, 120, "Hello, world!"), cue(rate, 120, 200, "Second caption\nwith two lines"), cue(rate, 260, 300, "Café ♪ “q”")],
            ..Default::default()
        }
    }

    /// Both streams of the schedule decode back to the captions, frame-exact at 29.97 fps and
    /// within a frame at other rates.
    #[test]
    fn schedule_round_trips_at_every_rate() {
        for rate in
            [FrameRate::FPS_29_97, FrameRate::FPS_30, FrameRate::FPS_24, FrameRate::FPS_23_976, FrameRate::FPS_25, FrameRate::FPS_59_94, FrameRate::FPS_60]
        {
            let d = doc(rate);
            let s = Schedule::new(&d, rate, Streams::default());
            let n = s.last_frame().unwrap() + 2;
            let fr = frames(&s, n);
            assert!(fr.iter().all(|(_, t)| t.len() == cc_count(rate)), "{rate:?}: every frame is padded to cc_count");
            let want = timed(rate, &d.cues);
            for stream in [Stream::Cea608, Stream::Cea708] {
                let got = timed(rate, &decode(&fr, rate, stream));
                assert_eq!(got.len(), want.len(), "{rate:?} {stream:?}: {got:?}");
                let tol = if rate == FrameRate::FPS_29_97 { 0 } else { 1 };
                for (g, w) in got.iter().zip(&want) {
                    assert_eq!(g.2, w.2, "{rate:?} {stream:?}");
                    assert!((g.0 - w.0).abs() <= tol && (g.1 - w.1).abs() <= tol, "{rate:?} {stream:?}: {g:?} vs {w:?}");
                }
            }
        }
    }

    #[test]
    fn one_stream_only() {
        let rate = FrameRate::FPS_29_97;
        let d = doc(rate);
        let only608 = Schedule::new(&d, rate, Streams { cea608: true, cea708: false });
        let fr = frames(&only608, only608.last_frame().unwrap() + 2);
        assert!(fr.iter().flat_map(|(_, t)| t).all(|t| t[0] & 0x02 == 0 || t[0] & 0x04 == 0), "no DTVCC data");
        assert_eq!(decode(&fr, rate, Stream::Cea608).len(), 3);
        assert!(decode(&fr, rate, Stream::Cea708).is_empty());
        let only708 = Schedule::new(&d, rate, Streams { cea608: false, cea708: true });
        let fr = frames(&only708, only708.last_frame().unwrap() + 2);
        assert!(decode(&fr, rate, Stream::Cea608).is_empty());
        assert_eq!(decode(&fr, rate, Stream::Cea708).len(), 3);
        assert!(Schedule::new(&Document::default(), rate, Streams::default()).is_empty());
    }

    /// Control codes, very long lines, characters 608 cannot show, zero-length, reversed,
    /// overlapping and far-away cues schedule and decode without a panic, and nothing but the
    /// caption's printable text reaches the decoder.
    #[test]
    fn hostile_captions_never_panic() {
        let rate = FrameRate::FPS_29_97;
        let long: String = "word ".repeat(400);
        let ctl: String = (0u8..32).map(char::from).chain("\u{7f}\u{85}\u{9b}\u{200b}\u{202e}漢字🎬".chars()).collect();
        let d = Document {
            cues: vec![
                cue(rate, 0, 1, &ctl),
                cue(rate, 5, 5, "zero"),
                cue(rate, 9, 3, "reversed"),
                cue(rate, 10, 400, &long),
                cue(rate, 20, 30, "overlap"),
                cue(rate, 20, 30, "\u{14}\u{2c}\u{94}\u{2c}"),
                cue(rate, 1_000_000, 1_000_010, "far"),
                Cue { start: Tick(i64::MAX - 10), end: Tick(i64::MAX), text: "end of time".into(), ..Default::default() },
                Cue { start: Tick(i64::MIN), end: Tick(i64::MIN + 5), text: "before time".into(), ..Default::default() },
            ],
            ..Default::default()
        };
        for r in [rate, FrameRate::FPS_24, FrameRate::FPS_60, FrameRate::new(1000, 1), FrameRate::new(1, 1)] {
            let s = Schedule::new(&d, r, Streams::default());
            // decode the first stretch (the far cues are hours away)
            let fr = frames(&s, 3000);
            for stream in [Stream::Cea608, Stream::Cea708] {
                for c in decode(&fr, r, stream) {
                    assert!(!c.text.chars().any(|ch| ch.is_control() && ch != '\n'), "{r:?} {stream:?}: {:?}", c.text);
                }
            }
            let _ = s.sei_nal(0, Codec::H264);
            let _ = s.sei_nal(i64::MAX, Codec::Hevc);
        }
    }

    /// Mutation fuzz of the parsers: truncated and bit-flipped SEI NAL units and samples.
    #[test]
    fn parsers_survive_corrupt_input() {
        let rate = FrameRate::FPS_29_97;
        let s = Schedule::new(&doc(rate), rate, Streams::default());
        let nal = s.sei_nal(62, Codec::H264);
        let mut au = (nal.len() as u32).to_be_bytes().to_vec();
        au.extend_from_slice(&nal);
        au.extend_from_slice(&[0, 0, 0, 2, 0x65, 0x88]);
        let mut x = 0x1234_5678u32;
        for i in 0..2000 {
            let mut m = au.clone();
            x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let at = (x as usize >> 8) % m.len();
            m[at] ^= 1 << (x % 8);
            if i % 3 == 0 {
                m.truncate((x as usize >> 4) % (m.len() + 1));
            }
            for codec in [Codec::H264, Codec::Hevc] {
                for ls in [1, 2, 3, 4, 5] {
                    let _ = sample_cc(&m, codec, ls);
                    let _ = insert_sei(&m, &nal, codec, ls);
                }
            }
            let _ = parse_sei_rbsp(&m);
            let _ = parse_t35(&m);
        }
        assert!(parse_t35(&[0xb5, 0, 0x31, b'G', b'A', b'9', b'4', 3, 0xdf, 0xff, 1]).is_none(), "cc_count past the payload");
        assert_eq!(parse_sei_rbsp(&[0xff, 0xff, 0xff]), vec![]);
    }
}
