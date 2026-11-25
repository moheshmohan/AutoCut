//! A 14-byte sequence header must not make the decoder size its frame buffers at 16383 x 16383
//! (gigabytes per picture). Hand-built, no ffmpeg.

use filmcraft_mpeg2v::Decoder;

#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    n: usize,
}

impl Bits {
    fn put(&mut self, value: u32, bits: usize) {
        for i in (0..bits).rev() {
            if self.n.is_multiple_of(8) {
                self.bytes.push(0);
            }
            if (value >> i) & 1 == 1
                && let Some(b) = self.bytes.last_mut()
            {
                *b |= 0x80 >> (self.n % 8);
            }
            self.n += 1;
        }
    }
}

/// Sequence header plus sequence extension for a `w` x `h` 4:4:4 stream.
fn sequence(w: u32, h: u32, chroma_format: u32) -> Vec<u8> {
    let mut s = Bits::default();
    s.put(w & 0xfff, 12);
    s.put(h & 0xfff, 12);
    s.put(1, 4); // aspect_ratio_information
    s.put(4, 4); // frame_rate_code
    s.put(1000, 18); // bit_rate_value
    s.put(1, 1); // marker
    s.put(100, 10); // vbv_buffer_size_value
    s.put(0, 3); // constrained_parameters_flag, load_intra, load_non_intra
    let mut e = Bits::default();
    e.put(1, 4); // sequence_extension
    e.put(0x48, 8); // profile_and_level_indication
    e.put(1, 1); // progressive_sequence
    e.put(chroma_format, 2);
    e.put(w >> 12, 2);
    e.put(h >> 12, 2);
    e.put(0, 12); // bit_rate_extension
    e.put(1, 1); // marker
    e.put(0, 8 + 1 + 2 + 5); // vbv_buffer_size_extension, low_delay, frame_rate_extension_n/d
    let mut out = vec![0, 0, 1, 0xb3];
    out.extend(s.bytes);
    out.extend([0, 0, 1, 0xb5]);
    out.extend(e.bytes);
    out
}

fn accepted(w: u32, h: u32, chroma_format: u32) -> bool {
    let mut dec = Decoder::new();
    let _ = dec.decode(&sequence(w, h, chroma_format), 0);
    dec.sequence().is_some()
}

#[test]
fn ordinary_sequences_are_accepted() {
    assert!(accepted(1920, 1080, 1));
    assert!(accepted(7680, 4320, 3));
}

#[test]
fn gigantic_sequences_are_refused() {
    assert!(!accepted(16383, 16383, 3));
    assert!(!accepted(16383, 16383, 1));
    assert!(!accepted(8192, 8192, 1));
}

/// A picture of `coding_type` (1 = I, 2 = P): picture header, coding extension, one slice with a junk payload.
fn picture_of(coding_type: u32) -> Vec<u8> {
    let mut h = Bits::default();
    h.put(0, 10); // temporal_reference
    h.put(coding_type, 3); // picture_coding_type
    h.put(0xffff, 16); // vbv_delay
    if coding_type == 2 {
        h.put(0, 1); // full_pel_forward_vector
        h.put(7, 3); // forward_f_code
    }
    let mut e = Bits::default();
    e.put(8, 4); // picture_coding_extension
    e.put(0xffff, 16); // f_codes
    e.put(0, 2); // intra_dc_precision
    e.put(3, 2); // picture_structure: frame
    e.put(0, 10); // flags up to progressive_frame
    e.put(1, 1); // progressive_frame
    e.put(0, 1); // composite_display_flag
    let mut out = vec![0, 0, 1, 0x00];
    out.extend(h.bytes);
    out.extend([0, 0, 1, 0xb5]);
    out.extend(e.bytes);
    out.extend([0, 0, 1, 0x01, 0x80, 0x80]);
    out
}

fn picture() -> Vec<u8> {
    picture_of(1)
}

#[test]
fn an_access_unit_of_many_pictures_stops_early() {
    let mut data = sequence(64, 64, 1);
    for _ in 0..200 {
        data.extend(picture());
    }
    let mut dec = Decoder::new();
    assert!(dec.decode(&data, 0).is_err(), "200 pictures in one access unit must be refused");
    // pictures after the refusal are not decoded: the per-picture slice errors stay near the output cap
    assert!(dec.errors() <= 32, "{} picture errors", dec.errors());
}

#[test]
fn a_dimension_change_does_not_hide_an_overflow() {
    let mut data = sequence(64, 64, 1);
    for _ in 0..20 {
        data.extend(picture());
    }
    data.extend(sequence(128, 64, 1));
    data.extend(picture());
    let mut dec = Decoder::new();
    assert!(dec.decode(&data, 0).is_err(), "the reset for the new dimensions must not clear the overflow");
}

#[test]
fn a_refused_access_unit_leaves_no_stale_references() {
    let mut data = sequence(64, 64, 1);
    for _ in 0..200 {
        data.extend(picture());
    }
    let mut dec = Decoder::new();
    assert!(dec.decode(&data, 0).is_err());
    // a P picture cannot continue from the refused access unit's anchor
    assert!(dec.decode(&picture_of(2), 1).unwrap().is_empty());
    assert!(dec.flush().is_empty(), "a P picture after the refusal was decoded from stale references");
    // an I picture restarts decoding; nothing from the refused access unit comes out before it
    let two = [picture(), picture()].concat();
    assert_eq!(dec.decode(&two, 2).unwrap().len(), 1, "the first I picture is shown when the second arrives");
    assert_eq!(dec.flush().len(), 1);
}

/// `data` without its extension units (an MPEG-1 stream): everything from a 00 00 01 b5 start code up to the next start code.
fn mpeg1(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < data.len() {
        if data[i..].starts_with(&[0, 0, 1, 0xb5]) {
            i += 4;
            while i < data.len() && !data[i..].starts_with(&[0, 0, 1]) {
                i += 1;
            }
        } else {
            out.push(data[i]);
            i += 1;
        }
    }
    out
}

#[test]
fn an_mpeg1_d_picture_restarts_decoding_after_a_refused_access_unit() {
    let mut data = mpeg1(&sequence(64, 64, 1));
    for _ in 0..200 {
        data.extend(mpeg1(&picture()));
    }
    let mut dec = Decoder::new();
    assert!(dec.decode(&data, 0).is_err());
    let two = [mpeg1(&picture_of(4)), mpeg1(&picture_of(4))].concat();
    assert_eq!(dec.decode(&two, 1).unwrap().len(), 1, "the first D picture is shown when the second arrives");
}

/// 352x288 4:2:0 pads to 352x288: 152064 bytes a picture, 13 pictures held.
#[test]
fn sequence_over_the_memory_budget_is_refused() {
    let need = 13 * (352 * 288 * 3 / 2);
    for (budget, ok) in [(need, true), (need - 1, false)] {
        let mut dec = Decoder::new();
        dec.set_memory_budget(budget);
        assert_eq!(dec.memory_budget(), budget);
        let _ = dec.decode(&sequence(352, 288, 1), 0);
        assert_eq!(dec.sequence().is_some(), ok, "budget {budget}");
        assert_eq!(dec.errors(), usize::from(!ok));
        assert_eq!(dec.last_error().is_some(), !ok);
    }
    // 8K 4:4:4 (13 pictures of about 100 MB) fits the default budget.
    assert_eq!(Decoder::new().memory_budget(), filmcraft_mpeg2v::DECODE_MEMORY_BUDGET);
    assert!(accepted(7680, 4320, 3));
}

/// A refused sequence allocates and decodes nothing: the pictures after it are errors, and the
/// stream decodes again from the next sequence that fits.
#[test]
fn pictures_of_a_refused_sequence_are_not_decoded() {
    let need = 13 * (64 * 64 * 3 / 2);
    let mut dec = Decoder::new();
    dec.set_memory_budget(need - 1);
    let mut data = sequence(64, 64, 1);
    data.extend(picture());
    data.extend(picture());
    assert!(dec.decode(&data, 0).is_ok_and(|p| p.is_empty()));
    assert!(dec.flush().is_empty());
    assert!(dec.sequence().is_none());
    assert_eq!(dec.last_error(), Some("sequence exceeds the decoder memory budget"));
    let mut fits = sequence(48, 48, 1);
    fits.extend(picture());
    fits.extend(picture());
    assert_eq!(dec.decode(&fits, 1).map(|p| p.len()).ok(), Some(1));
    assert_eq!(dec.sequence().map(|s| (s.width, s.height)), Some((48, 48)));
}

/// Past 8K the caller raises the area limit; the memory budget still bounds what it lets in.
#[test]
fn a_raised_area_limit_is_still_bounded_by_the_budget() {
    let raised = |w: u32, h: u32, chroma: u32, area: u64, budget: Option<u64>| {
        let mut dec = Decoder::new();
        dec.set_max_picture_luma_samples(area);
        assert_eq!(dec.max_picture_luma_samples(), area);
        if let Some(b) = budget {
            dec.set_memory_budget(b);
        }
        let _ = dec.decode(&sequence(w, h, chroma), 0);
        dec.sequence().map(|s| (s.width, s.height))
    };
    assert_eq!(Decoder::new().max_picture_luma_samples(), filmcraft_mpeg2v::MAX_PICTURE_AREA);
    // 8000 x 8000 is 64 million luma samples, over the default 2^25
    assert!(!accepted(8000, 8000, 1));
    assert_eq!(raised(8000, 8000, 1, 1 << 26, None), Some((8000, 8000)));
    assert_eq!(raised(8000, 8000, 1, 8000 * 8000, None), Some((8000, 8000)));
    assert_eq!(raised(8000, 8000, 1, 8000 * 8000 - 1, None), None);
    // 13 pictures of 8000 x 8000 4:2:0 are 1.25 GB
    let need = 13 * 8000 * 8000 * 3 / 2;
    assert_eq!(raised(8000, 8000, 1, 1 << 26, Some(need)), Some((8000, 8000)));
    assert_eq!(raised(8000, 8000, 1, 1 << 26, Some(need - 1)), None);
    // the largest the header can code, 4:4:4, is 10 GB of pictures: over the default budget
    assert_eq!(raised(16383, 16383, 3, u64::MAX, None), None);
    // the probe follows the same limit
    let big = sequence(8000, 8000, 1);
    assert!(filmcraft_mpeg2v::probe(&big).is_none());
    assert_eq!(filmcraft_mpeg2v::probe_with_limit(&big, 1 << 26).map(|s| (s.width, s.height)), Some((8000, 8000)));
}
