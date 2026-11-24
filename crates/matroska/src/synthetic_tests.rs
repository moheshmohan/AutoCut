//! Hand-built files exercising features FFmpeg does not write: unknown-size Segment/Clusters,
//! header stripping, BlockGroups, Colour + mastering metadata, Chapters, Tags, Attachments, CRC-32.

use crate::codec::Codec;
use crate::demux::MAX_TRACKS;
use crate::ebml::{self, write_id, write_unknown_size};
use crate::ids::*;
use crate::mux::{el, el_float, el_str, el_uint};
use crate::track::ContentEncoding;
use crate::{ByteSource, Demuxer, Error, OpenOptions, Packet, open_with};

fn block(track: u8, rel: i16, flags: u8, data: &[u8]) -> Vec<u8> {
    let mut b = vec![0x80 | track];
    b.extend_from_slice(&rel.to_be_bytes());
    b.push(flags);
    b.extend_from_slice(data);
    b
}

fn synthetic() -> Vec<u8> {
    let mut f = Vec::new();
    let mut h = Vec::new();
    el_str(&mut h, DOC_TYPE, "matroska");
    el_uint(&mut h, DOC_TYPE_VERSION, 4);
    el(&mut f, EBML, &h);
    write_id(&mut f, SEGMENT);
    write_unknown_size(&mut f, 8);
    // Info with a correct CRC-32 and a Void
    let mut info = Vec::new();
    el_uint(&mut info, TIMESTAMP_SCALE, 1_000_000);
    el_str(&mut info, TITLE, "Synthetic");
    el(&mut info, VOID, &[0; 3]);
    let mut crc_info = Vec::new();
    el(&mut crc_info, CRC32, &ebml::crc32(&info).to_le_bytes());
    crc_info.extend_from_slice(&info);
    el(&mut f, INFO, &crc_info);
    // Tracks
    let mut v = Vec::new();
    el_uint(&mut v, TRACK_NUMBER, 1);
    el_uint(&mut v, TRACK_UID, 11);
    el_uint(&mut v, TRACK_TYPE, 1);
    el_str(&mut v, CODEC_ID, "V_MPEG4/ISO/AVC");
    el(&mut v, CODEC_PRIVATE, &[1, 0x64, 0, 0x1F]);
    el_uint(&mut v, DEFAULT_DURATION, 40_000_000);
    let mut mm = Vec::new();
    for (id, val) in [(PRIMARY_R_X, 0.708), (PRIMARY_R_Y, 0.292), (PRIMARY_G_X, 0.17), (PRIMARY_G_Y, 0.797), (PRIMARY_B_X, 0.131), (PRIMARY_B_Y, 0.046)] {
        el_float(&mut mm, id, val);
    }
    el_float(&mut mm, WHITE_POINT_X, 0.3127);
    el_float(&mut mm, WHITE_POINT_Y, 0.329);
    el_float(&mut mm, LUMINANCE_MAX, 1000.0);
    el_float(&mut mm, LUMINANCE_MIN, 0.0001);
    let mut col = Vec::new();
    el_uint(&mut col, MATRIX_COEFFICIENTS, 9);
    el_uint(&mut col, RANGE, 2);
    el_uint(&mut col, TRANSFER_CHARACTERISTICS, 16);
    el_uint(&mut col, PRIMARIES, 9);
    el_uint(&mut col, MAX_CLL, 1000);
    el_uint(&mut col, MAX_FALL, 400);
    el(&mut col, MASTERING_METADATA, &mm);
    let mut vid = Vec::new();
    el_uint(&mut vid, PIXEL_WIDTH, 1440);
    el_uint(&mut vid, PIXEL_HEIGHT, 1080);
    el_uint(&mut vid, DISPLAY_WIDTH, 1920);
    el_uint(&mut vid, DISPLAY_HEIGHT, 1080);
    el(&mut vid, COLOUR, &col);
    // rectangular, a quarter turn clockwise (portrait phone video)
    let mut proj = Vec::new();
    el_uint(&mut proj, PROJECTION_TYPE, 0);
    el_float(&mut proj, PROJECTION_POSE_ROLL, -90.0);
    el(&mut vid, PROJECTION, &proj);
    el(&mut v, VIDEO, &vid);
    // header stripping: every frame starts with 00 00 00 01
    let mut comp = Vec::new();
    el_uint(&mut comp, CONTENT_COMP_ALGO, 3);
    el(&mut comp, CONTENT_COMP_SETTINGS, &[0, 0, 0, 1]);
    let mut enc = Vec::new();
    el(&mut enc, CONTENT_COMPRESSION, &comp);
    let mut encs = Vec::new();
    el(&mut encs, CONTENT_ENCODING, &enc);
    el(&mut v, CONTENT_ENCODINGS, &encs);
    let mut a = Vec::new();
    el_uint(&mut a, TRACK_NUMBER, 2);
    el_uint(&mut a, TRACK_TYPE, 2);
    el_str(&mut a, CODEC_ID, "A_AAC/MPEG4/LC/SBR");
    el_str(&mut a, LANGUAGE, "ger");
    let mut au = Vec::new();
    el_float(&mut au, SAMPLING_FREQUENCY, 24000.0);
    el_float(&mut au, OUTPUT_SAMPLING_FREQUENCY, 48000.0);
    el_uint(&mut au, CHANNELS, 2);
    el(&mut a, AUDIO, &au);
    let mut tr = Vec::new();
    el(&mut tr, TRACK_ENTRY, &v);
    el(&mut tr, TRACK_ENTRY, &a);
    el(&mut f, TRACKS, &tr);
    // Chapters
    let mut disp = Vec::new();
    el_str(&mut disp, CHAP_STRING, "Intro");
    let mut atom = Vec::new();
    el_uint(&mut atom, CHAPTER_UID, 5);
    el_uint(&mut atom, CHAPTER_TIME_START, 0);
    el_uint(&mut atom, CHAPTER_TIME_END, 40_000_000);
    el(&mut atom, CHAPTER_DISPLAY, &disp);
    let mut ed = Vec::new();
    el(&mut ed, CHAPTER_ATOM, &atom);
    let mut ch = Vec::new();
    el(&mut ch, EDITION_ENTRY, &ed);
    el(&mut f, CHAPTERS, &ch);
    // Attachments
    let mut af = Vec::new();
    el_str(&mut af, FILE_NAME, "font.ttf");
    el_str(&mut af, FILE_MEDIA_TYPE, "font/ttf");
    el_uint(&mut af, FILE_UID, 77);
    el(&mut af, FILE_DATA, &[9; 100]);
    let mut att = Vec::new();
    el(&mut att, ATTACHED_FILE, &af);
    el(&mut f, ATTACHMENTS, &att);
    // Cluster 1 (unknown size): keyframe SimpleBlock, BlockGroup with ReferenceBlock + duration
    write_id(&mut f, CLUSTER);
    write_unknown_size(&mut f, 1);
    el_uint(&mut f, TIMESTAMP, 1000);
    el(&mut f, SIMPLE_BLOCK, &block(1, 0, 0x80, &[0x65, 1, 2]));
    el(&mut f, SIMPLE_BLOCK, &block(2, 5, 0x80, &[0xAA; 10]));
    let mut g = Vec::new();
    el(&mut g, BLOCK, &block(1, 40, 0, &[0x41, 3]));
    el(&mut g, REFERENCE_BLOCK, &[0xD8]);
    el_uint(&mut g, BLOCK_DURATION, 80);
    el(&mut f, BLOCK_GROUP, &g);
    el(&mut f, SIMPLE_BLOCK, &block(9, 0, 0x80, &[1])); // unknown track: ignored
    // Cluster 2 (unknown size), ended by Tags
    write_id(&mut f, CLUSTER);
    write_unknown_size(&mut f, 8);
    el_uint(&mut f, TIMESTAMP, 1120);
    el(&mut f, SIMPLE_BLOCK, &block(1, 0, 0x80, &[0x65, 4]));
    let mut st = Vec::new();
    el_str(&mut st, TAG_NAME, "ARTIST");
    el_str(&mut st, TAG_STRING, "Nobody");
    let mut tg = Vec::new();
    let mut tgt = Vec::new();
    el_uint(&mut tgt, TAG_TRACK_UID, 11);
    el(&mut tg, TARGETS, &tgt);
    el(&mut tg, SIMPLE_TAG, &st);
    let mut tags = Vec::new();
    el(&mut tags, TAG, &tg);
    el(&mut f, TAGS, &tags);
    f
}

#[test]
fn synthetic_file() {
    let bytes = synthetic();
    let f = open_with(&bytes[..], &OpenOptions { index: true, verify_crc: true }).unwrap();
    assert!(f.live);
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
    assert_eq!((f.crc_checked, f.crc_errors.len()), (1, 0));
    assert_eq!(f.info.title.as_deref(), Some("Synthetic"));
    let v = &f.tracks[0];
    assert!(matches!(&v.codec, Codec::Avc { avcc } if avcc[1] == 0x64));
    let vi = v.video.as_ref().unwrap();
    assert_eq!(vi.pixel_aspect(), (4, 3));
    assert_eq!(vi.projection.as_ref().map(|p| p.roll), Some(-90.0));
    assert_eq!(vi.display_rotation(), Some(1));
    let c = vi.colour.as_ref().unwrap();
    assert!(c.full_range());
    assert_eq!((c.matrix_coefficients, c.transfer_characteristics, c.primaries, c.max_cll, c.max_fall), (Some(9), Some(16), Some(9), Some(1000), Some(400)));
    let m = c.mastering.as_ref().unwrap();
    assert_eq!(m.primaries[1], (0.17, 0.797));
    assert_eq!(m.white_point, (0.3127, 0.329));
    assert_eq!((m.luminance_max, m.luminance_min), (1000.0, 0.0001));
    assert_eq!(v.encodings, vec![ContentEncoding::HeaderStripping { bytes: vec![0, 0, 0, 1], scope: 1 }]);
    // legacy AAC ID: ASC synthesised with explicit SBR signalling (LC, 24 kHz, stereo → 48 kHz)
    assert_eq!(f.tracks[1].codec, Codec::Aac { asc: vec![0x13, 0x10, 0x56, 0xE5, 0x98] });
    assert_eq!(f.tracks[1].language, "ger");
    assert_eq!(f.chapters[0].chapters[0].title(), Some("Intro"));
    assert_eq!(f.chapters[0].chapters[0].end_ns, Some(40_000_000));
    assert_eq!(f.attachments.len(), 1);
    let at = &f.attachments[0];
    assert_eq!((at.name.as_str(), at.media_type.as_str(), at.uid), ("font.ttf", "font/ttf", 77));
    assert_eq!(&bytes[at.data_offset as usize..][..100], &[9; 100]);
    assert_eq!(f.tags[0].targets.track_uids, vec![11]);
    assert_eq!(f.tags[0].simple[0].string.as_deref(), Some("Nobody"));
    assert_eq!(f.clusters.len(), 2);
    assert_eq!(v.samples.len(), 3);
    assert_eq!(v.keyframes(), &[0, 2]);

    let d = Demuxer::from_slice(&bytes).unwrap();
    let p: Vec<Packet> = d.map(|p| p.unwrap()).collect();
    assert_eq!(p.len(), 4);
    assert_eq!((p[0].track, p[0].pts, p[0].keyframe, p[0].duration), (0, 1000, true, 40));
    assert_eq!(p[0].data, [0, 0, 0, 1, 0x65, 1, 2]);
    assert_eq!((p[1].track, p[1].pts, p[1].data.len()), (1, 1005, 10));
    assert_eq!((p[2].pts, p[2].keyframe, p[2].duration), (1040, false, 80));
    assert_eq!(p[2].data, [0, 0, 0, 1, 0x41, 3]);
    assert_eq!((p[3].pts, p[3].pts_ns), (1120, 1_120_000_000));
    assert_eq!(f.read_sample(&bytes[..], 0, 1).unwrap(), p[2].data);

    // CRC mismatch is reported, not fatal
    let mut bad = bytes.clone();
    let title = bad.windows(9).position(|w| w == b"Synthetic").unwrap();
    bad[title] = b's';
    let f = open_with(&bad[..], &OpenOptions { index: true, verify_crc: true }).unwrap();
    assert_eq!(f.crc_errors.len(), 1);

    // seeking without index or cues builds the index
    let mut d = Demuxer::with_options(&bytes[..], OpenOptions { index: false, verify_crc: false }).unwrap();
    assert_eq!(d.file().tracks.len(), 2);
    assert_eq!(d.seek(0, 1_100_000_000).unwrap().pts, 1000);
    assert_eq!(d.seek(0, 1_130_000_000).unwrap().pts, 1120);
    assert_eq!(d.next_packet().unwrap().unwrap().pts, 1120);
    assert!(d.file().indexed);
}

/// One video track and one cluster at timestamp `ts` holding `content`.
fn one_cluster(ts: u64, content: &[u8]) -> Vec<u8> {
    one_cluster_scaled(1_000_000, ts, content)
}

fn one_cluster_scaled(scale: u64, ts: u64, content: &[u8]) -> Vec<u8> {
    let mut f = Vec::new();
    let mut h = Vec::new();
    el_str(&mut h, DOC_TYPE, "matroska");
    el(&mut f, EBML, &h);
    write_id(&mut f, SEGMENT);
    write_unknown_size(&mut f, 8);
    let mut info = Vec::new();
    el_uint(&mut info, TIMESTAMP_SCALE, scale);
    el(&mut f, INFO, &info);
    let mut v = Vec::new();
    el_uint(&mut v, TRACK_NUMBER, 1);
    el_uint(&mut v, TRACK_UID, 1);
    el_uint(&mut v, TRACK_TYPE, 1);
    el_str(&mut v, CODEC_ID, "V_VP9");
    let mut tr = Vec::new();
    el(&mut tr, TRACK_ENTRY, &v);
    el(&mut f, TRACKS, &tr);
    write_id(&mut f, CLUSTER);
    write_unknown_size(&mut f, 8);
    el_uint(&mut f, TIMESTAMP, ts);
    f.extend_from_slice(content);
    f
}

#[test]
fn empty_laced_frames_exhaust_the_frame_budget() {
    // Each 7-byte SimpleBlock is a fixed-size lace of 256 empty frames.
    let mut c = Vec::new();
    for _ in 0..100 {
        el(&mut c, SIMPLE_BLOCK, &block(1, 0, 0x04, &[0xFF]));
    }
    let r = open_with(&one_cluster(0, &c)[..], &OpenOptions::default());
    assert!(matches!(r, Err(Error::TooLarge(_))), "{r:?}");
    // Laces of one-byte frames are paid for by their bytes.
    let mut c = Vec::new();
    for _ in 0..4 {
        let mut data = vec![0xFF];
        data.extend_from_slice(&[0; 256]);
        el(&mut c, SIMPLE_BLOCK, &block(1, 0, 0x04, &data));
    }
    let f = open_with(&one_cluster(0, &c)[..], &OpenOptions::default()).unwrap();
    assert_eq!(f.tracks[0].samples.len(), 1024);
}

#[test]
fn track_entries_past_the_cap_are_rejected() {
    let file = |entries: usize| {
        let mut f = Vec::new();
        let mut h = Vec::new();
        el_str(&mut h, DOC_TYPE, "matroska");
        el(&mut f, EBML, &h);
        write_id(&mut f, SEGMENT);
        write_unknown_size(&mut f, 8);
        let mut tr = Vec::new();
        for _ in 0..entries {
            el(&mut tr, TRACK_ENTRY, &[0xD7, 0x81, 0x02]);
        }
        el(&mut f, TRACKS, &tr);
        f
    };
    assert_eq!(open_with(&file(3)[..], &OpenOptions::default()).unwrap().tracks.len(), 3);
    let r = open_with(&file(MAX_TRACKS + 1)[..], &OpenOptions::default());
    assert!(matches!(r, Err(Error::TooLarge(_))), "{r:?}");
}

/// Counts the bytes read from `data`.
struct Counting<'a>(&'a [u8], std::cell::Cell<u64>);

impl ByteSource for Counting<'_> {
    fn len(&self) -> u64 {
        self.0.len() as u64
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        self.1.set(self.1.get() + buf.len() as u64);
        self.0.read_at(offset, buf)
    }
}

#[test]
fn overlapping_seek_head_targets_are_read_once() {
    // SeekHead entries point at nested Tags headers, each running to the end of the file.
    let (n, pad) = (64u64, 128usize << 10);
    let entry = |pos: u64| {
        let mut s = Vec::new();
        el(&mut s, SEEK_ID, &TAGS.to_be_bytes());
        el(&mut s, SEEK_POSITION, &pos.to_be_bytes());
        let mut o = Vec::new();
        el(&mut o, SEEK, &s);
        o
    };
    let seek_head = |first: u64| {
        let entries: Vec<u8> = (0..n).flat_map(|i| entry(first + i * 12)).collect();
        let mut o = Vec::new();
        el(&mut o, SEEK_HEAD, &entries);
        o
    };
    let mut f = Vec::new();
    let mut h = Vec::new();
    el_str(&mut h, DOC_TYPE, "matroska");
    el(&mut f, EBML, &h);
    write_id(&mut f, SEGMENT);
    write_unknown_size(&mut f, 8);
    let first = seek_head(0).len() as u64;
    f.extend_from_slice(&seek_head(first));
    let total = n * 12 + pad as u64;
    for i in 0..n {
        write_id(&mut f, TAGS);
        ebml::write_size(&mut f, total - (i + 1) * 12, 8);
    }
    f.resize(f.len() + pad, 0xEC);
    let src = Counting(&f, std::cell::Cell::new(0));
    let file = open_with(&src, &OpenOptions { index: false, verify_crc: false }).unwrap();
    assert_eq!(file.seek_head.len(), n as usize);
    assert!(src.1.get() <= 4 * f.len() as u64, "read {} bytes of a {}-byte file", src.1.get(), f.len());
}

/// Fake Cluster candidates every 16 bytes: a Cluster of 512 KiB whose first child is a 64 KiB Void, so
/// `plausible` has to look 64 KiB away from each candidate and then finds a Cluster ID, not a Timestamp.
fn dense_fake_clusters(total: usize) -> Vec<u8> {
    let unit = fake_cluster_unit();
    assert_eq!(unit.len(), 16);
    let mut f = Vec::new();
    let mut h = Vec::new();
    el_str(&mut h, DOC_TYPE, "matroska");
    el(&mut f, EBML, &h);
    write_id(&mut f, SEGMENT);
    write_unknown_size(&mut f, 8);
    // an undecodable first Segment child forces a resync into the fake candidates
    f.push(0x00);
    while f.len() < total {
        f.extend_from_slice(&unit);
    }
    f
}

#[test]
fn resync_over_dense_fake_clusters_is_bounded_by_the_recovery_budget() {
    let f = dense_fake_clusters(1 << 20);
    let src = Counting(&f, std::cell::Cell::new(0));
    let _ = open_with(&src, &OpenOptions::default());
    let read = src.1.get();
    // 64 MiB of recovery work plus the one candidate that overdraws it; unbounded this is gigabytes
    assert!(read <= (64 << 20) + (1 << 20), "read {read} bytes of a {}-byte file", f.len());
}

#[test]
fn ebml_header_as_large_as_the_file_is_rejected_without_reading_it() {
    let mut f = Vec::new();
    write_id(&mut f, EBML);
    ebml::write_size(&mut f, (8 << 20) - 12, 8);
    f.resize(8 << 20, 0);
    let src = Counting(&f, std::cell::Cell::new(0));
    let r = open_with(&src, &OpenOptions::default());
    assert!(matches!(r, Err(Error::NotMatroska(_))), "{r:?}");
    assert!(src.1.get() <= 256 << 10, "read {} bytes", src.1.get());
}

#[test]
fn warnings_are_capped() {
    // 3000 SeekHead entries, each pointing at a Void instead of the Tags it claims
    let mut sh2 = Vec::new();
    for i in 0..3000u32 {
        let mut s = Vec::new();
        el(&mut s, SEEK_ID, &TAGS.to_be_bytes());
        el(&mut s, SEEK_POSITION, &(1000 + i).to_be_bytes());
        el(&mut sh2, SEEK, &s);
    }
    let mut g = Vec::new();
    let mut h = Vec::new();
    el_str(&mut h, DOC_TYPE, "matroska");
    el(&mut g, EBML, &h);
    write_id(&mut g, SEGMENT);
    write_unknown_size(&mut g, 8);
    el(&mut g, SEEK_HEAD, &sh2);
    el(&mut g, VOID, &vec![0xEC; 100_000]);
    let file = open_with(&g[..], &OpenOptions::default()).unwrap();
    assert_eq!(file.warnings.len(), 257, "256 kept and the note");
    assert_eq!(file.warnings.last().map(String::as_str), Some("further warnings omitted"));
}

/// `prefix` followed by zeros up to `len` bytes, without holding them.
struct Sparse {
    prefix: Vec<u8>,
    len: u64,
}

impl ByteSource for Sparse {
    fn len(&self) -> u64 {
        self.len
    }
    fn read_at(&self, offset: u64, buf: &mut [u8]) -> std::io::Result<()> {
        if offset.checked_add(buf.len() as u64).is_none_or(|e| e > self.len) {
            return Err(std::io::ErrorKind::UnexpectedEof.into());
        }
        if buf.len() > 64 << 20 {
            return Err(std::io::Error::other("a read this large is the bug under test"));
        }
        for (i, b) in buf.iter_mut().enumerate() {
            *b = self.prefix.get((offset as usize).saturating_add(i)).copied().unwrap_or(0);
        }
        Ok(())
    }
}

#[test]
fn a_block_larger_than_u32_is_not_indexed_with_a_truncated_size() {
    let frame = (1u64 << 32) + 100;
    let mut hdr = Vec::new();
    write_id(&mut hdr, SIMPLE_BLOCK);
    ebml::write_size(&mut hdr, 4 + frame, 8);
    hdr.extend_from_slice(&[0x81, 0, 0, 0x80]);
    let prefix = one_cluster(0, &hdr);
    let len = prefix.len() as u64 + frame;
    let f = open_with(Sparse { prefix, len }, &OpenOptions::default()).unwrap();
    assert!(f.tracks[0].samples.is_empty(), "{:?}", f.tracks[0].samples);
}

#[test]
fn a_laced_block_larger_than_the_cap_is_skipped_without_reading_its_body() {
    // flags 0x02 / 0x06 / 0x04: Xiph, EBML and fixed-size lacing
    for flags in [0x82u8, 0x86, 0x84] {
        let frame = (1u64 << 32) + 100;
        let mut hdr = Vec::new();
        write_id(&mut hdr, SIMPLE_BLOCK);
        ebml::write_size(&mut hdr, 4 + frame, 8);
        hdr.extend_from_slice(&[0x81, 0, 0, flags]);
        let prefix = one_cluster(0, &hdr);
        let len = prefix.len() as u64 + frame;
        let f = open_with(Sparse { prefix, len }, &OpenOptions::default());
        let f = f.unwrap_or_else(|e| panic!("flags {flags:#x}: {e:?}"));
        assert!(f.tracks[0].samples.is_empty(), "flags {flags:#x}: {:?}", f.tracks[0].samples);
    }
}

/// One fake Cluster candidate: it costs the resync two window refills, about 128 KiB.
fn fake_cluster_unit() -> Vec<u8> {
    let mut unit = Vec::new();
    write_id(&mut unit, CLUSTER);
    ebml::write_size(&mut unit, 512 << 10, 8);
    write_id(&mut unit, VOID);
    ebml::write_size(&mut unit, 64 << 10, 3);
    unit
}

/// Fake Cluster candidates, then a real Cluster.
fn damaged_before_a_cluster() -> Vec<u8> {
    let unit = fake_cluster_unit();
    let mut f = one_cluster(0, &[]);
    // cut the real cluster off: keep EBML, Segment, Info and Tracks
    let cluster_at = f.windows(4).position(|w| w == CLUSTER.to_be_bytes()).unwrap();
    f.truncate(cluster_at);
    f.push(0x00);
    // about 50 MiB of recovery reads: under the 64 MiB budget once, over it twice
    for _ in 0..400 {
        f.extend_from_slice(&unit);
    }
    f.resize(f.len() + (700 << 10), 0);
    write_id(&mut f, CLUSTER);
    write_unknown_size(&mut f, 8);
    el_uint(&mut f, TIMESTAMP, 0);
    el(&mut f, SIMPLE_BLOCK, &block(1, 0, 0x80, &[1, 2, 3]));
    f
}

#[test]
fn building_the_index_gets_its_own_recovery_budget() {
    let f = damaged_before_a_cluster();
    let opts = OpenOptions { index: false, verify_crc: false };
    let one = open_with(&f[..], &OpenOptions::default()).unwrap();
    assert_eq!(one.tracks[0].samples.len(), 1, "one pass finds the cluster: {:?}", one.warnings);
    let mut d = Demuxer::with_options(&f[..], opts).unwrap();
    d.build_index().unwrap();
    assert_eq!(d.file().tracks[0].samples.len(), 1, "the second pass found the cluster too: {:?}", d.file().warnings);
}

#[test]
fn packet_reading_keeps_its_recovery_budget_after_the_index_is_built() {
    // a Cluster, about 40 MiB of recovery reads of damage, then a second Cluster
    let mut f = one_cluster(0, &{
        let mut b = Vec::new();
        el(&mut b, SIMPLE_BLOCK, &block(1, 0, 0x80, &[1]));
        b
    });
    f.push(0x00);
    for _ in 0..320 {
        f.extend_from_slice(&fake_cluster_unit());
    }
    f.resize(f.len() + (700 << 10), 0);
    write_id(&mut f, CLUSTER);
    write_unknown_size(&mut f, 8);
    el_uint(&mut f, TIMESTAMP, 10);
    el(&mut f, SIMPLE_BLOCK, &block(1, 0, 0x80, &[2]));
    let mut d = Demuxer::with_options(&f[..], OpenOptions { index: false, verify_crc: false }).unwrap();
    d.build_index().unwrap();
    assert_eq!(d.file().tracks[0].samples.len(), 2, "the index found both clusters: {:?}", d.file().warnings);
    let mut n = 0;
    while d.next_packet().unwrap().is_some() {
        n += 1;
    }
    assert_eq!(n, 2, "reading ended at the damage: {:?}", d.file().warnings);
}

fn one_block(ts: u64, rel: i16) -> Vec<u8> {
    let mut b = Vec::new();
    el(&mut b, SIMPLE_BLOCK, &block(1, rel, 0x80, &[0]));
    one_cluster(ts, &b)
}

#[test]
fn hostile_cluster_timestamps_saturate() {
    let pts = |bytes: &[u8]| {
        let first = Demuxer::from_slice(bytes).unwrap().next_packet().unwrap().unwrap().pts;
        let indexed = open_with(bytes, &OpenOptions::default()).unwrap().tracks[0].samples[0].pts;
        assert_eq!(first, indexed);
        first
    };
    assert_eq!(pts(&one_block(1000, 5)), 1005);
    assert_eq!(pts(&one_block(i64::MAX as u64, 0)), i64::MAX);
    assert_eq!(pts(&one_block(i64::MAX as u64, 1)), i64::MAX);
    assert_eq!(pts(&one_block(u64::MAX, -3)), i64::MAX - 3);
}

#[test]
fn cue_position_past_the_file_falls_back_to_the_index() {
    for pos in [1 << 40, u64::MAX - 1] {
        let mut ctp = Vec::new();
        el_uint(&mut ctp, CUE_TRACK, 1);
        el_uint(&mut ctp, CUE_CLUSTER_POSITION, pos);
        let mut cp = Vec::new();
        el_uint(&mut cp, CUE_TIME, 0);
        el(&mut cp, CUE_TRACK_POSITIONS, &ctp);
        let mut cues = Vec::new();
        el(&mut cues, CUE_POINT, &cp);
        let mut bytes = one_block(1000, 0);
        let at = bytes.windows(4).position(|w| w == CLUSTER.to_be_bytes()).unwrap();
        let mut el_cues = Vec::new();
        el(&mut el_cues, CUES, &cues);
        bytes.splice(at..at, el_cues);
        let mut d = Demuxer::with_options(bytes, OpenOptions { index: false, verify_crc: false }).unwrap();
        assert_eq!(d.file().cues.len(), 1);
        assert_eq!(d.seek(0, 0).unwrap().pts, 1000);
    }
}

#[test]
fn huge_timestamp_scale_is_ignored() {
    // Three laced frames sharing the largest BlockDuration: with TimestampScale near 2^64 the
    // per-lace split overflowed u128 and i128.
    let mut g = Vec::new();
    el(&mut g, BLOCK, &block(1, 0, 0x02, &[2, 1, 1, b'a', b'b', b'c']));
    el_uint(&mut g, BLOCK_DURATION, u64::MAX);
    let mut c = Vec::new();
    el(&mut c, BLOCK_GROUP, &g);
    let f = open_with(&one_cluster_scaled(u64::MAX, i64::MAX as u64, &c)[..], &OpenOptions::default()).unwrap();
    assert_eq!(f.info.timestamp_scale, 1_000_000);
    assert_eq!(f.tracks[0].samples.len(), 3);
    let f = open_with(&one_cluster_scaled(1 << 32, 0, &c)[..], &OpenOptions::default()).unwrap();
    assert_eq!(f.info.timestamp_scale, 1 << 32);
}

#[test]
fn duration_of_a_block_at_the_largest_timestamp_saturates() {
    for d in [80, u64::MAX] {
        let mut g = Vec::new();
        el(&mut g, BLOCK, &block(1, 0, 0, &[0]));
        el_uint(&mut g, BLOCK_DURATION, d);
        let mut c = Vec::new();
        el(&mut c, BLOCK_GROUP, &g);
        let f = open_with(&one_cluster(i64::MAX as u64, &c)[..], &OpenOptions::default()).unwrap();
        assert_eq!(f.duration_ns(), Some(u64::MAX));
    }
}
