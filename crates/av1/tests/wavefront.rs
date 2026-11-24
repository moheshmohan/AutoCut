//! Stress tests of the frame threading (parse / reconstruct split and the row wavefront across
//! frames): every thread count gives the single-thread pictures, run after run, and dropping or
//! flushing a busy decoder never hangs. A hang fails the test through the deadline.

mod common;
use common::*;
use filmcraft_av1::{Decoder, Picture};

/// A GOP with many superblock rows (the wavefront has rows to overlap), hierarchical references
/// and motion reaching across rows.
fn spec() -> Spec {
    Spec::new("wavefront_640x480", "testsrc2=s=640x480:r=25,rotate=a=t*0.5:c=black", 24, "yuv420p", &["-preset", "8", "-crf", "38"])
}

/// Seconds a test may take before it counts as a hang (override: FILMCRAFT_HOSTILE_TIMEOUT_SECS).
const DEADLINE_SECS: u64 = 300;

fn decode(tus: &[Vec<u8>], threads: usize) -> Vec<Picture> {
    let mut dec = Decoder::with_threads(threads);
    let mut pics = Vec::new();
    for (i, tu) in tus.iter().enumerate() {
        pics.extend(dec.decode_pts(tu, i as i64).unwrap_or_else(|e| panic!("threads {threads}, unit {i}: {e}")));
    }
    pics.extend(dec.flush_result().unwrap_or_else(|e| panic!("threads {threads}, flush: {e}")));
    pics
}

/// `spec` decoded with every thread count of `threads`, `rounds` times each, gives the
/// single-thread pictures (and libdav1d's, through `check_bit_exact`).
fn same_pictures_at_every_thread_count(ff: &std::path::Path, spec: &Spec, frames: usize, threads: &'static [usize], rounds: usize) {
    check_bit_exact(ff, spec);
    let tus = ivf_frames(&make(ff, spec));
    let name = spec.name;
    filmcraft_testkit::with_deadline(DEADLINE_SECS, move || {
        let want = decode(&tus, 1);
        assert_eq!(want.len(), frames, "{name}");
        for &threads in threads {
            for round in 0..rounds {
                assert!(decode(&tus, threads) == want, "{name}: threads {threads}, round {round}: pictures differ");
            }
        }
    });
}

#[test]
fn every_thread_count_gives_the_single_thread_pictures_repeatedly() {
    let Some(ff) = ffmpeg() else { return };
    same_pictures_at_every_thread_count(&ff, &spec(), 24, &[2, 3, 4, 5, 8, 14, 16], 4);
}

/// 10-bit, 128x128 superblocks in two tile columns, film grain on the output: the wavefront over
/// the superblock rows of every tile column.
#[test]
fn superblocks_of_128_in_tile_columns_give_the_single_thread_pictures() {
    let Some(ff) = ffmpeg() else { return };
    let enc = ["-preset", "6", "-crf", "38", "-svtav1-params", "sb-size=128:tile-columns=1:film-grain=8"];
    let spec = Spec::new("wavefront_sb128_tiles_10bit", "testsrc2=s=720x400:r=25,rotate=a=t*0.5:c=black", 16, "yuv420p10le", &enc);
    same_pictures_at_every_thread_count(&ff, &spec, 16, &[2, 3, 4, 8, 16], 2);
}

/// An odd frame size (partial superblocks and 8x8 blocks on the right and bottom edges) in a
/// 2x2 tile grid.
#[test]
fn odd_sizes_in_tiles_give_the_single_thread_pictures() {
    let Some(ff) = ffmpeg() else { return };
    let enc = ["-preset", "7", "-crf", "36", "-svtav1-params", "tile-columns=1:tile-rows=1"];
    let spec = Spec::new("wavefront_odd_tiles", "mandelbrot=s=771x443:r=25", 12, "yuv420p", &enc);
    same_pictures_at_every_thread_count(&ff, &spec, 12, &[2, 4, 8, 14], 2);
}

#[test]
fn dropping_or_flushing_a_busy_decoder_never_hangs() {
    let Some(ff) = ffmpeg() else { return };
    let tus = ivf_frames(&make(&ff, &spec()));
    filmcraft_testkit::with_deadline(DEADLINE_SECS, move || {
        let want = decode(&tus, 1);
        for threads in [2, 3, 8, 16] {
            for cut in [1, 2, 7, 12, 23] {
                // Dropped with frames in flight (no flush).
                let mut dec = Decoder::with_threads(threads);
                for (i, tu) in tus.iter().take(cut).enumerate() {
                    dec.decode_pts(tu, i as i64).expect("decode");
                }
                drop(dec);
                // Flushed mid-stream (as before a seek), then continued: the same pictures.
                let mut dec = Decoder::with_threads(threads);
                let mut pics = Vec::new();
                for (i, tu) in tus.iter().enumerate() {
                    if i == cut {
                        pics.extend(dec.flush_result().expect("flush"));
                    }
                    pics.extend(dec.decode_pts(tu, i as i64).expect("decode"));
                }
                pics.extend(dec.flush_result().expect("flush"));
                assert!(pics == want, "threads {threads}, flush before unit {cut}: pictures differ");
            }
        }
    });
}
