//! A tiny SoundFont made in code, for tests: no download, no file in the repository.
//!
//! [`tiny`] holds two samples — a looped 441 Hz sine (100-sample period, root key A4) and a
//! decaying noise burst — as two instruments, "Sine" and "Noise", and presets for the General MIDI
//! programs the music engine plays (all on "Sine") plus the drum kit bank 128 program 0 ("Noise").
//! [`build`] makes a font with the programs you choose.

/// Generator ids used here (SoundFont 2.04 §8.1.3).
const GEN_RELEASE_VOL_ENV: u16 = 38;
const GEN_COARSE_TUNE: u16 = 51;
const GEN_INSTRUMENT: u16 = 41;
const GEN_SAMPLE_ID: u16 = 53;
const GEN_SAMPLE_MODES: u16 = 54;

/// The General MIDI programs the music engine's patches use.
pub const MUSIC_PROGRAMS: [u16; 13] = [0, 4, 11, 33, 38, 39, 45, 48, 61, 73, 81, 89, 90];

fn put_u16(v: &mut Vec<u8>, x: u16) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn put_u32(v: &mut Vec<u8>, x: u32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn put_i32(v: &mut Vec<u8>, x: i32) {
    v.extend_from_slice(&x.to_le_bytes());
}

fn name20(v: &mut Vec<u8>, name: &str) {
    let mut b = [0u8; 20];
    for (d, s) in b.iter_mut().zip(name.bytes().take(19)) {
        *d = s;
    }
    v.extend_from_slice(&b);
}

fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(body.len() + 9);
    v.extend_from_slice(id);
    put_u32(&mut v, u32::try_from(body.len()).unwrap_or(u32::MAX));
    v.extend_from_slice(body);
    if !body.len().is_multiple_of(2) {
        v.push(0);
    }
    v
}

fn list(kind: &[u8; 4], chunks: &[Vec<u8>]) -> Vec<u8> {
    let mut body = kind.to_vec();
    for c in chunks {
        body.extend_from_slice(c);
    }
    chunk(b"LIST", &body)
}

/// A font named `name` with the "Sine" instrument on each of `programs` (bank 0) and, with
/// `drums`, the "Noise" instrument as the bank 128 kit 0.
pub fn build(name: &str, programs: &[u16], drums: bool) -> Vec<u8> {
    build_inner(name, programs, drums, &[], (100, 1900))
}

/// A font that passes [`crate::check::check`] but makes rustysynth fail while it plays: its sine
/// loops over two samples and is tuned ten octaves up, so the read position outruns the loop.
/// Used to prove a failure inside rustysynth silences the part instead of escaping.
pub fn runaway() -> Vec<u8> {
    build_inner("Runaway Font", &MUSIC_PROGRAMS, false, &[(GEN_COARSE_TUNE, 120)], (100, 102))
}

fn build_inner(name: &str, programs: &[u16], drums: bool, sine_extra: &[(u16, u16)], sine_loop: (usize, usize)) -> Vec<u8> {
    // samples: a 2000-sample sine (20 periods of 100) then 46 zeros, a 4410-sample noise burst then 46 zeros
    let mut smpl: Vec<i16> = (0..2000).map(|i| ((i as f64 * std::f64::consts::TAU / 100.0).sin() * 20_000.0) as i16).collect();
    smpl.extend(std::iter::repeat_n(0, 46));
    let noise_start = smpl.len();
    let mut x: u32 = 0x1234_5678;
    for i in 0..4410 {
        x = x.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let white = f64::from((x >> 16) as u16) / 32_768.0 - 1.0;
        smpl.push((white * 24_000.0 * (-(i as f64) / 600.0).exp()) as i16);
    }
    let noise_end = smpl.len();
    smpl.extend(std::iter::repeat_n(0, 46));
    let mut smpl_bytes = Vec::with_capacity(smpl.len() * 2);
    for s in &smpl {
        smpl_bytes.extend_from_slice(&s.to_le_bytes());
    }

    let mut ifil = Vec::new();
    put_u16(&mut ifil, 2);
    put_u16(&mut ifil, 1);
    let mut inam = name.as_bytes().to_vec();
    inam.push(0);
    if !inam.len().is_multiple_of(2) {
        inam.push(0);
    }
    let info = list(b"INFO", &[chunk(b"ifil", &ifil), chunk(b"isng", b"EMU8000\0"), chunk(b"INAM", &inam)]);
    let sdta = list(b"sdta", &[chunk(b"smpl", &smpl_bytes)]);

    // presets: one zone each, its only generator the instrument
    let mut presets: Vec<(&str, u16, u16, u16)> = programs.iter().map(|p| ("Sine", *p, 0u16, 0u16)).collect();
    if drums {
        presets.push(("Noise Kit", 0, 128, 1));
    }
    let (mut phdr, mut pbag, mut pgen) = (Vec::new(), Vec::new(), Vec::new());
    for (i, (pname, program, bank, inst)) in presets.iter().enumerate() {
        name20(&mut phdr, pname);
        put_u16(&mut phdr, *program);
        put_u16(&mut phdr, *bank);
        put_u16(&mut phdr, i as u16);
        put_u32(&mut phdr, 0);
        put_u32(&mut phdr, 0);
        put_u32(&mut phdr, 0);
        put_u16(&mut pbag, i as u16);
        put_u16(&mut pbag, 0);
        put_u16(&mut pgen, GEN_INSTRUMENT);
        put_u16(&mut pgen, *inst);
    }
    name20(&mut phdr, "EOP");
    put_u16(&mut phdr, 0);
    put_u16(&mut phdr, 0);
    put_u16(&mut phdr, presets.len() as u16);
    phdr.extend_from_slice(&[0; 12]);
    put_u16(&mut pbag, presets.len() as u16);
    put_u16(&mut pbag, 0);
    pgen.extend_from_slice(&[0; 4]);

    // instruments: Sine (looped, half-second release), Noise
    let mut inst = Vec::new();
    for (i, n) in ["Sine", "Noise", "EOI"].iter().enumerate() {
        name20(&mut inst, n);
        put_u16(&mut inst, i as u16);
    }
    let mut sine: Vec<(u16, u16)> = sine_extra.to_vec();
    sine.extend([(GEN_SAMPLE_MODES, 1u16), (GEN_RELEASE_VOL_ENV, (-1200i16) as u16), (GEN_SAMPLE_ID, 0)]);
    let n = sine.len() as u16;
    let mut ibag = Vec::new();
    for (g, m) in [(0u16, 0u16), (n, 0), (n + 1, 0)] {
        put_u16(&mut ibag, g);
        put_u16(&mut ibag, m);
    }
    let mut igen = Vec::new();
    for (g, v) in sine.into_iter().chain([(GEN_SAMPLE_ID, 1), (0, 0)]) {
        put_u16(&mut igen, g);
        put_u16(&mut igen, v);
    }

    let mut shdr = Vec::new();
    let mut header = |name: &str, start: usize, end: usize, ls: usize, le: usize, root: u8| {
        name20(&mut shdr, name);
        for x in [start, end, ls, le] {
            put_i32(&mut shdr, i32::try_from(x).unwrap_or(0));
        }
        put_u32(&mut shdr, 44_100);
        shdr.push(root);
        shdr.push(0);
        put_u16(&mut shdr, 0);
        put_u16(&mut shdr, if name == "EOS" { 0 } else { 1 });
    };
    header("Sine", 0, 2000, sine_loop.0, sine_loop.1, 69);
    header("Noise", noise_start, noise_end, noise_start, noise_end, 60);
    header("EOS", 0, 0, 0, 0, 0);

    let terminal_mod = [0u8; 10];
    let pdta = list(
        b"pdta",
        &[
            chunk(b"phdr", &phdr),
            chunk(b"pbag", &pbag),
            chunk(b"pmod", &terminal_mod),
            chunk(b"pgen", &pgen),
            chunk(b"inst", &inst),
            chunk(b"ibag", &ibag),
            chunk(b"imod", &terminal_mod),
            chunk(b"igen", &igen),
            chunk(b"shdr", &shdr),
        ],
    );
    let mut body = b"sfbk".to_vec();
    body.extend_from_slice(&info);
    body.extend_from_slice(&sdta);
    body.extend_from_slice(&pdta);
    chunk(b"RIFF", &body)
}

/// The test font: every program the music engine plays, and a drum kit.
pub fn tiny() -> Vec<u8> {
    build("FilmCraft Test Font", &MUSIC_PROGRAMS, true)
}
