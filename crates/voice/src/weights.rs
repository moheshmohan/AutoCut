//! The DeepFilterNet 3 weights, taken out of the model archive's three ONNX files and the
//! training config. Nothing in the files runs: [`crate::net`] implements the network, and this
//! module finds each layer by the name of the node that computes it, checks the node's operator,
//! attributes (groups, kernel, strides, padding) and every tensor's shape against what the code
//! implements, and refuses anything else, so a different file is an error, never misread.

use std::collections::HashMap;

use filmcraft_vision::onnx::{self, Attr, Data, Limits, Model, Node, TensorData};

use crate::VoiceError;
use crate::archive;
use crate::net::{CH, Conv, ConvT, DF_ORDER, EMB, GLinear, Gru, HIDDEN, NB_DF, Pointwise, Weights};

fn bad(what: impl Into<String>) -> VoiceError {
    VoiceError::Model(what.into())
}

/// One ONNX file: its nodes by first output, and its tensors (initializers and constants).
struct Graph {
    file: &'static str,
    nodes: HashMap<String, Node>,
    tensors: HashMap<String, TensorData>,
}

impl Graph {
    fn new(file: &'static str, m: Model) -> Self {
        let mut tensors = m.initializers;
        let mut nodes = HashMap::new();
        for n in m.nodes {
            if n.op == "Constant"
                && let (Some(out), Some(Attr::Tensor(t))) = (n.outputs.first(), n.attr("value"))
            {
                tensors.insert(out.clone(), t.clone());
            }
            if let Some(out) = n.outputs.first() {
                nodes.insert(out.clone(), n);
            }
        }
        Graph { file, nodes, tensors }
    }

    fn node(&self, out: &str, op: &str) -> Result<&Node, VoiceError> {
        let n = self.nodes.get(out).ok_or_else(|| bad(format!("{}: no node computes {out}", self.file)))?;
        if n.op != op {
            return Err(bad(format!("{}: {out} is computed by {}, not {op}", self.file, n.op)));
        }
        Ok(n)
    }

    /// Input `i` of `n` as a float tensor of shape `dims`.
    fn input(&self, n: &Node, i: usize, dims: &[usize]) -> Result<Vec<f32>, VoiceError> {
        let name = n.inputs.get(i).filter(|s| !s.is_empty()).ok_or_else(|| bad(format!("{}: {} lacks input {i}", self.file, n.name)))?;
        self.tensor(name, dims)
    }

    fn tensor(&self, name: &str, dims: &[usize]) -> Result<Vec<f32>, VoiceError> {
        let t = self.tensors.get(name).ok_or_else(|| bad(format!("{}: tensor {name} is not in the file", self.file)))?;
        if t.dims != dims {
            return Err(bad(format!("{}: {name} is {:?}, expected {dims:?}", self.file, t.dims)));
        }
        match &t.data {
            Data::F32(v) if v.iter().all(|x| x.is_finite()) => Ok(v.clone()),
            Data::F32(_) => Err(bad(format!("{}: {name} has non-finite weights", self.file))),
            Data::I64(_) => Err(bad(format!("{}: {name} is not float", self.file))),
        }
    }

    fn ints(&self, n: &Node, attr: &str) -> Vec<i64> {
        n.ints(attr).map(<[i64]>::to_vec).unwrap_or_default()
    }

    /// A `Conv` node: `out_ch × (in_ch / groups) × kt × kf`, with the given stride along frequency
    /// and symmetric frequency padding `kf / 2` (none in time: the history supplies it).
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, out: &str, out_ch: usize, in_ch: usize, groups: usize, kt: usize, kf: usize, stride: usize, bias: bool) -> Result<Conv, VoiceError> {
        let n = self.node(out, "Conv")?;
        let g = n.int("group").unwrap_or(1);
        let pad = kf / 2;
        let want_pads = [0, pad as i64, 0, pad as i64];
        let pads = self.ints(n, "pads");
        let pads_ok = pads.is_empty() && pad == 0 || pads == want_pads;
        let strides = self.ints(n, "strides");
        let strides_ok = strides.is_empty() && stride == 1 || strides == [1, stride as i64];
        let dil = self.ints(n, "dilations");
        if g != groups as i64 || !pads_ok || !strides_ok || !(dil.is_empty() || dil == [1, 1]) || self.ints(n, "kernel_shape") != [kt as i64, kf as i64] {
            return Err(bad(format!(
                "{}: {out} is not the {out_ch}×{in_ch} {kt}×{kf} convolution (groups {groups}, stride {stride}) this code implements",
                self.file
            )));
        }
        let w = self.input(n, 1, &[out_ch, in_ch / groups, kt, kf])?;
        let b = if bias {
            self.input(n, 2, &[out_ch])?
        } else {
            if n.inputs.get(2).is_some_and(|s| !s.is_empty()) {
                return Err(bad(format!("{}: {out} has a bias this code does not expect", self.file)));
            }
            vec![0.0; out_ch]
        };
        Ok(Conv { out_ch, in_ch, groups, kt, kf, stride, pad, w, b })
    }

    fn pointwise(&self, out: &str, ch_out: usize, ch_in: usize) -> Result<Pointwise, VoiceError> {
        let c = self.conv(out, ch_out, ch_in, 1, 1, 1, 1, true)?;
        Ok(Pointwise { out_ch: ch_out, in_ch: ch_in, w: c.w, b: c.b })
    }

    fn conv_t(&self, out: &str) -> Result<ConvT, VoiceError> {
        let n = self.node(out, "ConvTranspose")?;
        if n.int("group") != Some(CH as i64)
            || self.ints(n, "kernel_shape") != [1, 3]
            || self.ints(n, "strides") != [1, 2]
            || self.ints(n, "pads") != [0, 1, 0, 1]
            || self.ints(n, "output_padding") != [0, 1]
            || n.inputs.get(2).is_some_and(|s| !s.is_empty())
        {
            return Err(bad(format!("{}: {out} is not the depthwise 1×3 stride-2 transposed convolution this code implements", self.file)));
        }
        Ok(ConvT { ch: CH, w: self.input(n, 1, &[CH, 1, 1, 3])? })
    }

    fn glinear(&self, out: &str, groups: usize, inp: usize, outp: usize) -> Result<GLinear, VoiceError> {
        let n = self.node(out, "Einsum")?;
        match n.attr("equation") {
            Some(Attr::Str(e)) if e.replace(' ', "") == "btgi,gih->btgh" => {}
            _ => return Err(bad(format!("{}: {out} is not a grouped linear layer", self.file))),
        }
        Ok(GLinear { groups, inp, out: outp, w: self.input(n, 1, &[groups, inp, outp])? })
    }

    fn gru(&self, out: &str, input: usize) -> Result<Gru, VoiceError> {
        let n = self.node(out, "GRU")?;
        let dir_ok = match n.attr("direction") {
            None => true,
            Some(Attr::Str(d)) => d == "forward",
            Some(_) => false,
        };
        if n.int("hidden_size") != Some(HIDDEN as i64) || n.int("linear_before_reset") != Some(1) || !dir_ok || n.attr("activations").is_some() {
            return Err(bad(format!("{}: {out} is not the GRU this code implements", self.file)));
        }
        let w = self.input(n, 1, &[1, 3 * HIDDEN, input])?;
        let r = self.input(n, 2, &[1, 3 * HIDDEN, HIDDEN])?;
        let b = self.input(n, 3, &[1, 6 * HIDDEN])?;
        let (bw, br) = b.split_at(3 * HIDDEN);
        Ok(Gru { input, hidden: HIDDEN, w, r, bw: bw.to_vec(), br: br.to_vec() })
    }

    /// A scalar constant feeding `out` (`Mul` / `Add` with a constant).
    fn scalar_input(&self, out: &str, op: &str) -> Result<f32, VoiceError> {
        let n = self.node(out, op)?;
        for name in &n.inputs {
            if let Some(t) = self.tensors.get(name)
                && t.dims.iter().product::<usize>() == 1
                && let Data::F32(v) = &t.data
                && let Some(x) = v.first()
            {
                return Ok(*x);
            }
        }
        Err(bad(format!("{}: {out} has no constant operand", self.file)))
    }
}

/// The model's `[df]` / `[deepfilternet]` settings this code was written for.
const EXPECT: &[(&str, &str, &str)] = &[
    ("df", "sr", "48000"),
    ("df", "fft_size", "960"),
    ("df", "hop_size", "480"),
    ("df", "nb_erb", "32"),
    ("df", "nb_df", "96"),
    ("df", "norm_tau", "1"),
    ("df", "lsnr_max", "35"),
    ("df", "lsnr_min", "-15"),
    ("df", "min_nb_erb_freqs", "2"),
    ("df", "df_order", "5"),
    ("df", "df_lookahead", "2"),
    ("deepfilternet", "conv_lookahead", "2"),
    ("deepfilternet", "conv_ch", "64"),
    ("deepfilternet", "emb_hidden_dim", "256"),
    ("deepfilternet", "emb_num_layers", "3"),
    ("deepfilternet", "df_hidden_dim", "256"),
    ("deepfilternet", "df_num_layers", "2"),
    ("deepfilternet", "mask_pf", "False"),
    ("train", "model", "deepfilternet3"),
];

/// `section.key → value` of an INI file.
fn ini(text: &str) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut section = String::new();
    for line in text.lines().take(10_000) {
        let l = line.trim();
        if l.starts_with('[') && l.ends_with(']') {
            section = l[1..l.len() - 1].trim().to_string();
        } else if let Some((k, v)) = l.split_once('=') {
            out.insert(format!("{section}.{}", k.trim()), v.trim().to_string());
        }
    }
    out
}

/// The weights from the model archive (`DeepFilterNet3_onnx.tar.gz`).
pub fn from_archive(bytes: &[u8]) -> Result<Weights, VoiceError> {
    let tar = archive::gunzip(bytes)?;
    let files = archive::untar(&tar)?;
    let get =
        |name: &str| files.iter().find(|e| e.base_name() == name).map(|e| e.data.as_slice()).ok_or_else(|| bad(format!("the model archive has no {name}")));
    let cfg = ini(&String::from_utf8_lossy(get("config.ini")?));
    for (sec, key, want) in EXPECT {
        let got = cfg.get(&format!("{sec}.{key}")).map(String::as_str).unwrap_or("");
        let same = got == *want || got.parse::<f64>().ok().zip(want.parse::<f64>().ok()).is_some_and(|(a, b)| a == b);
        if !same {
            return Err(bad(format!("the model's {sec}.{key} is {got:?}; this code implements {want}")));
        }
    }
    let lim = Limits::default();
    let enc = Graph::new("enc.onnx", onnx::parse(get("enc.onnx")?, &lim).map_err(|e| bad(format!("enc.onnx: {e}")))?);
    let erb = Graph::new("erb_dec.onnx", onnx::parse(get("erb_dec.onnx")?, &lim).map_err(|e| bad(format!("erb_dec.onnx: {e}")))?);
    let df = Graph::new("df_dec.onnx", onnx::parse(get("df_dec.onnx")?, &lim).map_err(|e| bad(format!("df_dec.onnx: {e}")))?);
    from_graphs(&enc, &erb, &df)
}

fn from_graphs(enc: &Graph, erb: &Graph, df: &Graph) -> Result<Weights, VoiceError> {
    let sep = |g: &Graph, dw: &str, pw: &str, stride: usize| -> Result<(Conv, Pointwise), VoiceError> {
        Ok((g.conv(dw, CH, CH, CH, 1, 3, stride, false)?, g.pointwise(pw, CH, CH)?))
    };
    let lsnr_n = enc.node("/lsnr_fc/0/MatMul_output_0", "MatMul")?;
    let lsnr_w = enc.input(lsnr_n, 1, &[EMB, 1])?;
    let lsnr_b = enc.scalar_input("/lsnr_fc/0/Add_output_0", "Add")?;
    let lsnr_scale = enc.scalar_input("/Mul_output_0", "Mul")?;
    let lsnr_offset = enc.scalar_input("lsnr", "Add")?;
    if (lsnr_scale - 50.0).abs() > 1e-4 || (lsnr_offset + 15.0).abs() > 1e-4 {
        return Err(bad(format!("enc.onnx: the local SNR is scaled by {lsnr_scale} and offset by {lsnr_offset}, not 50 and −15")));
    }
    let pathway = |g: &Graph, out: &str| g.conv(out, CH, CH, CH, 1, 1, 1, true);
    Ok(Weights {
        erb_conv0: enc.conv("/erb_conv0/1/Conv_output_0", CH, 1, 1, 3, 3, 1, true)?,
        erb_conv1: sep(enc, "/erb_conv1/0/Conv_output_0", "/erb_conv1/1/Conv_output_0", 2)?,
        erb_conv2: sep(enc, "/erb_conv2/Conv_output_0", "/erb_conv2/Conv_1_output_0", 2)?,
        erb_conv3: sep(enc, "/erb_conv3/0/Conv_output_0", "/erb_conv3/1/Conv_output_0", 1)?,
        df_conv0: (enc.conv("/df_conv0/1/Conv_output_0", CH, 2, 2, 3, 3, 1, false)?, enc.pointwise("/df_conv0/2/Conv_output_0", CH, CH)?),
        df_conv1: sep(enc, "/df_conv1/Conv_output_0", "/df_conv1/Conv_1_output_0", 2)?,
        df_fc_emb: enc.glinear("/df_fc_emb/0/Einsum_output_0", 32, CH * NB_DF / 2 / 32, EMB / 32)?,
        enc_in: enc.glinear("/emb_gru/linear_in/0/Einsum_output_0", 16, EMB / 16, HIDDEN / 16)?,
        enc_gru: enc.gru("/emb_gru/GRU_output_0", HIDDEN)?,
        enc_out: enc.glinear("/emb_gru/linear_out/0/Einsum_output_0", 16, HIDDEN / 16, EMB / 16)?,
        lsnr_w,
        lsnr_b,
        lsnr_scale,
        lsnr_offset,
        erb_in: erb.glinear("/emb_gru/linear_in/Einsum_output_0", 16, EMB / 16, HIDDEN / 16)?,
        erb_gru: [erb.gru("/emb_gru/GRU_output_0", HIDDEN)?, erb.gru("/emb_gru/GRU_1_output_0", HIDDEN)?],
        erb_out: erb.glinear("/emb_gru/linear_out/Einsum_output_0", 16, HIDDEN / 16, EMB / 16)?,
        conv3p: pathway(erb, "/conv3p/0/Conv_output_0")?,
        convt3: sep(erb, "/convt3/Conv_output_0", "/convt3/Conv_1_output_0", 1)?,
        conv2p: pathway(erb, "/conv2p/Conv_output_0")?,
        convt2: (erb.conv_t("/convt2/0/ConvTranspose_output_0")?, erb.pointwise("/convt2/1/Conv_output_0", CH, CH)?),
        conv1p: pathway(erb, "/conv1p/Conv_output_0")?,
        convt1: (erb.conv_t("/convt1/ConvTranspose_output_0")?, erb.pointwise("/convt1/Conv_output_0", CH, CH)?),
        conv0p: pathway(erb, "/conv0p/Conv_output_0")?,
        conv0_out: erb.conv("/conv0_out/0/Conv_output_0", 1, CH, 1, 1, 3, 1, true)?,
        df_in: df.glinear("/df_gru/linear_in/linear_in.0/Einsum_output_0", 8, EMB / 8, HIDDEN / 8)?,
        df_gru: [df.gru("/df_gru/gru/GRU_output_0", HIDDEN)?, df.gru("/df_gru/gru/GRU_1_output_0", HIDDEN)?],
        df_skip: df.glinear("/df_skip/Einsum_output_0", 16, EMB / 16, HIDDEN / 16)?,
        df_convp: (
            df.conv("/df_convp/df_convp.1/Conv_output_0", 2 * DF_ORDER, CH, 2, DF_ORDER, 1, 1, false)?,
            df.pointwise("/df_convp/df_convp.2/Conv_output_0", 2 * DF_ORDER, 2 * DF_ORDER)?,
        ),
        df_out: df.glinear("/df_out/df_out.0/Einsum_output_0", 16, HIDDEN / 16, NB_DF * 2 * DF_ORDER / 16)?,
    })
}

/// Weights of the right shapes filled with small pseudo-random values (tests of the plumbing).
pub fn random(seed: u64) -> Weights {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut v = |n: usize, scale: f32| -> Vec<f32> {
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    };
    let mut conv = |out_ch: usize, in_ch: usize, groups: usize, kt: usize, kf: usize, stride: usize| {
        let fan = (in_ch / groups * kt * kf) as f32;
        Conv { out_ch, in_ch, groups, kt, kf, stride, pad: kf / 2, w: v(out_ch * in_ch / groups * kt * kf, 1.0 / fan.sqrt()), b: v(out_ch, 0.05) }
    };
    let erb_conv0 = conv(CH, 1, 1, 3, 3, 1);
    let [dw0, dw1, dw2, dw3, dw4] = [2, 2, 1, 2, 1].map(|stride| conv(CH, CH, CH, 1, 3, stride));
    let df_conv0 = conv(CH, 2, 2, 3, 3, 1);
    let [p0, p1, p2, p3] = [0; 4].map(|_| conv(CH, CH, CH, 1, 1, 1));
    let conv0_out = conv(1, CH, 1, 1, 3, 1);
    let df_convp = conv(2 * DF_ORDER, CH, 2, DF_ORDER, 1, 1);
    let mut pw = |o: usize, i: usize| Pointwise { out_ch: o, in_ch: i, w: v(o * i, 1.0 / (i as f32).sqrt()), b: v(o, 0.05) };
    let [w0, w1, w2, w3, w4, w5, w6, w7] = [0; 8].map(|_| pw(CH, CH));
    let pw10 = pw(2 * DF_ORDER, 2 * DF_ORDER);
    let mut gl = |g: usize, i: usize, o: usize| GLinear { groups: g, inp: i, out: o, w: v(g * i * o, 1.0 / (i as f32).sqrt()) };
    let gls = [
        gl(32, CH * NB_DF / 2 / 32, EMB / 32),
        gl(16, EMB / 16, HIDDEN / 16),
        gl(16, HIDDEN / 16, EMB / 16),
        gl(16, EMB / 16, HIDDEN / 16),
        gl(16, HIDDEN / 16, EMB / 16),
        gl(8, EMB / 8, HIDDEN / 8),
        gl(16, EMB / 16, HIDDEN / 16),
        gl(16, HIDDEN / 16, NB_DF * 2 * DF_ORDER / 16),
    ];
    let mut gru = || Gru {
        input: HIDDEN,
        hidden: HIDDEN,
        w: v(3 * HIDDEN * HIDDEN, 0.06),
        r: v(3 * HIDDEN * HIDDEN, 0.06),
        bw: v(3 * HIDDEN, 0.1),
        br: v(3 * HIDDEN, 0.1),
    };
    let grus = [gru(), gru(), gru(), gru(), gru()];
    let convts = [ConvT { ch: CH, w: v(3 * CH, 0.5) }, ConvT { ch: CH, w: v(3 * CH, 0.5) }];
    let lsnr_w = v(EMB, 0.05);
    let [g0, g1, g2, g3, g4, g5, g6, g7] = gls;
    let [r0, r1, r2, r3, r4] = grus;
    let [t0, t1] = convts;
    Weights {
        erb_conv0,
        erb_conv1: (dw0, w0),
        erb_conv2: (dw1, w1),
        erb_conv3: (dw2, w2),
        df_conv0: (df_conv0, w3),
        df_conv1: (dw3, w4),
        df_fc_emb: g0,
        enc_in: g1,
        enc_gru: r0,
        enc_out: g2,
        lsnr_w,
        lsnr_b: 0.0,
        lsnr_scale: 50.0,
        lsnr_offset: -15.0,
        erb_in: g3,
        erb_gru: [r1, r2],
        erb_out: g4,
        conv3p: p0,
        convt3: (dw4, w5),
        conv2p: p1,
        convt2: (t0, w6),
        conv1p: p2,
        convt1: (t1, w7),
        conv0p: p3,
        conv0_out,
        df_in: g5,
        df_gru: [r3, r4],
        df_skip: g6,
        df_convp: (df_convp, pw10),
        df_out: g7,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_config_must_be_the_one_implemented() {
        let mut cfg = String::from("[train]\nmodel = deepfilternet3\n[df]\n");
        for (s, k, v) in EXPECT.iter().filter(|e| e.0 == "df") {
            let _ = s;
            cfg.push_str(&format!("{k} = {v}\n"));
        }
        cfg.push_str("[deepfilternet]\n");
        for (_, k, v) in EXPECT.iter().filter(|e| e.0 == "deepfilternet") {
            cfg.push_str(&format!("{k} = {v}\n"));
        }
        let m = ini(&cfg);
        assert_eq!(m.get("df.sr").map(String::as_str), Some("48000"));
        // an archive with the right config but no networks is refused by name
        let t = archive::tests::tar(&[("x/config.ini", cfg.as_bytes())]);
        let e = from_archive(&archive::tests::gzip(&t)).unwrap_err();
        assert!(e.to_string().contains("enc.onnx"), "{e}");
        // another hop size is refused
        let t = archive::tests::tar(&[("x/config.ini", cfg.replace("hop_size = 480", "hop_size = 240").as_bytes())]);
        let e = from_archive(&archive::tests::gzip(&t)).unwrap_err();
        assert!(e.to_string().contains("hop_size"), "{e}");
    }

    #[test]
    fn random_weights_have_the_network_shapes() {
        let w = random(1);
        assert_eq!(w.erb_conv1.0.stride, 2);
        assert_eq!(w.erb_conv3.0.stride, 1);
        assert_eq!(w.df_conv1.0.stride, 2);
        assert_eq!(w.convt3.0.stride, 1);
        assert_eq!(w.df_out.out * w.df_out.groups, NB_DF * 2 * DF_ORDER);
        assert_eq!(w.erb_gru[1].w.len(), 3 * HIDDEN * HIDDEN);
    }
}
