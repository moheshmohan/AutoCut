//! A small, strict reader of ONNX model files (the protobuf `ModelProto`), enough to take the
//! weights and the layer list out of the pinned detector files. No code from the file runs and
//! nothing is evaluated: the networks are written out by hand (`crate::net`) and only ask this
//! reader for each layer's attributes and tensors, checking their shapes against what the code
//! implements.
//!
//! Only what the detectors use is decoded — the graph's nodes (op type, inputs, outputs, integer
//! and float attributes, tensor attributes) and its initializers (float32 and int64 tensors, as
//! `raw_data` or packed values). Every length is checked against what is left of the buffer,
//! counts and tensor sizes are capped ([`Limits`]), and anything malformed is an error, never a
//! panic: the file is untrusted input even though the catalogue pins its SHA-256.

use std::collections::HashMap;

use crate::VisionError;

/// Caps on what a model file may hold.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Largest file read.
    pub max_bytes: usize,
    /// Most graph nodes.
    pub max_nodes: usize,
    /// Most initializers.
    pub max_initializers: usize,
    /// Most elements in one tensor.
    pub max_elements: usize,
    /// Most dimensions of one tensor.
    pub max_dims: usize,
    /// Most inputs, outputs or attributes of one node.
    pub max_node_fields: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self { max_bytes: 64 << 20, max_nodes: 10_000, max_initializers: 10_000, max_elements: 16 << 20, max_dims: 8, max_node_fields: 64 }
    }
}

/// Tensor values.
#[derive(Clone, Debug, PartialEq)]
pub enum Data {
    F32(Vec<f32>),
    I64(Vec<i64>),
}

/// A tensor: its dimensions and values (row-major).
#[derive(Clone, Debug, PartialEq)]
pub struct TensorData {
    pub name: String,
    pub dims: Vec<usize>,
    pub data: Data,
}

impl TensorData {
    pub fn len(&self) -> usize {
        match &self.data {
            Data::F32(v) => v.len(),
            Data::I64(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn f32s(&self) -> Option<&[f32]> {
        match &self.data {
            Data::F32(v) => Some(v),
            Data::I64(_) => None,
        }
    }
}

/// An attribute value.
#[derive(Clone, Debug, PartialEq)]
pub enum Attr {
    Int(i64),
    Float(f32),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
    Str(String),
    Tensor(TensorData),
}

/// One graph node.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Node {
    pub op: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attrs: Vec<(String, Attr)>,
}

impl Node {
    pub fn attr(&self, name: &str) -> Option<&Attr> {
        self.attrs.iter().find(|(n, _)| n == name).map(|(_, a)| a)
    }
    pub fn ints(&self, name: &str) -> Option<&[i64]> {
        match self.attr(name)? {
            Attr::Ints(v) => Some(v),
            _ => None,
        }
    }
    pub fn int(&self, name: &str) -> Option<i64> {
        match self.attr(name)? {
            Attr::Int(v) => Some(*v),
            _ => None,
        }
    }
}

/// The parts of a model the detectors read.
#[derive(Clone, Debug, Default)]
pub struct Model {
    /// The default domain's opset version.
    pub opset: i64,
    pub nodes: Vec<Node>,
    pub initializers: HashMap<String, TensorData>,
}

impl Model {
    /// The nodes of type `op`, in graph order.
    pub fn nodes_of<'a>(&'a self, op: &'a str) -> impl Iterator<Item = &'a Node> + 'a {
        self.nodes.iter().filter(move |n| n.op == op)
    }
}

fn bad(what: impl Into<String>) -> VisionError {
    VisionError::Model(format!("ONNX: {}", what.into()))
}

/// A protobuf reader over a byte slice.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

/// A decoded field.
enum Field<'a> {
    Varint(u64),
    Fixed64,
    Bytes(&'a [u8]),
    Fixed32(u32),
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn done(&self) -> bool {
        self.pos >= self.buf.len()
    }

    fn varint(&mut self) -> Result<u64, VisionError> {
        let mut v = 0u64;
        for shift in (0..64).step_by(7) {
            let b = *self.buf.get(self.pos).ok_or_else(|| bad("truncated varint"))?;
            self.pos += 1;
            v |= u64::from(b & 0x7f) << shift;
            if b & 0x80 == 0 {
                return Ok(v);
            }
        }
        Err(bad("varint longer than 10 bytes"))
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8], VisionError> {
        let end = self.pos.checked_add(n).ok_or_else(|| bad("length overflow"))?;
        let s = self.buf.get(self.pos..end).ok_or_else(|| bad("field runs past the end"))?;
        self.pos = end;
        Ok(s)
    }

    /// The next `(field number, value)`.
    fn field(&mut self) -> Result<(u64, Field<'a>), VisionError> {
        let key = self.varint()?;
        let (num, wire) = (key >> 3, key & 7);
        if num == 0 {
            return Err(bad("field number 0"));
        }
        let f = match wire {
            0 => Field::Varint(self.varint()?),
            1 => {
                // no field the detectors read is a fixed64 (double)
                self.take(8)?;
                Field::Fixed64
            }
            2 => {
                let n = usize::try_from(self.varint()?).map_err(|_| bad("length too large"))?;
                Field::Bytes(self.take(n)?)
            }
            5 => {
                let b = self.take(4)?;
                let mut a = [0u8; 4];
                a.copy_from_slice(b);
                Field::Fixed32(u32::from_le_bytes(a))
            }
            w => return Err(bad(format!("unsupported wire type {w}"))),
        };
        Ok((num, f))
    }
}

fn string(b: &[u8]) -> Result<String, VisionError> {
    std::str::from_utf8(b).map(str::to_string).map_err(|_| bad("a name is not UTF-8"))
}

/// Append the int64 values of a field that may be packed (bytes) or a single varint.
fn push_ints(out: &mut Vec<i64>, f: Field<'_>, cap: usize) -> Result<(), VisionError> {
    match f {
        Field::Varint(v) => out.push(v as i64),
        Field::Bytes(b) => {
            let mut r = Reader::new(b);
            while !r.done() {
                if out.len() >= cap {
                    return Err(bad("too many integers"));
                }
                out.push(r.varint()? as i64);
            }
        }
        _ => return Err(bad("integers with the wrong wire type")),
    }
    if out.len() > cap {
        return Err(bad("too many integers"));
    }
    Ok(())
}

/// Append the float values of a field that may be packed (bytes) or a single fixed32.
fn push_floats(out: &mut Vec<f32>, f: Field<'_>, cap: usize) -> Result<(), VisionError> {
    match f {
        Field::Fixed32(v) => out.push(f32::from_bits(v)),
        Field::Bytes(b) => {
            if b.len() % 4 != 0 {
                return Err(bad("packed floats not a multiple of 4 bytes"));
            }
            if out.len().saturating_add(b.len() / 4) > cap {
                return Err(bad("too many floats"));
            }
            out.extend(b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)));
        }
        _ => return Err(bad("floats with the wrong wire type")),
    }
    if out.len() > cap {
        return Err(bad("too many floats"));
    }
    Ok(())
}

/// `TensorProto` data types this reader takes.
const FLOAT: i64 = 1;
const INT64: i64 = 7;

fn tensor(b: &[u8], lim: &Limits) -> Result<TensorData, VisionError> {
    let mut r = Reader::new(b);
    let (mut dims, mut dtype, mut name) = (Vec::new(), 0i64, String::new());
    let (mut raw, mut floats, mut ints): (Option<&[u8]>, Vec<f32>, Vec<i64>) = (None, Vec::new(), Vec::new());
    while !r.done() {
        let (num, f) = r.field()?;
        match num {
            1 => push_ints(&mut dims, f, lim.max_dims)?,
            2 => match f {
                Field::Varint(v) => dtype = v as i64,
                _ => return Err(bad("data_type with the wrong wire type")),
            },
            4 => push_floats(&mut floats, f, lim.max_elements)?,
            7 => push_ints(&mut ints, f, lim.max_elements)?,
            8 => match f {
                Field::Bytes(s) => name = string(s)?,
                _ => return Err(bad("tensor name with the wrong wire type")),
            },
            9 => match f {
                Field::Bytes(s) => raw = Some(s),
                _ => return Err(bad("raw_data with the wrong wire type")),
            },
            13 | 14 => return Err(bad(format!("tensor `{name}`: external data is not supported"))),
            _ => {}
        }
    }
    let mut shape = Vec::with_capacity(dims.len());
    let mut count = 1usize;
    for d in &dims {
        let d = usize::try_from(*d).map_err(|_| bad(format!("tensor `{name}`: negative dimension")))?;
        count = count.checked_mul(d).ok_or_else(|| bad(format!("tensor `{name}`: too large")))?;
        shape.push(d);
    }
    if count > lim.max_elements {
        return Err(bad(format!("tensor `{name}`: {count} elements is more than {}", lim.max_elements)));
    }
    let data = match dtype {
        FLOAT => {
            let v = match raw {
                Some(raw) => {
                    if raw.len() != count.saturating_mul(4) {
                        return Err(bad(format!("tensor `{name}`: {} bytes for {count} floats", raw.len())));
                    }
                    raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect()
                }
                None => floats,
            };
            if v.len() != count {
                return Err(bad(format!("tensor `{name}`: {} values for {count} elements", v.len())));
            }
            Data::F32(v)
        }
        INT64 => {
            let v = match raw {
                Some(raw) => {
                    if raw.len() != count.saturating_mul(8) {
                        return Err(bad(format!("tensor `{name}`: {} bytes for {count} integers", raw.len())));
                    }
                    raw.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c)).collect()
                }
                None => ints,
            };
            if v.len() != count {
                return Err(bad(format!("tensor `{name}`: {} values for {count} elements", v.len())));
            }
            Data::I64(v)
        }
        t => return Err(bad(format!("tensor `{name}`: data type {t} is not supported"))),
    };
    Ok(TensorData { name, dims: shape, data })
}

fn attribute(b: &[u8], lim: &Limits) -> Result<(String, Attr), VisionError> {
    let mut r = Reader::new(b);
    let mut name = String::new();
    let (mut f, mut i, mut s, mut t): (Option<f32>, Option<i64>, Option<String>, Option<TensorData>) = (None, None, None, None);
    let (mut floats, mut ints) = (Vec::new(), Vec::new());
    let (mut has_floats, mut has_ints) = (false, false);
    let mut kind = 0u64;
    while !r.done() {
        let (num, v) = r.field()?;
        match (num, v) {
            (1, Field::Bytes(b)) => name = string(b)?,
            (2, Field::Fixed32(x)) => f = Some(f32::from_bits(x)),
            (3, Field::Varint(x)) => i = Some(x as i64),
            (4, Field::Bytes(b)) => s = Some(String::from_utf8_lossy(b).into_owned()),
            (5, Field::Bytes(b)) => t = Some(tensor(b, lim)?),
            (7, v) => {
                has_floats = true;
                push_floats(&mut floats, v, lim.max_elements)?;
            }
            (8, v) => {
                has_ints = true;
                push_ints(&mut ints, v, lim.max_elements)?;
            }
            (20, Field::Varint(x)) => kind = x,
            (1 | 2 | 3 | 4 | 5 | 20, _) => return Err(bad("attribute field with the wrong wire type")),
            _ => {}
        }
    }
    // AttributeType: FLOAT 1, INT 2, STRING 3, TENSOR 4, FLOATS 6, INTS 7
    let a = match kind {
        1 => Attr::Float(f.unwrap_or(0.0)),
        2 => Attr::Int(i.unwrap_or(0)),
        3 => Attr::Str(s.unwrap_or_default()),
        4 => Attr::Tensor(t.ok_or_else(|| bad(format!("attribute `{name}`: no tensor")))?),
        6 => Attr::Floats(floats),
        7 => Attr::Ints(ints),
        _ => match (f, i, s, t) {
            (_, _, _, Some(t)) => Attr::Tensor(t),
            (_, _, Some(s), _) => Attr::Str(s),
            (_, Some(i), _, _) => Attr::Int(i),
            (Some(f), _, _, _) => Attr::Float(f),
            _ if has_ints => Attr::Ints(ints),
            _ if has_floats => Attr::Floats(floats),
            _ => Attr::Ints(Vec::new()),
        },
    };
    Ok((name, a))
}

fn node(b: &[u8], lim: &Limits) -> Result<Node, VisionError> {
    let mut r = Reader::new(b);
    let mut n = Node::default();
    while !r.done() {
        let (num, v) = r.field()?;
        let Field::Bytes(b) = v else {
            continue;
        };
        match num {
            1 => n.inputs.push(string(b)?),
            2 => n.outputs.push(string(b)?),
            3 => n.name = string(b)?,
            4 => n.op = string(b)?,
            5 => n.attrs.push(attribute(b, lim)?),
            _ => {}
        }
        if n.inputs.len() > lim.max_node_fields || n.outputs.len() > lim.max_node_fields || n.attrs.len() > lim.max_node_fields {
            return Err(bad("a node has too many inputs, outputs or attributes"));
        }
    }
    Ok(n)
}

fn graph(b: &[u8], lim: &Limits, m: &mut Model) -> Result<(), VisionError> {
    let mut r = Reader::new(b);
    while !r.done() {
        let (num, v) = r.field()?;
        match (num, v) {
            (1, Field::Bytes(b)) => {
                if m.nodes.len() >= lim.max_nodes {
                    return Err(bad("too many nodes"));
                }
                m.nodes.push(node(b, lim)?);
            }
            (5, Field::Bytes(b)) => {
                if m.initializers.len() >= lim.max_initializers {
                    return Err(bad("too many initializers"));
                }
                let t = tensor(b, lim)?;
                m.initializers.insert(t.name.clone(), t);
            }
            (15, _) => return Err(bad("sparse initializers are not supported")),
            _ => {}
        }
    }
    Ok(())
}

/// Read a model from its bytes.
pub fn parse(bytes: &[u8], lim: &Limits) -> Result<Model, VisionError> {
    if bytes.len() > lim.max_bytes {
        return Err(bad(format!("the file is larger than {} bytes", lim.max_bytes)));
    }
    let mut m = Model { opset: 0, ..Model::default() };
    let mut r = Reader::new(bytes);
    let mut graphs = 0;
    while !r.done() {
        let (num, v) = r.field()?;
        match (num, v) {
            (7, Field::Bytes(b)) => {
                graphs += 1;
                if graphs > 1 {
                    return Err(bad("more than one graph"));
                }
                graph(b, lim, &mut m)?;
            }
            (8, Field::Bytes(b)) => {
                // OperatorSetIdProto: domain 1, version 2
                let mut o = Reader::new(b);
                let (mut domain, mut version) = (String::new(), 0i64);
                while !o.done() {
                    match o.field()? {
                        (1, Field::Bytes(d)) => domain = string(d)?,
                        (2, Field::Varint(v)) => version = v as i64,
                        _ => {}
                    }
                }
                if domain.is_empty() || domain == "ai.onnx" {
                    m.opset = version;
                }
            }
            _ => {}
        }
    }
    if graphs == 0 {
        return Err(bad("no graph"));
    }
    Ok(m)
}

/// Read a model file.
pub fn read(path: &std::path::Path, lim: &Limits) -> Result<Model, VisionError> {
    let len = std::fs::metadata(path).map_err(|e| VisionError::Model(format!("{}: {e}", path.display())))?.len();
    if len > lim.max_bytes as u64 {
        return Err(bad(format!("{} is larger than {} bytes", path.display(), lim.max_bytes)));
    }
    let bytes = std::fs::read(path).map_err(|e| VisionError::Model(format!("{}: {e}", path.display())))?;
    parse(&bytes, lim)
}

/// Protobuf writers for tests: build small models by hand.
#[cfg(test)]
pub(crate) mod write {
    pub fn varint(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let b = (v & 0x7f) as u8;
            v >>= 7;
            if v == 0 {
                out.push(b);
                return;
            }
            out.push(b | 0x80);
        }
    }
    pub fn key(out: &mut Vec<u8>, num: u64, wire: u64) {
        varint(out, (num << 3) | wire);
    }
    pub fn bytes(out: &mut Vec<u8>, num: u64, b: &[u8]) {
        key(out, num, 2);
        varint(out, b.len() as u64);
        out.extend_from_slice(b);
    }
    pub fn int(out: &mut Vec<u8>, num: u64, v: i64) {
        key(out, num, 0);
        varint(out, v as u64);
    }
    pub fn float_tensor(name: &str, dims: &[i64], vals: &[f32]) -> Vec<u8> {
        let mut t = Vec::new();
        for d in dims {
            int(&mut t, 1, *d);
        }
        int(&mut t, 2, 1);
        bytes(&mut t, 8, name.as_bytes());
        let raw: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        bytes(&mut t, 9, &raw);
        t
    }
    pub fn ints_attr(name: &str, vals: &[i64]) -> Vec<u8> {
        let mut a = Vec::new();
        bytes(&mut a, 1, name.as_bytes());
        let mut packed = Vec::new();
        for v in vals {
            varint(&mut packed, *v as u64);
        }
        bytes(&mut a, 8, &packed);
        int(&mut a, 20, 7);
        a
    }
    pub fn node(op: &str, inputs: &[&str], outputs: &[&str], attrs: &[Vec<u8>]) -> Vec<u8> {
        let mut n = Vec::new();
        for i in inputs {
            bytes(&mut n, 1, i.as_bytes());
        }
        for o in outputs {
            bytes(&mut n, 2, o.as_bytes());
        }
        bytes(&mut n, 4, op.as_bytes());
        for a in attrs {
            bytes(&mut n, 5, a);
        }
        n
    }
    pub fn model(nodes: &[Vec<u8>], inits: &[Vec<u8>]) -> Vec<u8> {
        let mut g = Vec::new();
        for n in nodes {
            bytes(&mut g, 1, n);
        }
        for t in inits {
            bytes(&mut g, 5, t);
        }
        let mut m = Vec::new();
        int(&mut m, 1, 7);
        bytes(&mut m, 7, &g);
        let mut op = Vec::new();
        int(&mut op, 2, 11);
        bytes(&mut m, 8, &op);
        m
    }
}

#[cfg(test)]
mod tests {
    use super::write::{bytes, float_tensor, int, ints_attr, model, node};
    use super::{Limits, parse};

    fn sample() -> Vec<u8> {
        let w = float_tensor("w", &[2, 1, 3, 3], &(0..18).map(|i| i as f32).collect::<Vec<_>>());
        let b = float_tensor("b", &[2], &[0.5, -0.5]);
        let conv = node("Conv", &["x", "w", "b"], &["y"], &[ints_attr("kernel_shape", &[3, 3]), ints_attr("pads", &[1, 1, 1, 1])]);
        model(&[conv, node("Relu", &["y"], &["z"], &[])], &[w, b])
    }

    #[test]
    fn reads_nodes_attributes_and_initializers() {
        let m = parse(&sample(), &Limits::default()).unwrap();
        assert_eq!(m.opset, 11);
        assert_eq!(m.nodes.len(), 2);
        let c = &m.nodes[0];
        assert_eq!((c.op.as_str(), c.inputs.len(), c.outputs[0].as_str()), ("Conv", 3, "y"));
        assert_eq!(c.ints("kernel_shape"), Some(&[3i64, 3][..]));
        assert_eq!(c.ints("pads"), Some(&[1i64, 1, 1, 1][..]));
        let w = &m.initializers["w"];
        assert_eq!(w.dims, vec![2, 1, 3, 3]);
        assert_eq!(w.f32s().unwrap()[17], 17.0);
        assert_eq!(m.initializers["b"].f32s().unwrap(), &[0.5, -0.5]);
        assert_eq!(m.nodes_of("Relu").count(), 1);
    }

    #[test]
    fn caps_are_enforced() {
        let lim = Limits { max_elements: 10, ..Limits::default() };
        assert!(parse(&sample(), &lim).unwrap_err().to_string().contains("elements"));
        let lim = Limits { max_nodes: 1, ..Limits::default() };
        assert!(parse(&sample(), &lim).is_err());
        let lim = Limits { max_bytes: 10, ..Limits::default() };
        assert!(parse(&sample(), &lim).is_err());
        // a tensor claiming more elements than its data holds
        let t = {
            let mut t = Vec::new();
            int(&mut t, 1, 1_000_000);
            int(&mut t, 2, 1);
            bytes(&mut t, 9, &[0, 0, 0, 0]);
            t
        };
        assert!(parse(&model(&[], &[t]), &Limits::default()).is_err());
        // dimensions whose product overflows
        let t = {
            let mut t = Vec::new();
            for _ in 0..4 {
                int(&mut t, 1, i64::MAX);
            }
            int(&mut t, 2, 1);
            t
        };
        assert!(parse(&model(&[], &[t]), &Limits::default()).is_err());
        assert!(parse(&[], &Limits::default()).unwrap_err().to_string().contains("no graph"));
    }

    /// Truncations and bit flips of a valid file give an error or a model, never a panic.
    #[test]
    fn mutations_never_panic() {
        let good = sample();
        for cut in 0..good.len() {
            let _ = std::panic::catch_unwind(|| parse(&good[..cut], &Limits::default())).expect("no panic on truncation");
        }
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        for _ in 0..4000 {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let mut m = good.clone();
            let i = (seed as usize) % m.len();
            m[i] ^= 1 << ((seed >> 32) % 8);
            if seed.is_multiple_of(3) {
                let j = ((seed >> 16) as usize) % m.len();
                m[j] = (seed >> 40) as u8;
            }
            let _ = std::panic::catch_unwind(|| parse(&m, &Limits::default())).expect("no panic on a corrupt file");
        }
    }
}
