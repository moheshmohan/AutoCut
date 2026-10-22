"""Reference outputs for the speech crate's Silero VAD and ECAPA-TDNN ports (test oracle only,
never shipped).

Silero: the authors' reference recipe for `silero_vad_16k.safetensors` (their
`tinygrad_model.py`, MIT) written with PyTorch's own layers (`silero`), or their ONNX release
(`silero_vad_16k_op15.onnx`) on onnxruntime fed as their Python wrapper feeds it (`silero-onnx`;
a slightly different graph: it agrees on 99 % of the speech decisions). ECAPA: SpeechBrain's own
Fbank, sentence normalisation and ECAPA_TDNN classes with the released checkpoint.

    python -I speech_oracle.py silero <silero_vad_16k.safetensors> <clip.wav> > silero.json
    python -I speech_oracle.py silero-onnx <model.onnx> <clip.wav> > silero-onnx.json
    python -I speech_oracle.py ecapa <embedding_model.ckpt> <clip.wav>... > ecapa.json

Clips are mono 16-bit or float WAV at 16 kHz. Run it in a virtual environment under the lane's
target directory (numpy, onnxruntime, torch, speechbrain); `tests/speech_models.rs` compares.
"""
import json
import sys
import wave

import numpy as np


def read_wav(path):
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, "16 kHz only"
        n = w.getnframes()
        raw = w.readframes(n)
        width = w.getsampwidth()
        ch = w.getnchannels()
    if width == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, dtype="<f4").astype(np.float32)
    else:
        raise SystemExit("16-bit or float WAV only")
    return x.reshape(-1, ch).mean(axis=1)


def silero(path, clip):
    import struct

    import torch

    F = torch.nn.functional
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    w = {}
    for k, v in header.items():
        if k != "__metadata__":
            a, b = v["data_offsets"]
            w[k] = torch.from_numpy(np.frombuffer(data[a:b], dtype="<f4").reshape(v["shape"]).copy())
    cell = torch.nn.LSTMCell(128, 128)
    for k in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        getattr(cell, k).data = w["lstm_cell." + k]
    x = read_wav(clip)
    x = np.concatenate([np.zeros(64, np.float32), x, np.zeros((-len(x)) % 512, np.float32)])
    x = torch.from_numpy(x)[None, :]
    state, probs = None, []
    with torch.no_grad():
        for i in range(64, x.shape[1], 512):
            y = F.pad(x[:, i - 64 : i + 512], (0, 64), mode="reflect").unsqueeze(1)
            y = F.conv1d(y, w["stft_conv.weight"], stride=128)
            y = (y[:, :129, :] ** 2 + y[:, 129:, :] ** 2).sqrt()
            y = F.relu(F.conv1d(y, w["conv1.weight"], w["conv1.bias"], padding=1))
            y = F.relu(F.conv1d(y, w["conv2.weight"], w["conv2.bias"], stride=2, padding=1))
            y = F.relu(F.conv1d(y, w["conv3.weight"], w["conv3.bias"], stride=2, padding=1))
            y = F.relu(F.conv1d(y, w["conv4.weight"], w["conv4.bias"], padding=1)).squeeze(-1)
            state = cell(y, state)
            z = torch.sigmoid(F.conv1d(F.relu(state[0].unsqueeze(-1)), w["final_conv.weight"], w["final_conv.bias"]))
            probs.append(float(z.squeeze(1).mean(axis=1)[0]))
    json.dump({"clip": clip, "probs": probs}, sys.stdout)


def silero_onnx(model, clip):
    import onnxruntime as ort

    s = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    x = read_wav(clip)
    pad = (-len(x)) % 512
    x = np.concatenate([x, np.zeros(pad, np.float32)])
    state = np.zeros((2, 1, 128), np.float32)
    context = np.zeros(64, np.float32)
    probs = []
    for i in range(0, len(x), 512):
        chunk = np.concatenate([context, x[i : i + 512]])
        out, state = s.run(None, {"input": chunk[None, :], "state": state, "sr": np.array(16000, dtype=np.int64)})
        probs.append(float(out[0][0]))
        context = chunk[-64:]
    json.dump({"clip": clip, "probs": probs}, sys.stdout)


def ecapa(ckpt, clips):
    import torch
    from speechbrain.lobes.features import Fbank
    from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN
    from speechbrain.processing.features import InputNormalization

    fb = Fbank(n_mels=80)
    norm = InputNormalization(norm_type="sentence", std_norm=False)
    m = ECAPA_TDNN(80, channels=[1024, 1024, 1024, 1024, 3072], kernel_sizes=[5, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 1], attention_channels=128, lin_neurons=192)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    out = []
    with torch.no_grad():
        for c in clips:
            x = torch.from_numpy(read_wav(c))[None, :]
            f = fb(x)
            f = norm(f, torch.ones(1))
            e = m(f, torch.ones(1))
            out.append({"clip": c, "fbank_first": f[0, :3, :].tolist(), "frames": int(f.shape[1]), "embedding": e[0, 0].tolist()})
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    if sys.argv[1] == "silero":
        silero(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "silero-onnx":
        silero_onnx(sys.argv[2], sys.argv[3])
    else:
        ecapa(sys.argv[2], sys.argv[3:])
