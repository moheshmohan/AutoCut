"""Test-only oracle for filmcraft-speech: decode clips with Hugging Face transformers' Whisper the
way `Transcriber::hear` does (one 30 s window, prompt <|startoftranscript|><|lang|><|transcribe|>
<|notimestamps|>, greedy, the model's suppress tokens, no timestamps) and write the tokens, for
the ignored live test `malayalam_hearing_matches_the_reference_oracle` to compare against.

Never shipped; run in a venv under the lane's target directory (torch, transformers, numpy).

usage: whisper_oracle.py <model dir> <language> <out.json> <clip.f32>...
       (clips: mono 16 kHz little-endian float32)
"""
import json
import sys

import numpy as np
import torch
from transformers import WhisperFeatureExtractor, WhisperForConditionalGeneration

model_dir, language, out = sys.argv[1:4]
clips = sys.argv[4:]
cfg = json.load(open(f"{model_dir}/config.json"))
gen = json.load(open(f"{model_dir}/generation_config.json"))
tok = json.load(open(f"{model_dir}/tokenizer.json"))
special = {a["content"]: a["id"] for a in tok["added_tokens"]}
model = WhisperForConditionalGeneration.from_pretrained(model_dir, torch_dtype=torch.float32).eval()
fe = WhisperFeatureExtractor(feature_size=cfg["num_mel_bins"])
prompt = [special["<|startoftranscript|>"], special[f"<|{language}|>"], special["<|transcribe|>"], special["<|notimestamps|>"]]
eot = special["<|endoftext|>"]
ts_begin = special["<|0.00|>"]
# what `hear` suppresses: the model's list and every special token but the end of text
suppress = set(gen.get("suppress_tokens", [])) | {i for i in special.values() if i != eot and i < ts_begin}
begin_suppress = set(gen.get("begin_suppress_tokens", [220, eot]))
room = cfg["max_target_positions"] - len(prompt) - 1
results = {}
for clip in clips:
    audio = np.fromfile(clip, dtype="<f4")
    feats = fe(audio, sampling_rate=16000, return_tensors="pt").input_features
    with torch.no_grad():
        enc = model.model.encoder(feats).last_hidden_state
        seq = []
        ids = list(prompt)
        for _ in range(room):
            out_logits = model(encoder_outputs=(enc,), decoder_input_ids=torch.tensor([ids])).logits[0, -1].clone()
            for s in suppress:
                out_logits[s] = -float("inf")
            out_logits[ts_begin:] = -float("inf")
            if not seq:
                for s in begin_suppress:
                    out_logits[s] = -float("inf")
            nxt = int(torch.argmax(out_logits))
            if nxt == eot:
                break
            seq.append(nxt)
            ids.append(nxt)
    results[clip.rsplit("/", 1)[-1]] = seq
    print(clip, len(seq), file=sys.stderr)
json.dump(results, open(out, "w"), indent=1)
# the texts too, for reading by eye
from tokenizers import Tokenizer  # noqa: E402

t = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
json.dump({k: t.decode(v) for k, v in results.items()}, open(out.replace(".json", ".text.json"), "w"), ensure_ascii=False, indent=1)
