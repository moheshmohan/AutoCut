"""ACE-Step 1.5 turbo reference run (CPU, float32) for FilmCraft's oracle tests.

Runs the official MIT/Apache-2.0 reference (the modeling file that ships with the weights in
huggingface.co/ACE-Step/Ace-Step1.5, transformers' Qwen3Model for the text encoder, diffusers'
AutoencoderOobleck for the VAE) on the text2music path the official pipeline takes without the LM
planner, and writes every intermediate tensor the Rust port compares against.

usage: python -I oracle.py <Ace-Step1.5 dir> <refcode dir> <out.safetensors> [seconds] [seed]
"""

import importlib.util
import json
import sys

import torch
from safetensors.torch import save_file

model_dir, ref_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
seconds = int(sys.argv[4]) if len(sys.argv) > 4 else 24
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 42
torch.set_grad_enabled(False)
torch.set_num_threads(8)


# the reference modules, loaded by path (configuration first: the modeling file imports it)
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cfg_mod = load("configuration_acestep_v15", f"{ref_dir}/configuration_acestep_v15.py")
mdl_mod = load("modeling_acestep_v15_turbo", f"{ref_dir}/modeling_acestep_v15_turbo.py")

from transformers import AutoModel, AutoTokenizer  # noqa: E402
from diffusers.models import AutoencoderOobleck  # noqa: E402

out = {}
meta = {}

# --- prompts, exactly as acestep/core/generation/handler builds them (constants.py) ---
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""
instruction = "Fill the audio semantic mask based on the given conditions:"
caption = "Upbeat electronic dance track with punchy drums, a warm synth bass and bright arpeggios, energetic and uplifting, instrumental"
metas = f"- bpm: 124\n- timesignature: 4\n- keyscale: A minor\n- duration: {seconds} seconds\n"
lyrics = "# Languages\nunknown\n\n# Lyric\n[Instrumental]<|endoftext|>"
text_prompt = SFT_GEN_PROMPT.format(instruction, caption, metas)
meta["text_prompt"] = text_prompt
meta["lyrics_text"] = lyrics
meta["caption"] = caption
meta["seconds"] = seconds
meta["seed"] = seed

tok = AutoTokenizer.from_pretrained(f"{model_dir}/Qwen3-Embedding-0.6B")
t = tok(text_prompt, padding="longest", truncation=True, max_length=256, return_tensors="pt")
l = tok(lyrics, padding="longest", truncation=True, max_length=2048, return_tensors="pt")
meta["text_ids"] = t.input_ids[0].tolist()
meta["lyric_ids"] = l.input_ids[0].tolist()
# extra tokenizer cases for the tokenizer test
cases = [
    "Hello world", "  two  spaces\n\nand lines\r\n", "Café crème brûlée — naïve façade", "数字 123456 and 3.14",
    "It's we'll they've I'M", "emoji 🎵🎶 music", "tabs\tand nbsp", "<|endoftext|>inside<|im_start|>x", "ﬁ ligature Å Å",
    "", " ", "\n", "a" * 50, "Lo-fi hip hop, 85 BPM, mellow piano & vinyl crackle!!!", "x  \n  y   ", "日本語のテキスト、カタカナ。",
    "'S 'T 'Re don't", "100,000 9999999 ١٢٣", "́accent first", "ends with space ",
]
meta["tokenizer_cases"] = [{"text": c, "ids": tok(c).input_ids} for c in cases]
long = ("word " * 400).strip()
meta["tokenizer_truncated"] = {"text": long, "ids": tok(long, truncation=True, max_length=256).input_ids}

# --- text encoder (Qwen3-Embedding-0.6B) and lyric embeddings ---
te = AutoModel.from_pretrained(f"{model_dir}/Qwen3-Embedding-0.6B", dtype=torch.float32).eval()
text_hidden = te(input_ids=t.input_ids).last_hidden_state
lyric_hidden = te.embed_tokens(l.input_ids)
out["text_hidden"] = text_hidden[0].contiguous()
out["lyric_embeds"] = lyric_hidden[0].contiguous()
layer0 = {}
te.layers[0].register_forward_hook(lambda m, i, o: (layer0.setdefault("x", (o[0] if isinstance(o, tuple) else o).detach().clone()), None)[1])
te(input_ids=t.input_ids)
out["text_layer0"] = layer0["x"][0].contiguous()
del te

# --- DiT ---
model = mdl_mod.AceStepConditionGenerationModel.from_pretrained(
    f"{model_dir}/acestep-v15-turbo", dtype=torch.float32, attn_implementation="sdpa"
).eval()
model.config._attn_implementation = "sdpa"
silence = torch.load(f"{model_dir}/acestep-v15-turbo/silence_latent.pt", weights_only=True).transpose(1, 2).float()
meta["silence_shape"] = list(silence.shape)

frames = max(128, seconds * 48000 // 1920)
src = silence[:, :frames, :].clone()
chunk = torch.ones(1, frames, 64)
is_covers = torch.tensor([False])
refer = silence[:, :750, :].clone()
order = torch.tensor([0], dtype=torch.long)
text_mask = t.attention_mask.bool()
lyric_mask = l.attention_mask.bool()

hooks = {}


def grab(name):
    def f(_m, _i, o):
        if name not in hooks:
            o0 = o[0] if isinstance(o, tuple) else o
            hooks[name] = o0.detach().clone()
    return f


enc = model.encoder
enc.lyric_encoder.register_forward_hook(lambda m, i, o: (hooks.setdefault("lyric_encoded", o.last_hidden_state.detach().clone()), None)[1])
enc.timbre_encoder.register_forward_hook(lambda m, i, o: (hooks.setdefault("timbre", o[0].detach().clone()), None)[1])
enc.text_projector.register_forward_hook(grab("text_projected"))

ehs, emask, ctx = model.prepare_condition(
    text_hidden_states=text_hidden, text_attention_mask=text_mask,
    lyric_hidden_states=lyric_hidden, lyric_attention_mask=lyric_mask,
    refer_audio_acoustic_hidden_states_packed=refer, refer_audio_order_mask=order,
    hidden_states=src, attention_mask=torch.ones(1, frames), silence_latent=silence,
    src_latents=src, chunk_masks=chunk, is_covers=is_covers,
)
out["lyric_encoded"] = hooks["lyric_encoded"][0].contiguous()
out["timbre"] = hooks["timbre"][0].contiguous()
out["text_projected"] = hooks["text_projected"][0].contiguous()
out["encoder_hidden"] = ehs[0].contiguous()
out["encoder_mask"] = emask[0].float().contiguous()
out["context"] = ctx[0].contiguous()

gen = torch.Generator(device="cpu").manual_seed(seed)
noise = torch.randn((1, frames, 64), generator=gen, dtype=torch.float32)
out["noise"] = noise[0].contiguous()

dec = model.decoder
dec.proj_in.register_forward_hook(grab("dit_proj_in"))
dec.condition_embedder.register_forward_hook(grab("dit_cond"))
dec.layers[0].register_forward_hook(grab("dit_layer0"))
dec.layers[1].register_forward_hook(grab("dit_layer1"))
dec.layers[-1].register_forward_hook(grab("dit_layer_last"))
dec.time_embed.register_forward_hook(lambda m, i, o: (hooks.setdefault("temb_t", o[0].detach().clone()), None)[1])
dec.time_embed.register_forward_hook(lambda m, i, o: (hooks.setdefault("tproj_t", o[1].detach().clone()), None)[1])

schedule = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3]
xt = noise.clone()
for i, tc in enumerate(schedule):
    tt = torch.full((1,), tc)
    vt = dec(hidden_states=xt, timestep=tt, timestep_r=tt, attention_mask=None,
             encoder_hidden_states=ehs, encoder_attention_mask=emask, context_latents=ctx,
             use_cache=False)[0]
    out[f"xt_{i}"] = xt[0].contiguous()
    out[f"vt_{i}"] = vt[0].contiguous()
    if i == len(schedule) - 1:
        xt = xt - vt * tc
    else:
        xt = xt - vt * (tc - schedule[i + 1])
    print("step", i, flush=True)
out["latents"] = xt[0].contiguous()
for k in ["dit_proj_in", "dit_cond", "dit_layer0", "dit_layer1", "dit_layer_last", "temb_t", "tproj_t"]:
    out[k] = hooks[k][0].contiguous()

# the reference's own sampler must agree with the loop above
ref = model.generate_audio(
    text_hidden_states=text_hidden, text_attention_mask=text_mask,
    lyric_hidden_states=lyric_hidden, lyric_attention_mask=lyric_mask,
    refer_audio_acoustic_hidden_states_packed=refer, refer_audio_order_mask=order,
    src_latents=src, chunk_masks=chunk, is_covers=is_covers, silence_latent=silence,
    seed=seed, shift=3.0, infer_method="ode",
)["target_latents"]
meta["generate_audio_max_diff"] = float((ref - xt).abs().max())
del model

# --- VAE decode ---
vae = AutoencoderOobleck.from_pretrained(f"{model_dir}/vae", torch_dtype=torch.float32).eval()
wav = vae.decode(xt.transpose(1, 2)).sample
out["audio"] = wav[0].contiguous()
meta["audio_peak"] = float(wav.abs().max())
# a short fixed latent decode (the VAE alone)
g2 = torch.Generator().manual_seed(7)
z = torch.randn((1, 64, 40), generator=g2)
out["vae_z"] = z[0].contiguous()
out["vae_out"] = vae.decode(z).sample[0].contiguous()

save_file({k: v.float().contiguous() for k, v in out.items()}, out_path)
with open(out_path.replace(".safetensors", ".json"), "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print("wrote", out_path, {k: list(v.shape) for k, v in out.items()})
print("generate_audio vs loop", meta["generate_audio_max_diff"])
