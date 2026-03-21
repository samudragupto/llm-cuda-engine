import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading TinyLlama-1.1B...")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

print("Exporting tokenizer.bin (with fixed spaces)...")
import struct
with open("tokenizer.bin", "wb") as f:
    f.write(struct.pack("I", tokenizer.vocab_size))
    for i in range(tokenizer.vocab_size):
        # Use convert_ids_to_tokens to preserve the raw " " ( U+2581 ) characters!
        text = tokenizer.convert_ids_to_tokens(i)
        if text is None: text = ""
        # Handle special byte fallback tokens like <0x0A>
        if text.startswith("<0x") and text.endswith(">"):
            pass # Keep it as <0x0A> so C++ can parse it
        else:
            text = text.replace(" ", " ") # Replace sentencepiece block with space
            
        text_bytes = text.encode("utf-8")
        f.write(struct.pack("I", len(text_bytes)))
        f.write(text_bytes)

print("Exporting 2.2GB FP16 weights to model_fp16.bin...")
state = model.state_dict()
with open("model_fp16.bin", "wb") as f:
    f.write(state["model.embed_tokens.weight"].numpy().tobytes())
    for i in range(22):
        prefix = f"model.layers.{i}."
        f.write(state[prefix + "input_layernorm.weight"].numpy().tobytes())
        f.write(state[prefix + "self_attn.q_proj.weight"].numpy().tobytes())
        f.write(state[prefix + "self_attn.k_proj.weight"].numpy().tobytes())
        f.write(state[prefix + "self_attn.v_proj.weight"].numpy().tobytes())
        f.write(state[prefix + "self_attn.o_proj.weight"].numpy().tobytes())
        f.write(state[prefix + "post_attention_layernorm.weight"].numpy().tobytes())
        f.write(state[prefix + "mlp.gate_proj.weight"].numpy().tobytes())
        f.write(state[prefix + "mlp.up_proj.weight"].numpy().tobytes())
        f.write(state[prefix + "mlp.down_proj.weight"].numpy().tobytes())
    f.write(state["model.norm.weight"].numpy().tobytes())
    f.write(state["lm_head.weight"].numpy().tobytes())
print("Done!")