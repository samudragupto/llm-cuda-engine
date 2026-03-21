import torch
from transformers import AutoModelForCausalLM

print("Loading TinyLlama-1.1B... converting to FP16...")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16)
state = model.state_dict()

print("Exporting 2.2GB FP16 weights to model_fp16.bin...")
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