import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import struct

print("Loading TinyLlama...")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.float16)

def quantize_int8(w):
    scale = w.abs().max(dim=1).values / 127.0
    q = torch.round(w / scale.unsqueeze(1)).to(torch.int8)
    return q, scale.to(torch.float16)

state = model.state_dict()
print("Exporting Mixed Precision Weights (FP16 Attention, INT8 MLP)...")
with open("model_mixed.bin", "wb") as f:
    f.write(state["model.embed_tokens.weight"].numpy().tobytes())
    
    for i in range(22):
        pref = f"model.layers.{i}."
        f.write(state[pref + "input_layernorm.weight"].numpy().tobytes())
        
        # SENSITIVE: Keep Attention in pure FP16
        f.write(state[pref + "self_attn.q_proj.weight"].numpy().tobytes())
        f.write(state[pref + "self_attn.k_proj.weight"].numpy().tobytes())
        f.write(state[pref + "self_attn.v_proj.weight"].numpy().tobytes())
        f.write(state[pref + "self_attn.o_proj.weight"].numpy().tobytes())
        f.write(state[pref + "post_attention_layernorm.weight"].numpy().tobytes())
        
        # MASSIVE: Compress FFN to INT8
        for name in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
            q, s = quantize_int8(state[pref + name + ".weight"])
            f.write(q.numpy().tobytes())
            f.write(s.numpy().tobytes())
            
    f.write(state["model.norm.weight"].numpy().tobytes())
    f.write(state["lm_head.weight"].numpy().tobytes())

print("Done! This is perfectly balanced for accuracy and speed.")