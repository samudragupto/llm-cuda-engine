import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import struct
import os

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Downloading {model_id}...")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Export Tokenizer Vocabulary
print("Exporting tokenizer.bin...")
with open("tokenizer.bin", "wb") as f:
    f.write(struct.pack("I", tokenizer.vocab_size))
    for i in range(tokenizer.vocab_size):
        # HuggingFace tokenizers don't always have every ID mapped linearly. We safely decode.
        text = tokenizer.decode([i])
        text_bytes = text.encode("utf-8")
        f.write(struct.pack("I", len(text_bytes)))
        f.write(text_bytes)

# Export Model Weights
print("Exporting model.bin...")
with open("model.bin", "wb") as f:
    state_dict = model.state_dict()
    
    def write_tensor(name):
        t = state_dict[name].flatten().numpy()
        f.write(t.tobytes())
        print(f"  Exported {name} {list(state_dict[name].shape)}")

    # 1. Token Embeddings
    write_tensor("model.embed_tokens.weight")
    
    # 2. Transformer Blocks (TinyLlama has 22 layers)
    for i in range(model.config.num_hidden_layers):
        write_tensor(f"model.layers.{i}.input_layernorm.weight")
        write_tensor(f"model.layers.{i}.self_attn.q_proj.weight")
        write_tensor(f"model.layers.{i}.self_attn.k_proj.weight")
        write_tensor(f"model.layers.{i}.self_attn.v_proj.weight")
        write_tensor(f"model.layers.{i}.self_attn.o_proj.weight")
        write_tensor(f"model.layers.{i}.post_attention_layernorm.weight")
        write_tensor(f"model.layers.{i}.mlp.gate_proj.weight")
        write_tensor(f"model.layers.{i}.mlp.up_proj.weight")
        write_tensor(f"model.layers.{i}.mlp.down_proj.weight")
        
    # 3. Final Norm and LM Head
    write_tensor("model.norm.weight")
    write_tensor("lm_head.weight")

print("\nExport complete! You now have tokenizer.bin and model.bin")