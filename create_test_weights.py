import torch
from safetensors.torch import save_file
tensors = {
    "model.embed_tokens.weight": torch.randn(32000, 128),
    "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
    "model.layers.0.self_attn.k_proj.weight": torch.randn(128, 128),
    "model.layers.0.self_attn.v_proj.weight": torch.randn(128, 128),
    "model.layers.0.self_attn.o_proj.weight": torch.randn(128, 128),
    "model.layers.0.mlp.gate_proj.weight": torch.randn(256, 128),
    "model.layers.0.mlp.up_proj.weight": torch.randn(256, 128),
    "model.layers.0.mlp.down_proj.weight": torch.randn(128, 256),
    "model.layers.0.input_layernorm.weight": torch.ones(128),
    "model.layers.0.post_attention_layernorm.weight": torch.ones(128),
    "model.norm.weight": torch.ones(128),
    "lm_head.weight": torch.randn(32000, 128),
}
save_file(tensors, "test_model.safetensors")
print(f"Created test_model.safetensors with {len(tensors)} tensors")
for k, v in tensors.items():
    print(f"  {k}: {list(v.shape)}")