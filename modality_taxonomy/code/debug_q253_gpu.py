#!/usr/bin/env python3
"""Debug: replicate classify script's model loading + baukit hook for layer 29."""
import torch
from transformers import AutoConfig, AutoProcessor

model_path = 'modern_vlms/pretrained/Qwen2.5-VL-3B-Instruct'
device = 'cuda:0'

# Load exactly like load_model_qwen2vl
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
cfg_model_type = getattr(config, 'model_type', '')
print(f'config.model_type: {cfg_model_type}')

if 'qwen2_5' in cfg_model_type:
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print('Loaded with Qwen2_5_VLForConditionalGeneration')

# Check actual layers
import re
layers = set()
for n, _ in model.named_modules():
    m = re.search(r'layers\.(\d+)\.', n)
    if m:
        layers.add(int(m.group(1)))
print(f'Total layers: {len(layers)}, max: {max(layers)}')

# Check layer 27 and 29
for l in [27, 28, 29, 35]:
    name = f'model.language_model.layers.{l}.mlp.down_proj'
    found = False
    for n, _ in model.named_modules():
        if n == name:
            found = True
            break
    print(f'  {name}: {"FOUND" if found else "NOT FOUND"}')

# Try baukit hook on layer 27 (should work) and 29 (fails?)
from baukit import TraceDict
print('\nTesting baukit TraceDict...')

for l in [0, 27, 28, 29, 35]:
    name = f'model.language_model.layers.{l}.mlp.down_proj'
    try:
        with TraceDict(model, [name], retain_input=True) as td:
            pass
        print(f'  Layer {l}: OK')
    except LookupError as e:
        print(f'  Layer {l}: FAILED — {e}')
