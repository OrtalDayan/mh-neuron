"""Test text-only input for liuhaotian and llava-llama3."""
import torch, sys, os
sys.path.insert(0, 'code')
from equal_count_ablation import load_model_and_processor, generate_answer, DUMMY_IMAGE
from PIL import Image
import numpy as np

model_type = sys.argv[1]
device = 'cuda:0'

print(f'\n=== Loading {model_type} ===')
model, processor, image_processor = load_model_and_processor(model_type, device)
model.eval()

image_token_id = None
if hasattr(processor, 'tokenizer'):
    tok = processor.tokenizer
elif hasattr(processor, 'decode'):
    tok = processor
else:
    tok = processor
for name in ['<image>', '<image_placeholder>']:
    ids = tok.encode(name, add_special_tokens=False)
    if len(ids) == 1:
        image_token_id = ids[0]
        break

questions = [
    ('What is the capital of France?', ['paris']),
    ('What is 15 multiplied by 7?', ['105']),
    ('Who wrote Romeo and Juliet?', ['shakespeare', 'william shakespeare']),
    ('What planet is known as the Red Planet?', ['mars']),
    ('What is the chemical symbol for water?', ['h2o']),
]

for method, img in [('White', DUMMY_IMAGE),
                     ('Black', Image.new('RGB', (224,224), (0,0,0))),
                     ('Noise', Image.fromarray(np.random.randint(0,255,(224,224,3),dtype=np.uint8)))]:
    print(f'\n--- {method} image ---')
    for q, aliases in questions:
        prompt = f"Answer briefly.\nQuestion: {q}\nAnswer:"
        ans = generate_answer(model, model_type, processor, image_processor,
                              image_token_id, img, prompt, device, max_new_tokens=30)
        ok = any(a in ans.lower() for a in aliases)
        sym = 'Y' if ok else 'N'
        print(f'  [{sym}] Q: {q}  A: [{ans.strip()[:60]}]')

print('\nDone.')
