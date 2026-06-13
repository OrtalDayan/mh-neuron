#!/bin/bash
# Patch LLaVA_OneVision_HF.generate_inner_image to handle text-only inputs.
# Without this, TriviaQA cells crash because the OneVision processor calls
# image_processor([]) which raises IndexError on make_batched_images.

set -e

LLAVA_PY="modern_vlms/VLMEvalKit_brv/vlmeval/vlm/llava/llava.py"

if [[ ! -f "$LLAVA_PY" ]]; then
  echo "  ✗ ERROR: $LLAVA_PY not found"
  exit 1
fi

# Idempotent skip if already patched
if grep -q "Text-only TriviaQA path" "$LLAVA_PY"; then
  echo "  [skip] LLaVA_OneVision_HF text-only fix already applied"
  exit 0
fi

ts=$(date +%Y%m%d_%H%M%S)
mkdir -p backups
cp "$LLAVA_PY" "backups/llava.py.${ts}"
echo "  ✓ Backup: backups/llava.py.${ts}"

python3 <<'PYEOF'
path = "modern_vlms/VLMEvalKit_brv/vlmeval/vlm/llava/llava.py"
with open(path) as f:
    content = f.read()

# Replace the broken block in LLaVA_OneVision_HF.generate_inner_image.
# The original always includes {"type": "image"} in the conversation and always
# passes images=[] to the multimodal processor, which crashes on TriviaQA.
old = '''        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content.split("\\n", 1)[-1]},
                    {"type": "image"},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(0, torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=100)
        return self.processor.decode(output[0], skip_special_tokens=True)'''

new = '''        # Text-only TriviaQA path: when images is empty, drop the {"type":"image"}
        # entry from the conversation AND bypass the multimodal processor (which
        # unconditionally calls image_processor([]) and crashes with IndexError).
        text_content = content.split("\\n", 1)[-1]
        if images:
            user_content = [
                {"type": "text", "text": text_content},
                {"type": "image"},
            ]
        else:
            user_content = [{"type": "text", "text": text_content}]
        conversation = [{"role": "user", "content": user_content}]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        if images:
            inputs = self.processor(images=images, text=prompt, return_tensors="pt").to(0, torch.float16)
            output = self.model.generate(**inputs, max_new_tokens=100)
            return self.processor.decode(output[0], skip_special_tokens=True)
        else:
            # Tokenize directly (skip image_processor) and generate from the LLM.
            inputs = self.processor.tokenizer(prompt, return_tensors="pt").to(0)
            # Cap max_new_tokens for short-answer benchmarks; default 100 wastes time.
            output = self.model.language_model.generate(
                **inputs, max_new_tokens=50, do_sample=False
            )
            return self.processor.tokenizer.decode(output[0], skip_special_tokens=True)'''

if new.strip().split('\n')[0] in content:
    print("  [skip] already patched (marker line present)")
elif old in content:
    content = content.replace(old, new, 1)
    with open(path, 'w') as f:
        f.write(content)
    print("  ✓ Patched LLaVA_OneVision_HF.generate_inner_image")
else:
    print("  ✗ ERROR: target block not found — file structure may have changed")
    print("  Checking for partial match...")
    if '"type": "image"' in content:
        print("    found {\"type\": \"image\"} — file has the structure")
    if 'inputs = self.processor(images=images, text=prompt' in content:
        print("    found self.processor(images=images, text=prompt) call")
    print("  Manual inspection needed.")
PYEOF

echo ""
echo "── Verify patch ──"
grep -n "Text-only TriviaQA path" "$LLAVA_PY"
echo ""
echo "── Show patched function (first 30 lines) ──"
awk '/def generate_inner_image/{p=1} p{print; if(/return self.processor.tokenizer.decode/){exit}}' "$LLAVA_PY" | head -40

echo ""
echo "── Python import check (BRV venv) ──"
modern_vlms/VLMEvalKit_brv/.venv/bin/python -c "
import sys
sys.path.insert(0, 'modern_vlms/VLMEvalKit_brv')
from vlmeval.vlm.llava.llava import LLaVA_OneVision_HF
print('  ✓ LLaVA_OneVision_HF imports cleanly')
import inspect
src = inspect.getsource(LLaVA_OneVision_HF.generate_inner_image)
if 'Text-only TriviaQA path' in src:
    print('  ✓ patch is in the live class definition')
else:
    print('  ✗ patch not found in live class — restart needed?')
"
