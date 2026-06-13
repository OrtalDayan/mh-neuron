#!/bin/bash
# Deploy LLaVA_HF wrapper for HF-format LLaVA-1.5 (no llava package needed).
#
# This script makes 4 edits across 4 files, all idempotent:
#   1. Append LLaVA_HF class to vlmeval/vlm/llava/llava.py
#   2. Add LLaVA_HF to vlmeval/vlm/__init__.py exports
#   3. Register 'llava_v1.5_7b_hf' in vlmeval/config.py
#   4. Update _VE_MODEL dict in code/run1_ablation.py
#
# Run from project root (modality_taxonomy/).
# Run after restoring files to clean state.

set -e  # exit on any error

LLAVA_PY="modern_vlms/VLMEvalKit_brv/vlmeval/vlm/llava/llava.py"
INIT_PY="modern_vlms/VLMEvalKit_brv/vlmeval/vlm/__init__.py"
CONFIG_PY="modern_vlms/VLMEvalKit_brv/vlmeval/config.py"
RUN1_PY="code/run1_ablation.py"

# Sanity check: target files exist
for f in "$LLAVA_PY" "$INIT_PY" "$CONFIG_PY" "$RUN1_PY"; do
  if [[ ! -f "$f" ]]; then
    echo "  ✗ ERROR: target file missing: $f"
    exit 1
  fi
done

ts=$(date +%Y%m%d_%H%M%S)
mkdir -p backups
cp "$LLAVA_PY"  "backups/llava.py.${ts}"
cp "$INIT_PY"   "backups/init.py.${ts}"
cp "$CONFIG_PY" "backups/config.py.${ts}"
cp "$RUN1_PY"   "backups/run1_ablation.py.${ts}"
echo "  ✓ Backups saved with timestamp ${ts}"
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Edit 1: Append LLaVA_HF class to llava.py
# ─────────────────────────────────────────────────────────────────────────────
echo "═══ Edit 1: Append LLaVA_HF class to llava.py ═══"
# Use exact-line anchor to avoid matching LLaVA_OneVision_HF as a substring
if grep -qE "^class LLaVA_HF\b" "$LLAVA_PY"; then
  echo "  [skip] class LLaVA_HF already present"
else
  cat >> "$LLAVA_PY" <<'LLAVA_HF_CLASS_EOF'


class LLaVA_HF(BaseModel):
    """HF-format wrapper for LLaVA-1.5 models (e.g., llava-hf/llava-1.5-7b-hf).

    Uses transformers' LlavaProcessor + LlavaForConditionalGeneration directly,
    so it does NOT require the haotian-liu LLaVA package. The HF-format weights
    of llava-hf/llava-1.5-7b-hf are bit-equivalent (verified via torch.equal) to
    the original liuhaotian/llava-v1.5-7b weights, just repackaged in safetensors
    with HF-style key names.
    """
    INSTALL_REQ = False
    INTERLEAVE = False

    DEFAULT_IMAGE_TOKEN = "<image>"

    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf", **kwargs):
        from transformers import AutoProcessor, LlavaForConditionalGeneration
        assert model_path is not None, "Model path must be provided."
        self.model_path = model_path
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to(0)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

        kwargs_default = dict(
            do_sample=False,
            max_new_tokens=int(os.environ.get("VLMEVAL_MAX_NEW_TOKENS", 1024)),
            num_beams=1,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default

    def generate_inner(self, message, dataset=None):
        text_parts, images = [], []
        for msg in message:
            if msg["type"] == "text":
                text_parts.append(msg["value"])
            elif msg["type"] == "image":
                images.append(Image.open(msg["value"]).convert("RGB"))
        text = " ".join(text_parts)

        if images:
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text", "text": text})
        else:
            content = [{"type": "text", "text": text}]
        conversation = [{"role": "user", "content": content}]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )

        inputs = self.processor(
            images=images if images else None,
            text=prompt,
            return_tensors="pt",
        ).to(0, torch.float16)

        output = self.model.generate(**inputs, **self.kwargs)

        input_token_len = inputs["input_ids"].shape[1]
        generated_ids = output[0][input_token_len:]
        return self.processor.decode(generated_ids, skip_special_tokens=True)
LLAVA_HF_CLASS_EOF
  echo "  ✓ Appended LLaVA_HF class to $LLAVA_PY"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Edit 2: Export LLaVA_HF from __init__.py
# ─────────────────────────────────────────────────────────────────────────────
echo "═══ Edit 2: Add LLaVA_HF to __init__.py exports ═══"
# Use exact word-boundary match to avoid false positive on LLaVA_OneVision_HF
if grep -qE "\bLLaVA_HF\b" "$INIT_PY"; then
  echo "  [skip] LLaVA_HF already exported in __init__.py"
else
  # Append ', LLaVA_HF' before the newline at the end of the .llava import line
  sed -i "/^from .llava import.*LLaVA_OneVision_HF$/s/LLaVA_OneVision_HF$/LLaVA_OneVision_HF, LLaVA_HF/" "$INIT_PY"
  echo "  ✓ Updated $INIT_PY"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Edit 3: Register 'llava_v1.5_7b_hf' in config.py
# ─────────────────────────────────────────────────────────────────────────────
echo "═══ Edit 3: Register 'llava_v1.5_7b_hf' in config.py ═══"
if grep -q "llava_v1.5_7b_hf" "$CONFIG_PY"; then
  echo "  [skip] llava_v1.5_7b_hf already registered"
else
  sed -i "/'llava_v1.5_7b': partial(LLaVA, model_path='liuhaotian\/llava-v1.5-7b')/a\\    'llava_v1.5_7b_hf': partial(LLaVA_HF, model_path='llava-hf/llava-1.5-7b-hf')," "$CONFIG_PY"
  echo "  ✓ Added entry to $CONFIG_PY"
fi
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Edit 4: Update run1_ablation.py _VE_MODEL dict
# Use python instead of sed: more reliable for multi-line context.
# ─────────────────────────────────────────────────────────────────────────────
echo "═══ Edit 4: Update _VE_MODEL dict in run1_ablation.py ═══"
python3 <<'PYEOF'
import re

path = "code/run1_ablation.py"
with open(path) as f:
    content = f.read()

# The line we want to change is the SECOND occurrence (line ~611), not the first
# (line 144 which uses tuple format). Match the simple-dict form specifically.
old = "        'llava-liuhaotian': 'llava_v1.5_7b',"
new = "        'llava-liuhaotian': 'llava_v1.5_7b_hf',"

if new in content:
    print("  [skip] _VE_MODEL already updated")
elif old in content:
    content = content.replace(old, new, 1)  # replace only the first occurrence (line 611)
    with open(path, 'w') as f:
        f.write(content)
    print(f"  ✓ Updated _VE_MODEL: '{old.strip()}' → '{new.strip()}'")
else:
    print(f"  ✗ ERROR: target line not found in {path}")
    print("    Looking for: " + repr(old))
PYEOF
echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────
echo "═══ Verification ═══"
echo ""
echo "── 1. LLaVA_HF class in llava.py ──"
grep -nE "^class LLaVA_HF\b" "$LLAVA_PY"
echo ""
echo "── 2. __init__.py imports ──"
grep "from .llava import" "$INIT_PY"
echo ""
echo "── 3. config.py registry ──"
grep -n "llava_v1.5_7b" "$CONFIG_PY"
echo ""
echo "── 4. run1_ablation.py _VE_MODEL ──"
grep -n "'llava-liuhaotian':" "$RUN1_PY"
echo ""
echo "── 5. Python imports cleanly ──"
modern_vlms/VLMEvalKit_brv/.venv/bin/python -c "
import sys
sys.path.insert(0, 'modern_vlms/VLMEvalKit_brv')
from vlmeval.vlm import LLaVA_HF
print('  ✓ from vlmeval.vlm import LLaVA_HF works')
print(f'  Class: {LLaVA_HF}')
"
