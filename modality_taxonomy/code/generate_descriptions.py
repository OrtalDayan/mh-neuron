"""
generate_descriptions_both_backbone.py — Step 0 of Xu et al. classification

Generate detailed image descriptions using LLaVA-1.5-7B for all images
in the detail_23k subset. These descriptions serve as the text tokens
for neuron classification.

Supports five backends:
  --model_type llava-hf         : HuggingFace llava-hf/llava-1.5-7b-hf
  --model_type llava-liuhaotian : Original liuhaotian/llava-v1.5-7b via LLaVA repo
                            (requires transformers==4.37.2)
  --model_type llava-ov         : HuggingFace llava-hf/llava-onevision-qwen2-7b-ov-hf
                            (requires transformers>=4.45)
  --model_type internvl   : InternVL2.5 (requires transformers>=4.49, trust_remote_code)
  --model_type qwen2vl    : Qwen2.5-VL (requires transformers>=4.49, qwen-vl-utils)

Output: generated_descriptions.json — {image_id: {"token_ids": [...], "tokens": [...], "text": "..."}}

Usage:
    # Full dataset with HF backend
    python generate_descriptions_both_backbone.py --model_type llava-hf --start_idx 0 --end_idx 23000

    # Full dataset with liuhaotian backend
    python generate_descriptions_both_backbone.py --model_type llava-liuhaotian --start_idx 0 --end_idx 23000

    # Test on Figure 3 images (6 images, shows token/word format + stage 1 input layout)
    python generate_descriptions_both_backbone.py --model_type llava-hf --test_fig3
    python generate_descriptions_both_backbone.py --model_type llava-liuhaotian --test_fig3
"""

import argparse                                                           # command-line argument parsing
import json                                                               # reading/writing JSON files
import os                                                                 # file path manipulation
import sys                                                                # modifying the Python import path
import time                                                               # timing the generation loop

# Add LLaVA repo to path so we can import from llava.* (needed for liuhaotian backend)
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # two levels up from this script
_LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')                       # path to cloned LLaVA repo
if _LLAVA_PATH not in sys.path:                                           # avoid duplicate entries
    sys.path.insert(0, _LLAVA_PATH)                                       # insert at front so llava imports resolve here first

import torch                                                              # PyTorch — tensor operations, GPU, inference
from PIL import Image                                                     # Pillow — loading image files
from tqdm import tqdm                                                     # progress bar for the generation loop


def parse_args():
    """Parse command-line arguments for dataset paths, generation settings, and sharding."""
    p = argparse.ArgumentParser(
        description='Generate LLaVA descriptions for detail_23k images (both backends)')

    # Backend selection
    p.add_argument('--model_type', default='llava-hf',
                   choices=['llava-hf', 'llava-liuhaotian', 'internvl', 'qwen2vl', 'llava-ov'],
                   help='"llava-hf" | "llava-liuhaotian" | "internvl" | "qwen2vl" | "llava-ov"')   # which backend to use

    # Model paths
    p.add_argument('--hf_id', default='llava-hf/llava-1.5-7b-hf',
                   help='HuggingFace model ID (for --model_type llava-hf)')     # HF Hub ID
    p.add_argument('--original_model_path', default='liuhaotian/llava-v1.5-7b',
                   help='Original LLaVA model path (for --model_type llava-liuhaotian)')  # HF Hub ID or local path
    p.add_argument('--model_path', default=None,
                   help='Local model directory (for --model_type internvl, qwen2vl, or llava-ov)')  # path to downloaded weights

    # Data paths
    _script_dir = os.path.dirname(os.path.abspath(__file__))              # directory containing this script
    _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))  # project root (two levels up)
    p.add_argument('--detail_23k_path',
                   default=os.path.join(_project_root, 'detail_23k.json'),
                   help='Path to detail_23k.json (defines which images to use)')
    p.add_argument('--coco_img_dir',
                   default='/home/projects/bagon/shared/coco2017/images/train2017/',
                   help='Path to COCO train2017 images')

    # Output
    p.add_argument('--output_path',
                   default='1-describe/generated_descriptions.json',
                   help='Output JSON path')

    # Sharding — allows splitting across multiple GPUs/jobs
    p.add_argument('--start_idx', type=int, default=0,
                   help='Start index in detail_23k list')
    p.add_argument('--end_idx', type=int, default=None,
                   help='End index in detail_23k list (None = all)')

    # Generation parameters
    p.add_argument('--min_new_tokens', type=int, default=100,
                   help='Min tokens to generate per image')               # forces minimum description length
    p.add_argument('--max_new_tokens', type=int, default=550,
                   help='Max tokens to generate per image')               # caps maximum description length
    p.add_argument('--device', default='0')                               # CUDA GPU index

    # Test mode
    p.add_argument('--test_fig3', action='store_true',
                   help='Run on 6 Figure 3 images only. Shows token/word format '
                        'and stage 1 teacher-forcing input layout.')

    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════
# Model loading
# ═══════════════════════════════════════════════════════════════════

def load_model_hf(hf_id, device):
    """Load the HuggingFace LLaVA model.

    Uses LlavaForConditionalGeneration from transformers — the HF port
    of LLaVA-1.5 that bundles vision encoder + LLM in one model class.

    Returns: (model, processor)
        - model: LlavaForConditionalGeneration (vision + LLM)
        - processor: AutoProcessor (handles both text tokenisation + image preprocessing)
    """
    from transformers import AutoProcessor, LlavaForConditionalGeneration  # HF classes for LLaVA

    processor = AutoProcessor.from_pretrained(hf_id)                      # loads tokenizer + CLIP image processor
    model = LlavaForConditionalGeneration.from_pretrained(
        hf_id, torch_dtype=torch.float16, low_cpu_mem_usage=True          # float16 to save GPU memory
    ).to(device).eval()                                                   # move to GPU, set eval mode

    return model, processor


def load_model_original(model_path, device):
    """Load the original LLaVA model via the cloned LLaVA repo.

    Requires the LLaVA repo on sys.path and transformers==4.37.2.

    Returns: (model, tokenizer, image_processor)
        - model: the LLaVA model (LLaMA + CLIP vision encoder + projection)
        - tokenizer: the LLaMA SentencePiece tokenizer
        - image_processor: CLIP image processor (resizes/normalises images)
    """
    from llava.model.builder import load_pretrained_model                 # loads model weights from HF Hub or local
    from llava.mm_utils import get_model_name_from_path                   # derives model name string from path

    model_name = get_model_name_from_path(model_path)                     # e.g. "llava-v1.5-7b"
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map=device,                  # None = no model_base (not a LoRA model)
        torch_dtype=torch.float16                                         # half precision to save GPU memory
    )
    model.eval()                                                          # set to eval mode (disables dropout etc.)

    return model, tokenizer, image_processor


def load_model_internvl(model_path, device):
    """Load InternVL2.5 from a local directory.

    Uses AutoTokenizer + AutoModel with trust_remote_code=True (InternVL ships
    its own modelling code inside the weights folder).  Weights are loaded in
    bfloat16, which is the precision InternVL was trained in.

    Workarounds for transformers ≥4.49 + accelerate:
      1. torch.linspace monkey-patch: InternVL's vision encoder calls
         torch.linspace(...).item() during __init__.  accelerate creates tensors
         on a meta device, where .item() is illegal.  We force CPU creation.
      2. all_tied_weights_keys patch: newer transformers expects this attribute
         on the model class, but InternVL's custom code only defines
         _tied_weights_keys (the old name).  We patch the method that reads it.

    Returns: (model, tokenizer)
        - model   : InternVLChatModel (vision encoder + LLM)
        - tokenizer : InternLM2 / Qwen2 tokenizer bundled with the model
    """
    from transformers import AutoTokenizer, AutoModel                      # standard HF AutoClasses
    import transformers.modeling_utils as _mu                              # access PreTrainedModel internals

    tokenizer = AutoTokenizer.from_pretrained(                             # load bundled tokenizer from local dir
        model_path, trust_remote_code=True)

    # ── Patch 1: force torch.linspace onto CPU to avoid meta-tensor .item() crash ──
    _orig_linspace = torch.linspace                                        # save original function reference
    def _cpu_linspace(*args, **kwargs):                                    # replacement that forces CPU
        kwargs['device'] = 'cpu'                                           # override device to CPU always
        return _orig_linspace(*args, **kwargs)                             # call original with CPU device
    torch.linspace = _cpu_linspace                                         # install the patch

    # ── Patch 2: add missing all_tied_weights_keys if needed ─────────────
    _orig_adjust = getattr(_mu.PreTrainedModel,                            # save original method (may not exist in all versions)
                           '_adjust_tied_keys_with_tied_pointers', None)
    if _orig_adjust is not None:                                           # only patch if the method exists
        def _safe_adjust(self, *args, **kwargs):                           # wrapper that adds the missing attribute
            if not hasattr(self, 'all_tied_weights_keys'):                 # InternVL model lacks this new attribute
                self.all_tied_weights_keys = {}                         # create it as empty set
            return _orig_adjust(self, *args, **kwargs)                     # call original method
        _mu.PreTrainedModel._adjust_tied_keys_with_tied_pointers = _safe_adjust

    try:
        model = AutoModel.from_pretrained(                                 # load full InternVL model
            model_path,
            torch_dtype=torch.bfloat16,                                    # bfloat16 — InternVL's native dtype
            trust_remote_code=True,                                        # required: model code lives in the weights folder
        ).to(device).eval()                                                # move to GPU, disable dropout
    finally:
        torch.linspace = _orig_linspace                                    # always restore original linspace
        if _orig_adjust is not None:                                       # restore original method
            _mu.PreTrainedModel._adjust_tied_keys_with_tied_pointers = _orig_adjust

    # ── Patch 3: inject GenerationMixin into language model ──────────────
    # transformers >=4.50 removed GenerationMixin from PreTrainedModel.
    # InternVL's InternLM2ForCausalLM doesn't inherit it directly, so
    # .generate() is missing.  We mix it in dynamically after loading.
    from transformers import GenerationMixin
    lm = model.language_model                                              # InternLM2ForCausalLM inside InternVL
    if not isinstance(lm, GenerationMixin):
        lm.__class__ = type(lm.__class__.__name__, (lm.__class__, GenerationMixin), {})
        from transformers import GenerationConfig
        lm.generation_config = GenerationConfig()

    # ── Patch 4: fix DynamicCache incompatibility in prepare_inputs_for_generation ──
    # transformers >=4.49 passes DynamicCache instead of tuple-based KV cache.
    # InternLM2's custom code does `past_key_values[0][0].shape[2]` which
    # returns None with DynamicCache, causing AttributeError.
    # Additionally, newer transformers' generate() uses inspect.signature()
    # to check if prepare_inputs_for_generation explicitly accepts
    # inputs_embeds — **kwargs is not enough; the parameter must be named.
    # We monkey-patch the method to handle both issues.
    _orig_prepare = lm.prepare_inputs_for_generation                       # save original method

    def _patched_prepare(input_ids, past_key_values=None,                  # wrapper with explicit params
                         inputs_embeds=None, **kwargs):                    # inputs_embeds must be named (inspect check)
        from transformers.cache_utils import DynamicCache                   # DynamicCache class from transformers
        if isinstance(past_key_values, DynamicCache):                      # new-style cache object
            past_length = past_key_values.get_seq_length()                 # use the proper API method
            # After the first forward pass, use input_ids (generated tokens)
            if past_length > 0:                                            # not the first forward pass
                inputs_embeds = None                                       # drop embeds — use ids for subsequent steps
                input_ids = input_ids[:, past_length:]                     # keep only new tokens
            return {                                                       # return minimal inputs dict
                'input_ids': input_ids if inputs_embeds is None else None, # use ids unless first pass with embeds
                'inputs_embeds': inputs_embeds,                            # pass through on first forward pass
                'past_key_values': past_key_values,                        # pass DynamicCache through
                'use_cache': True,                                         # keep caching enabled
                'attention_mask': kwargs.get('attention_mask'),             # pass attention mask if provided
            }
        return _orig_prepare(input_ids, past_key_values=past_key_values,
                             inputs_embeds=inputs_embeds, **kwargs)

    lm.prepare_inputs_for_generation = _patched_prepare                    # install the patched method

    return model, tokenizer


def load_model_qwen2vl(model_path, device):
    """Load Qwen2.5-VL from a local directory.

    Uses the dedicated Qwen2_5_VLForConditionalGeneration class which handles
    dynamic-resolution vision inputs.  AutoProcessor bundles both the text
    tokenizer and the image processor.

    Returns: (model, processor)
        - model     : Qwen2_5_VLForConditionalGeneration
        - processor : AutoProcessor (text + dynamic-resolution image preprocessing)
    """
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # Qwen2.5-VL specific classes

    processor = AutoProcessor.from_pretrained(model_path)                  # loads tokenizer + image processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(            # load Qwen2.5-VL model
        model_path,
        torch_dtype=torch.bfloat16,                                        # bfloat16 — Qwen's native dtype
    ).to(device).eval()                                                    # move to GPU, disable dropout

    return model, processor


def load_model_llava_ov(model_path, device):
    """Load LLaVA-OneVision from HuggingFace.

    LLaVA-OneVision uses a Qwen2 language backbone + SigLIP vision encoder
    with a 2-layer MLP projector.  The HF port uses the standard
    LlavaOnevisionForConditionalGeneration class.

    Args:
        model_path: HF model ID (e.g. 'llava-hf/llava-onevision-qwen2-7b-ov-hf')
                    or local directory path.
        device:     torch device string (e.g. 'cuda:0').

    Returns: (model, processor)
        - model     : LlavaOnevisionForConditionalGeneration
        - processor : AutoProcessor (text tokenizer + SigLIP image processor)
    """
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path)                  # loads tokenizer + image processor
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(        # load LLaVA-OneVision model
        model_path,
        torch_dtype=torch.bfloat16,                                        # bfloat16 — Qwen2's native dtype
        low_cpu_mem_usage=True,                                            # reduce peak RAM during loading
    ).to(device).eval()                                                    # move to GPU, disable dropout

    return model, processor

def build_prompt_hf():
    """Build generation prompt for the HF backend.

    HF LLaVA was ported without the system prompt, so the prompt is:
        USER: <image>\\nCould you describe the image?\\nASSISTANT:

    Returns the formatted prompt string.
    """
    return "USER: <image>\nCould you describe the image?\nASSISTANT:"    # no system prompt, \\n separators


def build_prompt_original():
    """Build generation prompt for the liuhaotian backend using conv_templates["v1"].

    The v1 template (conv_vicuna_v1) formats the conversation as:
        {system_prompt} USER: <image>\\nCould you describe the image? ASSISTANT:

    Returns the formatted prompt string.
    """
    from llava.conversation import conv_templates                         # conversation template registry

    conv = conv_templates["v1"].copy()                                    # copy to avoid mutating the global template
    conv.append_message(conv.roles[0], "<image>\nCould you describe the image?")  # USER turn with <image> placeholder
    conv.append_message(conv.roles[1], None)                              # empty ASSISTANT turn (model will generate)
    return conv.get_prompt()                                              # serialise into formatted string


# ═══════════════════════════════════════════════════════════════════
# Generation
# ═══════════════════════════════════════════════════════════════════

def generate_description_hf(model, processor, prompt, img, device, max_new_tokens, min_new_tokens):
    """Generate a text description for a single image using the HF LLaVA model.

    Steps:
      1. Use processor to tokenise prompt + preprocess image together
      2. Run greedy decoding with model.generate()
      3. Extract only the NEW tokens (strip the prompt tokens)
      4. Decode into three formats

    Returns dict with three keys:
        token_ids    : list[int]   — raw token IDs of generated text only
        tokens       : list[str]   — subword strings like ["▁d","ough","n","uts"]
        text         : str         — human-readable "doughnuts..."
    """
    # Process text + image together via the HF processor
    inputs = processor(                                                   # AutoProcessor handles both modalities
        images=img, text=prompt, return_tensors='pt'                      # returns dict with input_ids, pixel_values, etc.
    ).to(device)                                                          # move all tensors to GPU

    prompt_len = inputs['input_ids'].shape[1]                             # number of tokens in the prompt (including expanded <image>)

    with torch.no_grad():                                                 # disable gradient computation (inference only)
        output_ids = model.generate(
            **inputs,                                                     # unpack input_ids, pixel_values, attention_mask
            max_new_tokens=max_new_tokens,                                # maximum number of tokens to generate
            min_new_tokens=min_new_tokens,                                # force minimum description length
            do_sample=False,                                              # greedy decoding (always pick highest prob token)
        )

    # HF generate() returns full sequence (prompt + generated), so strip prompt
    generated_ids = output_ids[0, prompt_len:]                            # slice off prompt tokens → only new tokens

    # Strip EOS token if present at the end
    tokenizer = processor.tokenizer                                       # access the underlying tokenizer
    if len(generated_ids) > 0 and generated_ids[-1].item() == tokenizer.eos_token_id:  # check last token
        generated_ids = generated_ids[:-1]                                # remove EOS

    # Three output formats:
    token_ids = generated_ids.tolist()                                    # convert tensor to plain Python list of ints
    tokens = [tokenizer.decode(tid) for tid in token_ids]                 # decode each token ID individually → subword strings

    # Build readable text
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()  # decode all IDs into one string

    # Strip common prefixes (model sometimes starts with a polite preamble)
    for prefix in ['Yes, I can describe the image.', 'Yes, I can describe the image',
                    'Yes,', 'Yes.']:
        if text.startswith(prefix):                                       # check if text starts with this prefix
            text = text[len(prefix):].strip()                             # remove prefix and trim whitespace
            break                                                         # only strip the first matching prefix

    # Strip matching prefix from token lists to keep them aligned with the cleaned text
    clean_tokens = [t.lstrip('▁').strip() for t in tokens]                # strip ▁ prefix for matching
    PREFIX_TOKENS = [                                                     # known prefix patterns
        ['Yes', ',', 'I', 'can', 'describe', 'the', 'image', '.'],       # "Yes, I can describe the image."
        ['Yes', ',', 'I', 'can', 'see', 'the', 'image', '.'],            # "Yes, I can see the image."
        ['Yes', ','],                                                     # "Yes,"
        ['Yes', '.'],                                                     # "Yes."
    ]
    for prefix_toks in PREFIX_TOKENS:
        n = len(prefix_toks)                                              # number of tokens in this prefix pattern
        if clean_tokens[:n] == prefix_toks:                               # match against clean_tokens (no ▁)
            token_ids = token_ids[n:]                                     # strip same count from all three lists
            tokens = tokens[n:]
            clean_tokens = clean_tokens[n:]
            break                                                         # only strip one prefix

    return {"token_ids": token_ids, "tokens": tokens, "text": text}


def generate_description_original(model, tokenizer, image_processor, prompt, img, device, max_new_tokens, min_new_tokens):
    """Generate a text description for a single image using the original LLaVA model.

    Steps:
      1. Tokenise the prompt, replacing <image> with IMAGE_TOKEN_INDEX (-200)
      2. Preprocess the image using CLIP's image processor (resize to 336x336, normalise)
      3. Run greedy decoding with model.generate()
      4. Decode the generated token IDs into three formats

    Returns dict with three keys:
        token_ids    : list[int]   — raw token IDs (used in step 2 classification)
        tokens       : list[str]   — subword strings like ["▁d","ough","n","uts"]
        text         : str         — human-readable "doughnuts..."
    """
    from llava.constants import IMAGE_TOKEN_INDEX                         # special token ID (-200) for <image>
    from llava.mm_utils import tokenizer_image_token                      # tokenise text, replacing <image> with special ID

    # Tokenise the prompt, replacing <image> text with IMAGE_TOKEN_INDEX (-200)
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'         # returns a PyTorch tensor
    ).unsqueeze(0).to(device)                                             # add batch dim → shape (1, seq_len), move to GPU

    # Preprocess image using CLIP processor (resize to 336x336, normalise pixel values)
    image_tensor = image_processor.preprocess(
        img, return_tensors='pt'                                          # returns dict with 'pixel_values' tensor
    )['pixel_values'].half().to(device)                                   # convert to float16, move to GPU → (1, 3, 336, 336)

    with torch.no_grad():                                                 # disable gradient computation (inference only)
        output_ids = model.generate(
            input_ids,                                                    # tokenised prompt with IMAGE_TOKEN_INDEX
            images=image_tensor,                                          # preprocessed image passed as keyword arg
            max_new_tokens=max_new_tokens,                                # maximum number of tokens to generate
            min_new_tokens=min_new_tokens,                                # force minimum description length
            do_sample=False,                                              # greedy decoding (always pick highest prob token)
        )

    # transformers 4.37.2: generate() returns full sequence (prompt + generated)
    generated_ids = output_ids[0]                                         # generate() returns only new tokens for original LLaVA model

    # Three output formats:
    token_ids = generated_ids.tolist()                                    # convert tensor to plain Python list of ints
    tokens = [tokenizer.decode(tid) for tid in token_ids]                 # decode each token ID individually → subword strings

    # Build readable text: tokenizer.decode() works correctly in transformers 4.37.2
    text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()  # decode all IDs into one string, remove <s>/</s>

    # Strip common prefixes from v1 conv template (model sometimes starts with a polite preamble)
    for prefix in ['Yes, I can describe the image.', 'Yes, I can describe the image',
                    'Yes,', 'Yes.']:
        if text.startswith(prefix):                                       # check if text starts with this prefix
            text = text[len(prefix):].strip()                             # remove prefix and trim whitespace
            break                                                         # only strip the first matching prefix

    # Strip matching prefix from token lists to keep them aligned with the cleaned text
    clean_tokens = [t.lstrip('▁').strip() for t in tokens]                # strip ▁ prefix for matching
    PREFIX_TOKENS = [                                                     # known v1 template prefix patterns
        ['Yes', ',', 'I', 'can', 'describe', 'the', 'image', '.'],       # "Yes, I can describe the image."
        ['Yes', ',', 'I', 'can', 'see', 'the', 'image', '.'],            # "Yes, I can see the image."
        ['Yes', ','],                                                     # "Yes,"
        ['Yes', '.'],                                                     # "Yes."
    ]
    for prefix_toks in PREFIX_TOKENS:
        n = len(prefix_toks)                                              # number of tokens in this prefix pattern
        if clean_tokens[:n] == prefix_toks:                               # match against clean_tokens (no ▁)
            token_ids = token_ids[n:]                                     # strip same count from all three lists
            tokens = tokens[n:]
            clean_tokens = clean_tokens[n:]
            break                                                         # only strip one prefix

    return {"token_ids": token_ids, "tokens": tokens, "text": text}


def generate_description_internvl(model, tokenizer, img, device, max_new_tokens, min_new_tokens):
    """Generate a text description for a single image using InternVL2.5.

    InternVL uses its own model.chat() high-level API.  The model handles
    image preprocessing internally — we pass a pixel_values tensor that we
    build ourselves using InternVL's required 448×448 / ImageNet normalisation.

    Steps:
      1. Resize image to 448×448, apply ImageNet normalisation → pixel_values
      2. Call model.chat() with a simple question string
      3. Tokenize the returned text to recover token_ids / tokens

    Returns dict with three keys:
        token_ids : list[int]  — token IDs of the generated description
        tokens    : list[str]  — subword strings
        text      : str        — human-readable description
    """
    import torchvision.transforms as T                                     # torchvision for image transforms
    from torchvision.transforms.functional import InterpolationMode        # BICUBIC interpolation constant

    # Build the preprocessing pipeline that InternVL expects
    _MEAN = (0.485, 0.456, 0.406)                                          # ImageNet channel means (RGB)
    _STD  = (0.229, 0.224, 0.225)                                          # ImageNet channel stds  (RGB)
    transform = T.Compose([                                                # compose sequential transforms
        T.Resize((448, 448),                                               # resize to InternVL's native resolution
                 interpolation=InterpolationMode.BICUBIC),                 # BICUBIC gives sharper edges than bilinear
        T.ToTensor(),                                                      # PIL → float32 tensor [0,1], shape (3,H,W)
        T.Normalize(mean=_MEAN, std=_STD),                                 # normalise to ImageNet distribution
    ])

    pixel_values = transform(img).unsqueeze(0)                             # add batch dim → (1, 3, 448, 448)
    pixel_values = pixel_values.to(dtype=torch.bfloat16, device=device)   # match model dtype + device

    generation_config = dict(                                              # dict passed to model.chat()
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=False,                                                   # greedy decoding
    )

    with torch.no_grad():                                                  # inference only — no gradients
        text = model.chat(                                                 # returns decoded string directly
            tokenizer,
            pixel_values,
            '<image>\nCould you describe the image?',                      # <image> token is replaced by visual features
            generation_config,
        )

    # model.chat() returns a plain string; tokenize it to get token_ids / tokens
    enc = tokenizer(text, add_special_tokens=False)                        # tokenise without BOS/EOS
    token_ids = enc['input_ids']                                           # list[int]
    tokens    = [tokenizer.decode(tid) for tid in token_ids]               # individual subword strings

    return {"token_ids": token_ids, "tokens": tokens, "text": text.strip()}


def generate_description_qwen2vl(model, processor, img, device, max_new_tokens, min_new_tokens):
    """Generate a text description for a single image using Qwen2.5-VL.

    Qwen2.5-VL uses the standard HF generate() interface, but image and text
    must be packaged together through process_vision_info() from qwen-vl-utils,
    which handles dynamic resolution tiling.

    Steps:
      1. Build a messages list in the Qwen chat format (role/content dicts)
      2. Apply the chat template to get the formatted text string
      3. Run process_vision_info() to get preprocessed image tensors
      4. Run processor() to combine text tokens + image tensors
      5. Call model.generate(), strip prompt tokens, decode

    Returns dict with three keys:
        token_ids : list[int]  — token IDs of the generated description only
        tokens    : list[str]  — subword strings
        text      : str        — human-readable description
    """
    from qwen_vl_utils import process_vision_info                          # Qwen helper: converts PIL images → tensors

    messages = [{                                                          # Qwen chat format: list of role/content dicts
        "role": "user",
        "content": [
            {"type": "image", "image": img},                               # PIL Image passed directly
            {"type": "text",  "text": "Could you describe the image?"},    # question text
        ],
    }]

    # Apply chat template — returns a formatted string with special tokens
    text = processor.apply_chat_template(                                  # converts messages list to model input string
        messages, tokenize=False, add_generation_prompt=True)             # add_generation_prompt appends the assistant prefix

    image_inputs, _ = process_vision_info(messages)                       # extract + preprocess images; ignore video (None)

    inputs = processor(                                                    # processor handles both text tokenisation + image tiling
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors='pt',
    ).to(device)                                                           # move all tensors to GPU

    prompt_len = inputs.input_ids.shape[1]                                 # number of tokens in prompt (text + image patches)

    with torch.no_grad():                                                  # inference only
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,                                               # greedy decoding
        )

    generated_ids = output_ids[0, prompt_len:]                            # strip prompt tokens → only new tokens

    # Strip EOS if present
    tokenizer = processor.tokenizer                                        # access underlying tokenizer
    if len(generated_ids) > 0 and generated_ids[-1].item() == tokenizer.eos_token_id:
        generated_ids = generated_ids[:-1]                                 # remove EOS token

    token_ids = generated_ids.tolist()                                     # tensor → list[int]
    tokens    = [tokenizer.decode(tid) for tid in token_ids]               # decode each ID individually → subword strings
    text      = tokenizer.decode(generated_ids,
                                 skip_special_tokens=True).strip()         # full readable string

    return {"token_ids": token_ids, "tokens": tokens, "text": text}


def generate_description_llava_ov(model, processor, img, device, max_new_tokens, min_new_tokens):
    """Generate a text description for a single image using LLaVA-OneVision (HF).

    LLaVA-OneVision uses the Qwen2 chat template via processor.apply_chat_template().
    The processor handles SigLIP image preprocessing and anyres-9 tiling internally.

    Steps:
      1. Build a messages list in the Qwen2 chat format
      2. Apply the chat template to get the formatted text string
      3. Run processor() to combine text tokens + image tensors
      4. Call model.generate(), strip prompt tokens, decode

    Returns dict with three keys:
        token_ids : list[int]  — token IDs of the generated description only
        tokens    : list[str]  — subword strings
        text      : str        — human-readable description
    """
    messages = [{                                                          # Qwen2 chat format
        "role": "user",
        "content": [
            {"type": "image"},                                             # image placeholder
            {"type": "text", "text": "Could you describe the image?"},    # question text
        ],
    }]

    # Apply Qwen2 chat template → formatted string with special tokens
    text = processor.apply_chat_template(                                  # converts messages → model input string
        messages, tokenize=False, add_generation_prompt=True)             # add assistant prefix for generation

    inputs = processor(                                                    # processor handles image tiling + tokenisation
        images=img,
        text=text,
        return_tensors='pt',
    ).to(device)                                                           # move all tensors to GPU

    prompt_len = inputs.input_ids.shape[1]                                 # number of tokens in prompt (text + image patches)

    with torch.no_grad():                                                  # inference only
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,                                               # greedy decoding
        )

    generated_ids = output_ids[0, prompt_len:]                            # strip prompt tokens → only new tokens

    # Strip EOS if present
    tokenizer = processor.tokenizer                                        # access underlying Qwen2 tokenizer
    if len(generated_ids) > 0 and generated_ids[-1].item() == tokenizer.eos_token_id:
        generated_ids = generated_ids[:-1]                                 # remove EOS token

    token_ids = generated_ids.tolist()                                     # tensor → list[int]
    tokens    = [tokenizer.decode(tid) for tid in token_ids]               # decode each ID individually → subword strings
    text      = tokenizer.decode(generated_ids,
                                 skip_special_tokens=True).strip()         # full readable string

    return {"token_ids": token_ids, "tokens": tokens, "text": text}


# ═══════════════════════════════════════════════════════════════════
# Figure 3 test — 6 images, shows token/word format + stage 1 input
# ═══════════════════════════════════════════════════════════════════

FIG3_IMAGES = {                                                           # COCO image IDs from Xu et al. Figure 3
    "000000403170": "(a) Visual neuron - airplane/motorcycles",
    "000000065793": "(b) Text neuron - teddy bears",
    "000000156852": "(c) Multi-modal neuron - kitchen/thumbs up",
    "000000323964": "(d) Multi-modal neuron - doughnuts",
    "000000276332": "(e) Multi-modal neuron - zebras",
    "000000060034": "(f) Multi-modal neuron - fire hydrant/pigeons",
}


def run_test_fig3(args):
    """Generate descriptions for the 6 Figure 3 images.

    For each image, prints:
      1. Word format  — human-readable decoded text
      2. Token format — subword pieces as the tokenizer produces them
      3. Clean token format — subword pieces with ▁ stripped
      4. Stage 1 input layout — exact sequence fed into LLaVA for
         teacher-forcing activation recording:

         [BOS] USER: <576 image patches> \\nCould you describe the image?
         \\nASSISTANT: <generated_tok_1> <generated_tok_2> ... [EOS]

         Positions 3–578   → "visual tokens"  (576 CLIP patches)
         Positions after ASSISTANT: → "text tokens" (generated description)
    """
    device = f'cuda:{args.device}' if args.device.isdigit() else args.device  # e.g. "0" → "cuda:0"

    backend_name = args.model_type.upper()                                # "HF" or "LIUHAOTIAN" for display
    print(f"\n{'█'*70}")
    print(f"  Running {backend_name} backend")
    print(f"{'█'*70}")

    # ── Load model ──────────────────────────────────────────
    prompt = None                                                         # internvl/qwen2vl build prompts internally
    if args.model_type == 'llava-hf':
        print(f'Loading HF model: {args.hf_id} ...')
        model, processor = load_model_hf(args.hf_id, device)             # returns model + processor
        prompt = build_prompt_hf()                                        # HF prompt (no system prompt)
    elif args.model_type == 'llava-liuhaotian':
        print(f'Loading original model: {args.original_model_path} ...')
        model, tokenizer, image_proc = load_model_original(              # returns model, tokenizer, image_processor
            args.original_model_path, device)
        prompt = build_prompt_original()                                  # v1 template prompt (with system prompt)
    elif args.model_type == 'internvl':
        print(f'Loading InternVL model: {args.model_path} ...')
        model, tokenizer = load_model_internvl(args.model_path, device)  # returns model + tokenizer
    elif args.model_type == 'qwen2vl':
        print(f'Loading Qwen2.5-VL model: {args.model_path} ...')
        model, processor = load_model_qwen2vl(args.model_path, device)  # returns model + processor
    elif args.model_type == 'llava-ov':
        _path = args.model_path or 'modern_vlms/pretrained/llava-onevision-qwen2-7b-ov-hf'
        print(f'Loading LLaVA-OneVision model: {_path} ...')
        model, processor = load_model_llava_ov(_path, device)           # returns model + processor

    if prompt is None:                                                     # internvl/qwen2vl/llava-ov embed the prompt in their API
        prompt = '(embedded in model API)'
    print(f'Prompt: {repr(prompt)}')                                      # show exact prompt with escape chars visible
    results = {}

    for img_id, label in FIG3_IMAGES.items():                             # iterate over the 6 Figure 3 images
        print(f"\n{'═'*70}")
        print(f"{label} — COCO ID: {img_id}")
        print('═'*70)

        img_path = os.path.join(args.coco_img_dir, f"{img_id}.jpg")       # construct full path to COCO image
        img = Image.open(img_path).convert('RGB')                         # load image and convert to RGB

        # ── Generate description ────────────────────────────
        if args.model_type == 'llava-hf':
            result = generate_description_hf(                             # HF generation path
                model, processor, prompt, img, device,
                args.max_new_tokens, args.min_new_tokens)
        elif args.model_type == 'llava-liuhaotian':
            result = generate_description_original(                       # liuhaotian generation path
                model, tokenizer, image_proc, prompt, img, device,
                args.max_new_tokens, args.min_new_tokens)
        elif args.model_type == 'internvl':
            result = generate_description_internvl(                       # InternVL generation path
                model, tokenizer, img, device,
                args.max_new_tokens, args.min_new_tokens)
        elif args.model_type == 'qwen2vl':
            result = generate_description_qwen2vl(                        # Qwen2.5-VL generation path
                model, processor, img, device,
                args.max_new_tokens, args.min_new_tokens)
        elif args.model_type == 'llava-ov':
            result = generate_description_llava_ov(                       # LLaVA-OneVision generation path
                model, processor, img, device,
                args.max_new_tokens, args.min_new_tokens)

        tokens = result['tokens']                                         # list of subword strings (with ▁)
        clean_tokens = [t.lstrip('▁').strip() for t in tokens]            # stripped version for display
        text = result['text']                                             # human-readable text

        print(f"\n[WORD FORMAT] ({len(text.split())} words):")
        print(f"  {text}")
        print(f"\n[TOKEN FORMAT] ({len(tokens)} tokens):")
        print(f"  {' '.join(tokens)}")
        print(f"\n[CLEAN TOKEN FORMAT] ({len(clean_tokens)} tokens):")
        print(f"  {' '.join(clean_tokens)}")

        # ── Stage 1: Show teacher-forcing input layout ──────
        # Approximate layout based on known LLaVA-1.5 structure
        desc_start = 10 + 576                                             # ~10 prompt tokens + 576 CLIP patches
        n_text_tokens = len(tokens)                                       # number of generated description tokens

        print(f"\n[STAGE 1 — TEACHER FORCING INPUT]")
        print(f"  Visual token positions:  3–578  (576 CLIP patches)")
        print(f"  Text token positions:    ~{desc_start}–{desc_start + n_text_tokens - 1}  "
              f"({n_text_tokens} generated tokens)")
        print(f"\n  Input sequence order:")
        print(f"    [BOS] → USER: → [576 image patches] → "
              f"\\nCould you describe the image?\\nASSISTANT: → "
              f"[{n_text_tokens} generated tokens]")

        results[img_id] = {                                               # store results for this image
            "label": label,
            "token_ids": result['token_ids'],
            "tokens": tokens,
            "text": text,
        }

    # ── Save results ─────────────────────────────────────────
    output_path = args.output_path.replace('.json', f'_fig3_{args.model_type}.json')  # append backend name to filename
    os.makedirs(os.path.dirname(output_path), exist_ok=True)              # create output directory if needed
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)                                   # write JSON with pretty formatting

    print(f"\n{'═'*70}")
    print(f"Saved results to {output_path}")
    print(f"\nSummary:")
    for img_id, r in results.items():                                     # print summary for each image
        print(f"  {r['label']}: {len(r['tokens'])} tokens, "
              f"{len(r['text'].split())} words")

    # ── Free GPU memory ──────────────────────────────────────
    if args.model_type == 'llava-hf':
        del model, processor                                              # delete HF model references
    elif args.model_type == 'llava-liuhaotian':
        del model, tokenizer, image_proc                                  # delete liuhaotian model references
    elif args.model_type == 'internvl':
        del model, tokenizer                                              # delete InternVL model references
    elif args.model_type == 'qwen2vl':
        del model, processor                                              # delete Qwen2.5-VL model references
    elif args.model_type == 'llava-ov':
        del model, processor                                              # delete LLaVA-OneVision model references
    torch.cuda.empty_cache()                                              # release GPU memory back to CUDA


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    args = parse_args()                                                   # parse command-line arguments

    # ── Per-model min_new_tokens override ─────────────────────
    # Some models (e.g. Qwen2.5-VL) are much more concise than LLaVA and
    # hit EOS at ~100 tokens.  The taxonomy needs 300-500 tokens for
    # proper neuron classification, so we bump the floor for concise models.
    _min_tokens_by_model = {                                               # only override if user didn't change the default
        'qwen2vl': 300,                                                    # Qwen2.5-VL defaults to ~100 tokens without this
    }
    if args.min_new_tokens == 100:                                         # 100 is the argparse default
        override = _min_tokens_by_model.get(args.model_type)               # check if this model needs a bump
        if override is not None:
            print(f'ℹ Overriding min_new_tokens {args.min_new_tokens} → '
                  f'{override} for {args.model_type} (concise model)')
            args.min_new_tokens = override

    # ── Test mode: run on Figure 3 images only ────────────
    if args.test_fig3:                                                    # if --test_fig3 flag was passed
        run_test_fig3(args)                                               # run test on 6 images and exit
        return

    device = f'cuda:{args.device}' if args.device.isdigit() else args.device  # e.g. "0" → "cuda:0", "cpu" → "cpu"

    # ─────────────────────────────────────────────────────────
    # Load image list from detail_23k.json
    # ─────────────────────────────────────────────────────────
    print(f'Loading image list from {args.detail_23k_path} ...')
    with open(args.detail_23k_path) as f:
        detail_data = json.load(f)                                        # load the JSON list of image entries

    image_list = []
    for item in detail_data:
        img_id = item['id']                                               # e.g. "000000323964"
        img_path = os.path.join(args.coco_img_dir,
                                os.path.basename(item['image']))          # full path to COCO image
        image_list.append((img_id, img_path))                             # list of (id, path) tuples

    print(f'Total images in detail_23k: {len(image_list)}')

    # Apply sharding — split image list by index range
    end_idx = args.end_idx if args.end_idx is not None else len(image_list)  # default to all images
    image_list = image_list[args.start_idx:end_idx]                       # slice to requested range
    print(f'Processing indices {args.start_idx} to {end_idx} '
          f'({len(image_list)} images)')

    # ─────────────────────────────────────────────────────────
    # Load model
    # ─────────────────────────────────────────────────────────
    prompt = None                                                         # internvl/qwen2vl build prompts internally
    if args.model_type == 'llava-hf':
        print(f'Loading HF model: {args.hf_id} ...')
        model, processor = load_model_hf(args.hf_id, device)             # HF: model + processor
        prompt = build_prompt_hf()                                        # HF prompt (no system prompt)
        tokenizer = None                                                  # not used — processor wraps it
        image_proc = None                                                 # not used — processor wraps it
    elif args.model_type == 'llava-liuhaotian':
        print(f'Loading original model: {args.original_model_path} ...')
        model, tokenizer, image_proc = load_model_original(              # liuhaotian: model + tokenizer + image_processor
            args.original_model_path, device)
        prompt = build_prompt_original()                                  # v1 template prompt (with system prompt)
        processor = None                                                  # not used — separate tokenizer/image_proc
    elif args.model_type == 'internvl':
        print(f'Loading InternVL model: {args.model_path} ...')
        model, tokenizer = load_model_internvl(args.model_path, device)  # InternVL: model + tokenizer
        processor = None                                                  # not used — model.chat() handles preprocessing
        image_proc = None
    elif args.model_type == 'qwen2vl':
        print(f'Loading Qwen2.5-VL model: {args.model_path} ...')
        model, processor = load_model_qwen2vl(args.model_path, device)  # Qwen2.5-VL: model + processor
        tokenizer = None                                                  # embedded in processor
        image_proc = None
    elif args.model_type == 'llava-ov':
        _path = args.model_path or 'modern_vlms/pretrained/llava-onevision-qwen2-7b-ov-hf'
        print(f'Loading LLaVA-OneVision model: {_path} ...')
        model, processor = load_model_llava_ov(_path, device)           # LLaVA-OV: model + processor
        tokenizer = None                                                  # embedded in processor
        image_proc = None

    print(f'Model loaded. Prompt: {repr(prompt[:80]) if prompt else "(embedded in model API)"}...')

    # ─────────────────────────────────────────────────────────
    # Generate descriptions
    # ─────────────────────────────────────────────────────────
    # Resume support: load existing output if present, skip done images
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)         # create output directory if needed
    if os.path.exists(args.output_path):                                  # check if output file already exists
        with open(args.output_path) as f:
            descriptions = json.load(f)                                   # load previously generated descriptions
        print(f'Resuming: loaded {len(descriptions)} existing descriptions')
    else:
        descriptions = {}                                                 # start fresh

    skipped = 0
    t0 = time.time()                                                      # start timer

    for img_id, img_path in tqdm(image_list, desc='Generating'):          # loop over images with progress bar
        # Skip already generated (resume support)
        if img_id in descriptions:                                        # already processed in a previous run
            continue

        # Load image
        try:
            img = Image.open(img_path).convert('RGB')                     # open and convert to RGB
        except Exception as e:
            tqdm.write(f'Skip {img_path}: {e}')                           # log error without breaking progress bar
            skipped += 1
            continue

        # Generate description — dispatch to correct backend
        if args.model_type == 'llava-hf':
            generated_text = generate_description_hf(                     # HF generation path
                model, processor, prompt, img, device,
                args.max_new_tokens, args.min_new_tokens)
        elif args.model_type == 'llava-liuhaotian':
            generated_text = generate_description_original(               # liuhaotian generation path
                model, tokenizer, image_proc, prompt, img, device,
                args.max_new_tokens, args.min_new_tokens)
        elif args.model_type == 'internvl':
            generated_text = generate_description_internvl(               # InternVL generation path
                model, tokenizer, img, device,
                args.max_new_tokens, args.min_new_tokens)
        elif args.model_type == 'qwen2vl':
            generated_text = generate_description_qwen2vl(                # Qwen2.5-VL generation path
                model, processor, img, device,
                args.max_new_tokens, args.min_new_tokens)
        elif args.model_type == 'llava-ov':
            generated_text = generate_description_llava_ov(               # LLaVA-OneVision generation path
                model, processor, img, device,
                args.max_new_tokens, args.min_new_tokens)

        descriptions[img_id] = generated_text                             # store result keyed by image ID

        # Save periodically to survive preemption (every 50 images)
        if len(descriptions) % 50 == 0:
            with open(args.output_path, 'w') as f:
                json.dump(descriptions, f, indent=2)                      # checkpoint save

    elapsed = time.time() - t0                                            # total generation time

    # ─────────────────────────────────────────────────────────
    # Save
    # ─────────────────────────────────────────────────────────
    with open(args.output_path, 'w') as f:
        json.dump(descriptions, f, indent=2)                              # final save of all descriptions

    print(f'\nDone in {elapsed/60:.1f} min')
    print(f'Generated: {len(descriptions)}, Skipped: {skipped}')
    print(f'Saved to {args.output_path}')

    # Print a few examples for sanity checking
    print('\n--- Sample descriptions ---')
    for i, (img_id, desc) in enumerate(descriptions.items()):
        if i >= 3:                                                        # only show first 3
            break
        print(f'\n{img_id}: {desc["text"][:150]}...')                     # first 150 chars of text
        print(f'  tokens ({len(desc["token_ids"])}): {desc["tokens"][:10]}...')  # first 10 tokens


if __name__ == '__main__':
    main()


# """
# generate_descriptions_both_backbone.py — Step 0 of Xu et al. classification

# Generate detailed image descriptions using a VLM for all images
# in the detail_23k subset. These descriptions serve as the text tokens
# for neuron classification.

# Supports four backends:
#   --model_type llava-hf         : HuggingFace llava-hf/llava-1.5-7b-hf
#   --model_type llava-liuhaotian : Original liuhaotian/llava-v1.5-7b via LLaVA repo
#                             (requires transformers==4.37.2)
#   --model_type internvl   : InternVL2.5-8B (OpenGVLab/InternVL2_5-8B)
#                             (requires trust_remote_code=True)
#   --model_type qwen2vl    : Qwen2.5-VL-7B-Instruct (Qwen/Qwen2.5-VL-7B-Instruct)
#                             (requires qwen-vl-utils)

# Output: generated_descriptions.json — {image_id: {"token_ids": [...], "tokens": [...], "text": "..."}}

# Usage:
#     # Full dataset with HF backend
#     python generate_descriptions_both_backbone.py --model_type llava-hf --start_idx 0 --end_idx 23000

#     # Full dataset with liuhaotian backend
#     python generate_descriptions_both_backbone.py --model_type llava-liuhaotian --start_idx 0 --end_idx 23000

#     # Full dataset with InternVL backend
#     python generate_descriptions_both_backbone.py --model_type internvl \\
#         --internvl_path pretrained/InternVL2_5-8B --start_idx 0 --end_idx 23000

#     # Full dataset with Qwen2.5-VL backend
#     python generate_descriptions_both_backbone.py --model_type qwen2vl \\
#         --qwen2vl_path pretrained/Qwen2.5-VL-7B-Instruct --start_idx 0 --end_idx 23000

#     # Test on Figure 3 images
#     python generate_descriptions_both_backbone.py --model_type internvl \\
#         --internvl_path pretrained/InternVL2_5-8B --test_fig3
#     python generate_descriptions_both_backbone.py --model_type qwen2vl \\
#         --qwen2vl_path pretrained/Qwen2.5-VL-7B-Instruct --test_fig3
# """

# import argparse                                                           # command-line argument parsing
# import json                                                               # reading/writing JSON files
# import os                                                                 # file path manipulation
# import sys                                                                # modifying the Python import path
# import time                                                               # timing the generation loop

# # Add LLaVA repo to path so we can import from llava.* (needed for liuhaotian backend)
# _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # two levels up from this script
# _LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')                       # path to cloned LLaVA repo
# if _LLAVA_PATH not in sys.path:                                           # avoid duplicate entries
#     sys.path.insert(0, _LLAVA_PATH)                                       # insert at front so llava imports resolve here first

# import torch                                                              # PyTorch — tensor operations, GPU, inference
# from PIL import Image                                                     # Pillow — loading image files
# from tqdm import tqdm                                                     # progress bar for the generation loop


# def parse_args():
#     """Parse command-line arguments for dataset paths, generation settings, and sharding."""
#     p = argparse.ArgumentParser(
#         description='Generate LLaVA descriptions for detail_23k images (both backends)')

#     # Backend selection
#     p.add_argument('--model_type', default='llava-hf',
#                    choices=['llava-hf', 'llava-liuhaotian', 'internvl', 'qwen2vl'],
#                    help='"llava-hf" for llava-hf/llava-1.5-7b-hf, '
#                         '"llava-liuhaotian" for original llava-v1.5-7b, '
#                         '"internvl" for InternVL2_5-8B, '
#                         '"qwen2vl" for Qwen2.5-VL-7B-Instruct')           # which backend to use

#     # Model paths
#     p.add_argument('--hf_id', default='llava-hf/llava-1.5-7b-hf',
#                    help='HuggingFace model ID (for --model_type llava-hf)')     # HF Hub ID
#     p.add_argument('--original_model_path', default='liuhaotian/llava-v1.5-7b',
#                    help='Original LLaVA model path (for --model_type llava-liuhaotian)')  # HF Hub ID or local path
#     p.add_argument('--internvl_path', default='pretrained/InternVL2_5-8B',
#                    help='Local path to InternVL2.5 model dir (for --model_type internvl)')  # local path only (trust_remote_code)
#     p.add_argument('--qwen2vl_path', default='pretrained/Qwen2.5-VL-7B-Instruct',
#                    help='Local path to Qwen2.5-VL model dir (for --model_type qwen2vl)')    # local path or HF Hub ID

#     # Data paths
#     _script_dir = os.path.dirname(os.path.abspath(__file__))              # directory containing this script
#     _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))  # project root (two levels up)
#     p.add_argument('--detail_23k_path',
#                    default=os.path.join(_project_root, 'detail_23k.json'),
#                    help='Path to detail_23k.json (defines which images to use)')
#     p.add_argument('--coco_img_dir',
#                    default='/home/projects/bagon/shared/coco2017/images/train2017/',
#                    help='Path to COCO train2017 images')

#     # Output
#     p.add_argument('--output_path',
#                    default='generated_descriptions/generated_descriptions.json',
#                    help='Output JSON path')

#     # Sharding — allows splitting across multiple GPUs/jobs
#     p.add_argument('--start_idx', type=int, default=0,
#                    help='Start index in detail_23k list')
#     p.add_argument('--end_idx', type=int, default=None,
#                    help='End index in detail_23k list (None = all)')

#     # Generation parameters
#     p.add_argument('--min_new_tokens', type=int, default=100,
#                    help='Min tokens to generate per image')               # forces minimum description length
#     p.add_argument('--max_new_tokens', type=int, default=550,
#                    help='Max tokens to generate per image')               # caps maximum description length
#     p.add_argument('--device', default='0')                               # CUDA GPU index

#     # Test mode
#     p.add_argument('--test_fig3', action='store_true',
#                    help='Run on 6 Figure 3 images only. Shows token/word format '
#                         'and stage 1 teacher-forcing input layout.')

#     return p.parse_args()


# # ═══════════════════════════════════════════════════════════════════
# # Model loading
# # ═══════════════════════════════════════════════════════════════════

# def load_model_hf(hf_id, device):
#     """Load the HuggingFace LLaVA model.

#     Uses LlavaForConditionalGeneration from transformers — the HF port
#     of LLaVA-1.5 that bundles vision encoder + LLM in one model class.

#     Returns: (model, processor)
#         - model: LlavaForConditionalGeneration (vision + LLM)
#         - processor: AutoProcessor (handles both text tokenisation + image preprocessing)
#     """
#     from transformers import AutoProcessor, LlavaForConditionalGeneration  # HF classes for LLaVA

#     processor = AutoProcessor.from_pretrained(hf_id)                      # loads tokenizer + CLIP image processor
#     model = LlavaForConditionalGeneration.from_pretrained(
#         hf_id, torch_dtype=torch.float16, low_cpu_mem_usage=True          # float16 to save GPU memory
#     ).to(device).eval()                                                   # move to GPU, set eval mode

#     return model, processor


# def load_model_original(model_path, device):
#     """Load the original LLaVA model via the cloned LLaVA repo.

#     Requires the LLaVA repo on sys.path and transformers==4.37.2.

#     Returns: (model, tokenizer, image_processor)
#         - model: the LLaVA model (LLaMA + CLIP vision encoder + projection)
#         - tokenizer: the LLaMA SentencePiece tokenizer
#         - image_processor: CLIP image processor (resizes/normalises images)
#     """
#     from llava.model.builder import load_pretrained_model                 # loads model weights from HF Hub or local
#     from llava.mm_utils import get_model_name_from_path                   # derives model name string from path

#     model_name = get_model_name_from_path(model_path)                     # e.g. "llava-v1.5-7b"
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path, None, model_name, device_map=device,                  # None = no model_base (not a LoRA model)
#         torch_dtype=torch.float16                                         # half precision to save GPU memory
#     )
#     model.eval()                                                          # set to eval mode (disables dropout etc.)


#     return model, tokenizer, image_processor


# # ═══════════════════════════════════════════════════════════════════
# # InternVL image preprocessing helpers
# # ═══════════════════════════════════════════════════════════════════
# # These replicate the InternVL2.5 official preprocessing pipeline.
# # InternVL tiles high-res images into up to max_num sub-patches so
# # that fine-grained detail is preserved for the ViT.

# _IMAGENET_MEAN = (0.485, 0.456, 0.406)                                    # ImageNet channel means (RGB)
# _IMAGENET_STD  = (0.229, 0.224, 0.225)                                    # ImageNet channel stds  (RGB)


# def _internvl_build_transform(input_size=448):
#     """Build torchvision transform for a single InternVL image patch.

#     Resizes the patch to input_size×input_size, converts to tensor,
#     and normalises with ImageNet statistics.
#     """
#     from torchvision import transforms                                      # torchvision transforms pipeline
#     return transforms.Compose([
#         transforms.Resize(
#             (input_size, input_size),
#             interpolation=transforms.InterpolationMode.BICUBIC),           # BICUBIC matches ViT pre-training
#         transforms.ToTensor(),                                             # HWC uint8 → CHW float32 in [0,1]
#         transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),     # normalise per channel
#     ])


# def _internvl_dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
#     """Tile a PIL image into sub-patches whose aspect ratio best matches
#     the image's own aspect ratio.

#     InternVL2.5 chooses the tiling grid (rows×cols) that minimises the
#     aspect-ratio difference to the original image, subject to the
#     constraint min_num ≤ rows*cols ≤ max_num.  An optional thumbnail
#     (full image resized to image_size²) is appended as the last patch
#     to give the ViT a global view.

#     Args:
#         image: PIL.Image (RGB)
#         min_num: minimum number of patches (1)
#         max_num: maximum number of patches (12 by default)
#         image_size: side length of each square patch in pixels
#         use_thumbnail: append a global thumbnail patch (True)

#     Returns:
#         list of PIL.Image patches, length = rows*cols [+1 thumbnail]
#     """
#     orig_w, orig_h = image.size                                            # original image dimensions
#     aspect_ratio = orig_w / orig_h                                         # width-to-height ratio

#     # Build all candidate (rows, cols) grids that satisfy the count constraint
#     target_ratios = set(
#         (i, j)
#         for n in range(min_num, max_num + 1)
#         for i in range(1, n + 1)
#         for j in range(1, n + 1)
#         if min_num <= i * j <= max_num                                     # keep grid sizes within [min, max]
#     )
#     target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])     # sort by total patch count

#     # Pick the grid whose aspect ratio is closest to the image's own ratio
#     best_ratio = min(
#         target_ratios,
#         key=lambda r: abs(aspect_ratio - r[0] / r[1])                     # minimise aspect-ratio error
#     )
#     rows, cols = best_ratio                                                # e.g. (2, 3) → 6 patches

#     # Resize the full image to exactly fit the chosen grid
#     target_w = image_size * rows                                           # e.g. 448 * 2 = 896 pixels wide
#     target_h = image_size * cols                                           # e.g. 448 * 3 = 1344 pixels tall
#     resized = image.resize((target_w, target_h),
#                             resample=Image.BICUBIC)                        # high-quality downscale

#     # Crop out each patch in row-major order
#     patches = []
#     for row in range(rows):
#         for col in range(cols):
#             left   = col * image_size                                      # left edge of this patch
#             upper  = row * image_size                                      # top edge of this patch
#             right  = left  + image_size                                    # right edge (exclusive)
#             lower  = upper + image_size                                    # bottom edge (exclusive)
#             patches.append(resized.crop((left, upper, right, lower)))     # crop → square patch

#     if use_thumbnail and len(patches) > 1:                                 # add thumbnail if requested and >1 patch
#         thumbnail = image.resize((image_size, image_size),
#                                   resample=Image.BICUBIC)                  # global view at native resolution
#         patches.append(thumbnail)                                          # appended last

#     return patches                                                         # list[PIL.Image], each image_size×image_size


# def _internvl_preprocess_image(img, max_num=12):
#     """Convert a PIL image to InternVL pixel_values tensor.

#     Tiles the image into up to max_num sub-patches (+ thumbnail),
#     applies the standard InternVL transform, and stacks into a
#     float16 tensor of shape (n_patches, 3, 448, 448).

#     Args:
#         img: PIL.Image (RGB)
#         max_num: max number of tiles (12)

#     Returns:
#         torch.Tensor, shape (n_patches, 3, 448, 448), dtype float16
#     """
#     transform = _internvl_build_transform(input_size=448)                 # resize+normalise transform
#     patches   = _internvl_dynamic_preprocess(img, image_size=448,         # tile image into sub-patches
#                                               use_thumbnail=True,
#                                               max_num=max_num)
#     pixel_values = torch.stack([transform(p) for p in patches])           # (n_patches, 3, 448, 448) float32
#     return pixel_values.half()                                             # cast to float16 for model inference


# # ═══════════════════════════════════════════════════════════════════
# # New model loaders
# # ═══════════════════════════════════════════════════════════════════

# def load_model_internvl(model_path, device):
#     """Load an InternVL2.5 model from a local directory.

#     InternVL2.5 ships custom modeling code inside the checkpoint, so
#     trust_remote_code=True is required.  The model is loaded in
#     bfloat16 to match its training precision.

#     Args:
#         model_path: local path to the InternVL2.5 checkpoint directory
#         device: torch device string, e.g. 'cuda:0'

#     Returns: (model, tokenizer)
#         - model: InternVLChatModel (vision encoder + LLM + projector)
#         - tokenizer: the InternLM / Qwen tokenizer bundled with the model
#     """
#     from transformers import AutoTokenizer, AutoModel               # standard HF loading interface

#     tokenizer = AutoTokenizer.from_pretrained(
#         model_path, trust_remote_code=True)                                # custom tokenizer code lives in checkpoint

#     model = AutoModel.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,                                        # bfloat16 — matches training precision
#         trust_remote_code=True,                                            # allow InternVL's custom modeling code
#     ).to(device).eval()                                                    # move to GPU, disable dropout


#     return model, tokenizer


# def load_model_qwen2vl(model_path, device):
#     """Load a Qwen2.5-VL model.

#     Uses the official Qwen2_5_VLForConditionalGeneration class from
#     transformers (≥4.45).  The processor handles both text tokenisation
#     and image preprocessing via its built-in vision processor.

#     Args:
#         model_path: local path or HuggingFace Hub ID
#         device: torch device string, e.g. 'cuda:0'

#     Returns: (model, processor)
#         - model: Qwen2_5_VLForConditionalGeneration
#         - processor: AutoProcessor (tokenizer + image processor combined)
#     """
#     from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor  # Qwen2.5-VL classes

#     processor = AutoProcessor.from_pretrained(model_path)                  # handles text + vision

#     model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#         model_path,
#         torch_dtype=torch.bfloat16,                                        # bfloat16 for memory efficiency
#     ).to(device).eval()                                                    # move to GPU, set eval mode

#     return model, processor


# # ═══════════════════════════════════════════════════════════════════
# # Prompt building
# # ═══════════════════════════════════════════════════════════════════

# def build_prompt_hf():
#     """Build generation prompt for the HF backend.

#     HF LLaVA was ported without the system prompt, so the prompt is:
#         USER: <image>\\nCould you describe the image?\\nASSISTANT:

#     Returns the formatted prompt string.
#     """
#     return "USER: <image>\nCould you describe the image?\nASSISTANT:"    # no system prompt, \\n separators


# def build_prompt_original():
#     """Build generation prompt for the liuhaotian backend using conv_templates["v1"].

#     The v1 template (conv_vicuna_v1) formats the conversation as:
#         {system_prompt} USER: <image>\\nCould you describe the image? ASSISTANT:

#     Returns the formatted prompt string.
#     """
#     from llava.conversation import conv_templates                         # conversation template registry

#     conv = conv_templates["v1"].copy()                                    # copy to avoid mutating the global template
#     conv.append_message(conv.roles[0], "<image>\nCould you describe the image?")  # USER turn with <image> placeholder
#     conv.append_message(conv.roles[1], None)                              # empty ASSISTANT turn (model will generate)
#     return conv.get_prompt()                                              # serialise into formatted string


# # ═══════════════════════════════════════════════════════════════════
# # Generation
# # ═══════════════════════════════════════════════════════════════════

# def generate_description_hf(model, processor, prompt, img, device, max_new_tokens, min_new_tokens):
#     """Generate a text description for a single image using the HF LLaVA model.

#     Steps:
#       1. Use processor to tokenise prompt + preprocess image together
#       2. Run greedy decoding with model.generate()
#       3. Extract only the NEW tokens (strip the prompt tokens)
#       4. Decode into three formats

#     Returns dict with three keys:
#         token_ids    : list[int]   — raw token IDs of generated text only
#         tokens       : list[str]   — subword strings like ["▁d","ough","n","uts"]
#         text         : str         — human-readable "doughnuts..."
#     """
#     # Process text + image together via the HF processor
#     inputs = processor(                                                   # AutoProcessor handles both modalities
#         images=img, text=prompt, return_tensors='pt'                      # returns dict with input_ids, pixel_values, etc.
#     ).to(device)                                                          # move all tensors to GPU

#     prompt_len = inputs['input_ids'].shape[1]                             # number of tokens in the prompt (including expanded <image>)

#     with torch.no_grad():                                                 # disable gradient computation (inference only)
#         output_ids = model.generate(
#             **inputs,                                                     # unpack input_ids, pixel_values, attention_mask
#             max_new_tokens=max_new_tokens,                                # maximum number of tokens to generate
#             min_new_tokens=min_new_tokens,                                # force minimum description length
#             do_sample=False,                                              # greedy decoding (always pick highest prob token)
#         )

#     # HF generate() returns full sequence (prompt + generated), so strip prompt
#     generated_ids = output_ids[0, prompt_len:]                            # slice off prompt tokens → only new tokens

#     # Strip EOS token if present at the end
#     tokenizer = processor.tokenizer                                       # access the underlying tokenizer
#     if len(generated_ids) > 0 and generated_ids[-1].item() == tokenizer.eos_token_id:  # check last token
#         generated_ids = generated_ids[:-1]                                # remove EOS

#     # Three output formats:
#     token_ids = generated_ids.tolist()                                    # convert tensor to plain Python list of ints
#     tokens = [tokenizer.decode(tid) for tid in token_ids]                 # decode each token ID individually → subword strings

#     # Build readable text
#     text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()  # decode all IDs into one string

#     # Strip common prefixes (model sometimes starts with a polite preamble)
#     for prefix in ['Yes, I can describe the image.', 'Yes, I can describe the image',
#                     'Yes,', 'Yes.']:
#         if text.startswith(prefix):                                       # check if text starts with this prefix
#             text = text[len(prefix):].strip()                             # remove prefix and trim whitespace
#             break                                                         # only strip the first matching prefix

#     # Strip matching prefix from token lists to keep them aligned with the cleaned text
#     clean_tokens = [t.lstrip('▁').strip() for t in tokens]                # strip ▁ prefix for matching
#     PREFIX_TOKENS = [                                                     # known prefix patterns
#         ['Yes', ',', 'I', 'can', 'describe', 'the', 'image', '.'],       # "Yes, I can describe the image."
#         ['Yes', ',', 'I', 'can', 'see', 'the', 'image', '.'],            # "Yes, I can see the image."
#         ['Yes', ','],                                                     # "Yes,"
#         ['Yes', '.'],                                                     # "Yes."
#     ]
#     for prefix_toks in PREFIX_TOKENS:
#         n = len(prefix_toks)                                              # number of tokens in this prefix pattern
#         if clean_tokens[:n] == prefix_toks:                               # match against clean_tokens (no ▁)
#             token_ids = token_ids[n:]                                     # strip same count from all three lists
#             tokens = tokens[n:]
#             clean_tokens = clean_tokens[n:]
#             break                                                         # only strip one prefix

#     return {"token_ids": token_ids, "tokens": tokens, "text": text}


# def generate_description_original(model, tokenizer, image_processor, prompt, img, device, max_new_tokens, min_new_tokens):
#     """Generate a text description for a single image using the original LLaVA model.

#     Steps:
#       1. Tokenise the prompt, replacing <image> with IMAGE_TOKEN_INDEX (-200)
#       2. Preprocess the image using CLIP's image processor (resize to 336x336, normalise)
#       3. Run greedy decoding with model.generate()
#       4. Decode the generated token IDs into three formats

#     Returns dict with three keys:
#         token_ids    : list[int]   — raw token IDs (used in step 2 classification)
#         tokens       : list[str]   — subword strings like ["▁d","ough","n","uts"]
#         text         : str         — human-readable "doughnuts..."
#     """
#     from llava.constants import IMAGE_TOKEN_INDEX                         # special token ID (-200) for <image>
#     from llava.mm_utils import tokenizer_image_token                      # tokenise text, replacing <image> with special ID

#     # Tokenise the prompt, replacing <image> text with IMAGE_TOKEN_INDEX (-200)
#     input_ids = tokenizer_image_token(
#         prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'         # returns a PyTorch tensor
#     ).unsqueeze(0).to(device)                                             # add batch dim → shape (1, seq_len), move to GPU

#     # Preprocess image using CLIP processor (resize to 336x336, normalise pixel values)
#     image_tensor = image_processor.preprocess(
#         img, return_tensors='pt'                                          # returns dict with 'pixel_values' tensor
#     )['pixel_values'].half().to(device)                                   # convert to float16, move to GPU → (1, 3, 336, 336)

#     with torch.no_grad():                                                 # disable gradient computation (inference only)
#         output_ids = model.generate(
#             input_ids,                                                    # tokenised prompt with IMAGE_TOKEN_INDEX
#             images=image_tensor,                                          # preprocessed image passed as keyword arg
#             max_new_tokens=max_new_tokens,                                # maximum number of tokens to generate
#             min_new_tokens=min_new_tokens,                                # force minimum description length
#             do_sample=False,                                              # greedy decoding (always pick highest prob token)
#         )

#     # transformers 4.37.2: generate() returns full sequence (prompt + generated)
#     generated_ids = output_ids[0]                                         # generate() returns only new tokens for original LLaVA model

#     # Three output formats:
#     token_ids = generated_ids.tolist()                                    # convert tensor to plain Python list of ints
#     tokens = [tokenizer.decode(tid) for tid in token_ids]                 # decode each token ID individually → subword strings

#     # Build readable text: tokenizer.decode() works correctly in transformers 4.37.2
#     text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()  # decode all IDs into one string, remove <s>/</s>

#     # Strip common prefixes from v1 conv template (model sometimes starts with a polite preamble)
#     for prefix in ['Yes, I can describe the image.', 'Yes, I can describe the image',
#                     'Yes,', 'Yes.']:
#         if text.startswith(prefix):                                       # check if text starts with this prefix
#             text = text[len(prefix):].strip()                             # remove prefix and trim whitespace
#             break                                                         # only strip the first matching prefix

#     # Strip matching prefix from token lists to keep them aligned with the cleaned text
#     clean_tokens = [t.lstrip('▁').strip() for t in tokens]                # strip ▁ prefix for matching
#     PREFIX_TOKENS = [                                                     # known v1 template prefix patterns
#         ['Yes', ',', 'I', 'can', 'describe', 'the', 'image', '.'],       # "Yes, I can describe the image."
#         ['Yes', ',', 'I', 'can', 'see', 'the', 'image', '.'],            # "Yes, I can see the image."
#         ['Yes', ','],                                                     # "Yes,"
#         ['Yes', '.'],                                                     # "Yes."
#     ]
#     for prefix_toks in PREFIX_TOKENS:
#         n = len(prefix_toks)                                              # number of tokens in this prefix pattern
#         if clean_tokens[:n] == prefix_toks:                               # match against clean_tokens (no ▁)
#             token_ids = token_ids[n:]                                     # strip same count from all three lists
#             tokens = tokens[n:]
#             clean_tokens = clean_tokens[n:]
#             break                                                         # only strip one prefix

#     return {"token_ids": token_ids, "tokens": tokens, "text": text}


# def generate_description_internvl(model, tokenizer, img, device,
#                                    max_new_tokens, min_new_tokens):
#     """Generate a text description for a single image using InternVL2.5.

#     InternVL2.5 exposes a high-level `model.chat()` API that takes
#     pre-processed pixel_values and returns a text string.  We then
#     re-tokenise the response to recover the same token_ids / tokens
#     format that the downstream classification pipeline expects.

#     Steps:
#       1. Tile the PIL image into sub-patches → pixel_values tensor
#       2. Call model.chat() with greedy decoding config
#       3. Re-tokenise the response text to get token_ids + tokens

#     Args:
#         model:          InternVLChatModel on GPU
#         tokenizer:      InternLM / Qwen tokenizer bundled with the model
#         img:            PIL.Image (RGB)
#         device:         torch device string, e.g. 'cuda:0'
#         max_new_tokens: upper bound on generated tokens
#         min_new_tokens: lower bound on generated tokens

#     Returns dict with three keys:
#         token_ids : list[int]   — token IDs of the generated text
#         tokens    : list[str]   — subword strings per token
#         text      : str         — decoded human-readable description
#     """
#     # Tile image into InternVL sub-patches and move to GPU
#     pixel_values = _internvl_preprocess_image(img, max_num=12)             # (n_patches, 3, 448, 448) float16
#     pixel_values = pixel_values.to(device)                                 # move tensor to same device as model

#     generation_config = dict(
#         max_new_tokens=max_new_tokens,                                     # cap on generated tokens
#         min_new_tokens=min_new_tokens,                                     # force minimum length
#         do_sample=False,                                                   # greedy decoding
#     )

#     question = 'Could you describe the image?'                             # same prompt intent as LLaVA versions

#     with torch.no_grad():                                                  # disable gradient tracking for inference
#         text = model.chat(                                                 # InternVL high-level chat API
#             tokenizer, pixel_values, question, generation_config)         # returns plain string response

#     # Strip common polite preambles the model may prepend
#     for prefix in ['Yes, I can describe the image.', 'Yes, I can describe the image',
#                    'Yes,', 'Yes.']:
#         if text.startswith(prefix):                                        # check for matching prefix
#             text = text[len(prefix):].strip()                             # remove and trim
#             break                                                          # only strip first match

#     # Re-tokenise to recover token_ids and subword token strings
#     token_ids = tokenizer.encode(text, add_special_tokens=False)           # list[int] — no BOS/EOS
#     tokens    = [tokenizer.decode([tid]) for tid in token_ids]             # list[str] — one string per token

#     return {"token_ids": token_ids, "tokens": tokens, "text": text}


# def generate_description_qwen2vl(model, processor, img, device,
#                                   max_new_tokens, min_new_tokens):
#     """Generate a text description for a single image using Qwen2.5-VL.

#     Qwen2.5-VL uses a chat-template processor.  We build a single-turn
#     conversation message containing the image and the question, apply
#     the template to produce the full input, run model.generate(), then
#     decode only the newly generated tokens.

#     Steps:
#       1. Build a messages list with image + question
#       2. Apply chat template via processor to get input_ids + pixel_values
#       3. Run model.generate() with greedy decoding
#       4. Decode only the generated suffix (strip prompt tokens)
#       5. Re-tokenise the response to get per-token strings

#     Args:
#         model:          Qwen2_5_VLForConditionalGeneration on GPU
#         processor:      AutoProcessor (tokenizer + image processor)
#         img:            PIL.Image (RGB)
#         device:         torch device string, e.g. 'cuda:0'
#         max_new_tokens: upper bound on generated tokens
#         min_new_tokens: lower bound on generated tokens

#     Returns dict with three keys:
#         token_ids : list[int]   — token IDs of the generated text
#         tokens    : list[str]   — subword strings per token
#         text      : str         — decoded human-readable description
#     """
#     from qwen_vl_utils import process_vision_info                          # official Qwen helper: extracts image tensors from messages

#     # Build a single-turn conversation containing the image and question
#     messages = [
#         {
#             "role": "user",
#             "content": [
#                 {"type": "image", "image": img},                          # PIL Image passed directly
#                 {"type": "text",  "text": "Could you describe the image?"},
#             ],
#         }
#     ]

#     # Apply chat template to produce the raw text prompt string
#     prompt_text = processor.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True)             # returns formatted string with <|im_start|> etc.

#     # Extract image tensors from the messages dict
#     image_inputs, video_inputs = process_vision_info(messages)            # list of PIL images (or video frames)

#     # Tokenise + preprocess image together
#     inputs = processor(
#         text=[prompt_text],                                               # list[str] → batch of 1
#         images=image_inputs,                                              # preprocessed image tensors
#         videos=video_inputs,                                              # empty for image-only input
#         padding=True,
#         return_tensors='pt',
#     ).to(device)                                                          # move all tensors to GPU

#     prompt_len = inputs['input_ids'].shape[1]                             # number of prompt tokens (to strip later)

#     with torch.no_grad():                                                  # inference only, no gradients
#         output_ids = model.generate(
#             **inputs,
#             max_new_tokens=max_new_tokens,                                # max generated tokens
#             min_new_tokens=min_new_tokens,                                # min generated tokens
#             do_sample=False,                                              # greedy decoding
#         )

#     # Qwen generate() returns full sequence: strip prompt tokens to get only response
#     generated_ids = output_ids[0, prompt_len:]                            # shape: (n_generated_tokens,)

#     # Strip EOS if present at the end
#     tokenizer = processor.tokenizer                                        # access the underlying tokenizer
#     if len(generated_ids) > 0 and generated_ids[-1].item() == tokenizer.eos_token_id:
#         generated_ids = generated_ids[:-1]                                # remove trailing EOS token

#     # Build three output formats
#     token_ids = generated_ids.tolist()                                    # list[int]
#     tokens    = [tokenizer.decode([tid]) for tid in token_ids]            # list[str] — one subword per entry
#     text      = tokenizer.decode(generated_ids,
#                                   skip_special_tokens=True).strip()       # full decoded string

#     # Strip common polite preambles
#     for prefix in ['Yes, I can describe the image.', 'Yes, I can describe the image',
#                    'Yes,', 'Yes.']:
#         if text.startswith(prefix):                                        # check for prefix match
#             text = text[len(prefix):].strip()                             # remove prefix
#             break                                                          # stop at first match

#     return {"token_ids": token_ids, "tokens": tokens, "text": text}

# FIG3_IMAGES = {                                                           # COCO image IDs from Xu et al. Figure 3
#     "000000403170": "(a) Visual neuron - airplane/motorcycles",
#     "000000065793": "(b) Text neuron - teddy bears",
#     "000000156852": "(c) Multi-modal neuron - kitchen/thumbs up",
#     "000000323964": "(d) Multi-modal neuron - doughnuts",
#     "000000276332": "(e) Multi-modal neuron - zebras",
#     "000000060034": "(f) Multi-modal neuron - fire hydrant/pigeons",
# }


# def run_test_fig3(args):
#     """Generate descriptions for the 6 Figure 3 images.

#     For each image, prints:
#       1. Word format  — human-readable decoded text
#       2. Token format — subword pieces as the tokenizer produces them
#       3. Clean token format — subword pieces with ▁ stripped
#       4. Stage 1 input layout — exact sequence fed into LLaVA for
#          teacher-forcing activation recording:

#          [BOS] USER: <576 image patches> \\nCould you describe the image?
#          \\nASSISTANT: <generated_tok_1> <generated_tok_2> ... [EOS]

#          Positions 3–578   → "visual tokens"  (576 CLIP patches)
#          Positions after ASSISTANT: → "text tokens" (generated description)
#     """
#     device = f'cuda:{args.device}' if args.device.isdigit() else args.device  # e.g. "0" → "cuda:0"

#     backend_name = args.model_type.upper()                                # "HF" or "LIUHAOTIAN" for display
#     print(f"\n{'█'*70}")
#     print(f"  Running {backend_name} backend")
#     print(f"{'█'*70}")

#     # ── Load model ──────────────────────────────────────────
#     if args.model_type == 'llava-hf':
#         print(f'Loading HF model: {args.hf_id} ...')
#         model, processor = load_model_hf(args.hf_id, device)             # returns model + processor
#         prompt = build_prompt_hf()                                        # HF prompt (no system prompt)
#     elif args.model_type == 'llava-liuhaotian':
#         print(f'Loading original model: {args.original_model_path} ...')
#         model, tokenizer, image_proc = load_model_original(              # returns model, tokenizer, image_processor
#             args.original_model_path, device)
#         prompt = build_prompt_original()                                  # v1 template prompt (with system prompt)
#     elif args.model_type == 'internvl':
#         print(f'Loading InternVL model: {args.internvl_path} ...')
#         model, tokenizer = load_model_internvl(args.internvl_path, device)  # returns model + tokenizer
#         prompt = 'Could you describe the image?'                          # InternVL uses plain-text question
#     else:  # qwen2vl
#         print(f'Loading Qwen2.5-VL model: {args.qwen2vl_path} ...')
#         model, processor = load_model_qwen2vl(args.qwen2vl_path, device)   # returns model + processor
#         prompt = 'Could you describe the image?'                          # Qwen uses chat-template, prompt shown for info only

#     prompt = prompt if 'prompt' in dir() else '(embedded in model.chat API)'
#     print(f'Prompt: {repr(prompt)}')                                      # show exact prompt with escape chars visible
#     results = {}

#     for img_id, label in FIG3_IMAGES.items():                             # iterate over the 6 Figure 3 images
#         print(f"\n{'═'*70}")
#         print(f"{label} — COCO ID: {img_id}")
#         print('═'*70)

#         img_path = os.path.join(args.coco_img_dir, f"{img_id}.jpg")       # construct full path to COCO image
#         img = Image.open(img_path).convert('RGB')                         # load image and convert to RGB

#         # ── Generate description ────────────────────────────
#         if args.model_type == 'llava-hf':
#             result = generate_description_hf(                             # HF generation path
#                 model, processor, prompt, img, device,
#                 args.max_new_tokens, args.min_new_tokens)
#         elif args.model_type == 'llava-liuhaotian':
#             result = generate_description_original(                       # liuhaotian generation path
#                 model, tokenizer, image_proc, prompt, img, device,
#                 args.max_new_tokens, args.min_new_tokens)
#         elif args.model_type == 'internvl':
#             result = generate_description_internvl(                       # InternVL generation path
#                 model, tokenizer, img, device,
#                 args.max_new_tokens, args.min_new_tokens)
#         else:  # qwen2vl
#             result = generate_description_qwen2vl(                        # Qwen2.5-VL generation path
#                 model, processor, img, device,
#                 args.max_new_tokens, args.min_new_tokens)

#         tokens = result['tokens']                                         # list of subword strings (with ▁)
#         clean_tokens = [t.lstrip('▁').strip() for t in tokens]            # stripped version for display
#         text = result['text']                                             # human-readable text

#         print(f"\n[WORD FORMAT] ({len(text.split())} words):")
#         print(f"  {text}")
#         print(f"\n[TOKEN FORMAT] ({len(tokens)} tokens):")
#         print(f"  {' '.join(tokens)}")
#         print(f"\n[CLEAN TOKEN FORMAT] ({len(clean_tokens)} tokens):")
#         print(f"  {' '.join(clean_tokens)}")

#         # ── Stage 1: Show teacher-forcing input layout ──────
#         # Approximate layout based on known LLaVA-1.5 structure
#         desc_start = 10 + 576                                             # ~10 prompt tokens + 576 CLIP patches
#         n_text_tokens = len(tokens)                                       # number of generated description tokens

#         print(f"\n[STAGE 1 — TEACHER FORCING INPUT]")
#         print(f"  Visual token positions:  3–578  (576 CLIP patches)")
#         print(f"  Text token positions:    ~{desc_start}–{desc_start + n_text_tokens - 1}  "
#               f"({n_text_tokens} generated tokens)")
#         print(f"\n  Input sequence order:")
#         print(f"    [BOS] → USER: → [576 image patches] → "
#               f"\\nCould you describe the image?\\nASSISTANT: → "
#               f"[{n_text_tokens} generated tokens]")

#         results[img_id] = {                                               # store results for this image
#             "label": label,
#             "token_ids": result['token_ids'],
#             "tokens": tokens,
#             "text": text,
#         }

#     # ── Save results ─────────────────────────────────────────
#     output_path = args.output_path.replace('.json', f'_fig3_{args.model_type}.json')  # append backend name to filename
#     os.makedirs(os.path.dirname(output_path), exist_ok=True)              # create output directory if needed
#     with open(output_path, 'w') as f:
#         json.dump(results, f, indent=2)                                   # write JSON with pretty formatting

#     print(f"\n{'═'*70}")
#     print(f"Saved results to {output_path}")
#     print(f"\nSummary:")
#     for img_id, r in results.items():                                     # print summary for each image
#         print(f"  {r['label']}: {len(r['tokens'])} tokens, "
#               f"{len(r['text'].split())} words")

#     # ── Free GPU memory ──────────────────────────────────────
#     if args.model_type == 'llava-hf':
#         del model, processor                                              # delete HF model references
#     elif args.model_type == 'llava-liuhaotian':
#         del model, tokenizer, image_proc                                  # delete liuhaotian model references
#     elif args.model_type == 'internvl':
#         del model, tokenizer                                              # delete InternVL model references
#     else:  # qwen2vl
#         del model, processor                                              # delete Qwen2.5-VL model references
#     torch.cuda.empty_cache()                                              # release GPU memory back to CUDA


# # ═══════════════════════════════════════════════════════════════════
# # Main
# # ═══════════════════════════════════════════════════════════════════

# def main():
#     args = parse_args()                                                   # parse command-line arguments

#     # ── Test mode: run on Figure 3 images only ────────────
#     if args.test_fig3:                                                    # if --test_fig3 flag was passed
#         run_test_fig3(args)                                               # run test on 6 images and exit
#         return

#     device = f'cuda:{args.device}' if args.device.isdigit() else args.device  # e.g. "0" → "cuda:0", "cpu" → "cpu"

#     # ─────────────────────────────────────────────────────────
#     # Load image list from detail_23k.json
#     # ─────────────────────────────────────────────────────────
#     print(f'Loading image list from {args.detail_23k_path} ...')
#     with open(args.detail_23k_path) as f:
#         detail_data = json.load(f)                                        # load the JSON list of image entries

#     image_list = []
#     for item in detail_data:
#         img_id = item['id']                                               # e.g. "000000323964"
#         img_path = os.path.join(args.coco_img_dir,
#                                 os.path.basename(item['image']))          # full path to COCO image
#         image_list.append((img_id, img_path))                             # list of (id, path) tuples

#     print(f'Total images in detail_23k: {len(image_list)}')

#     # Apply sharding — split image list by index range
#     end_idx = args.end_idx if args.end_idx is not None else len(image_list)  # default to all images
#     image_list = image_list[args.start_idx:end_idx]                       # slice to requested range
#     print(f'Processing indices {args.start_idx} to {end_idx} '
#           f'({len(image_list)} images)')

#     # ─────────────────────────────────────────────────────────
#     # Load model
#     # ─────────────────────────────────────────────────────────
#     if args.model_type == 'llava-hf':
#         print(f'Loading HF model: {args.hf_id} ...')
#         model, processor = load_model_hf(args.hf_id, device)             # HF: model + processor
#         prompt = build_prompt_hf()                                        # HF prompt (no system prompt)
#         tokenizer = None                                                  # not used — processor wraps it
#         image_proc = None                                                 # not used — processor wraps it
#     elif args.model_type == 'llava-liuhaotian':
#         print(f'Loading original model: {args.original_model_path} ...')
#         model, tokenizer, image_proc = load_model_original(              # liuhaotian: model + tokenizer + image_processor
#             args.original_model_path, device)
#         prompt = build_prompt_original()                                  # v1 template prompt (with system prompt)
#         processor = None                                                  # not used — separate tokenizer/image_proc
#     elif args.model_type == 'internvl':
#         print(f'Loading InternVL model: {args.internvl_path} ...')
#         model, tokenizer = load_model_internvl(args.internvl_path, device)  # InternVL: model + tokenizer
#         prompt = 'Could you describe the image?'                          # plain-text question for model.chat()
#         processor  = None                                                 # not used — InternVL has its own chat API
#         image_proc = None                                                 # not used — preprocessing is inline
#     else:  # qwen2vl
#         print(f'Loading Qwen2.5-VL model: {args.qwen2vl_path} ...')
#         model, processor = load_model_qwen2vl(args.qwen2vl_path, device)  # Qwen2.5-VL: model + processor
#         prompt = 'Could you describe the image?'                          # plain-text question; template applied in generate
#         tokenizer  = None                                                 # not used — processor wraps it
#         image_proc = None                                                 # not used — processor wraps it

#     print(f'Model loaded. Prompt: {repr(prompt[:80])}...')

#     # ─────────────────────────────────────────────────────────
#     # Generate descriptions
#     # ─────────────────────────────────────────────────────────
#     # Resume support: load existing output if present, skip done images
#     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)         # create output directory if needed
#     if os.path.exists(args.output_path):                                  # check if output file already exists
#         with open(args.output_path) as f:
#             descriptions = json.load(f)                                   # load previously generated descriptions
#         print(f'Resuming: loaded {len(descriptions)} existing descriptions')
#     else:
#         descriptions = {}                                                 # start fresh

#     skipped = 0
#     t0 = time.time()                                                      # start timer

#     for img_id, img_path in tqdm(image_list, desc='Generating'):          # loop over images with progress bar
#         # Skip already generated (resume support)
#         if img_id in descriptions:                                        # already processed in a previous run
#             continue

#         # Load image
#         try:
#             img = Image.open(img_path).convert('RGB')                     # open and convert to RGB
#         except Exception as e:
#             tqdm.write(f'Skip {img_path}: {e}')                           # log error without breaking progress bar
#             skipped += 1
#             continue

#         # Generate description — dispatch to correct backend
#         if args.model_type == 'llava-hf':
#             generated_text = generate_description_hf(                     # HF generation path
#                 model, processor, prompt, img, device,
#                 args.max_new_tokens, args.min_new_tokens)
#         elif args.model_type == 'llava-liuhaotian':
#             generated_text = generate_description_original(               # liuhaotian generation path
#                 model, tokenizer, image_proc, prompt, img, device,
#                 args.max_new_tokens, args.min_new_tokens)
#         elif args.model_type == 'internvl':
#             generated_text = generate_description_internvl(               # InternVL generation path
#                 model, tokenizer, img, device,
#                 args.max_new_tokens, args.min_new_tokens)
#         else:  # qwen2vl
#             generated_text = generate_description_qwen2vl(                # Qwen2.5-VL generation path
#                 model, processor, img, device,
#                 args.max_new_tokens, args.min_new_tokens)

#         descriptions[img_id] = generated_text                             # store result keyed by image ID

#         # Save periodically to survive preemption (every 50 images)
#         if len(descriptions) % 50 == 0:
#             with open(args.output_path, 'w') as f:
#                 json.dump(descriptions, f, indent=2)                      # checkpoint save

#     elapsed = time.time() - t0                                            # total generation time

#     # ─────────────────────────────────────────────────────────
#     # Save
#     # ─────────────────────────────────────────────────────────
#     with open(args.output_path, 'w') as f:
#         json.dump(descriptions, f, indent=2)                              # final save of all descriptions

#     print(f'\nDone in {elapsed/60:.1f} min')
#     print(f'Generated: {len(descriptions)}, Skipped: {skipped}')
#     print(f'Saved to {args.output_path}')

#     # Print a few examples for sanity checking
#     print('\n--- Sample descriptions ---')
#     for i, (img_id, desc) in enumerate(descriptions.items()):
#         if i >= 3:                                                        # only show first 3
#             break
#         print(f'\n{img_id}: {desc["text"][:150]}...')                     # first 150 chars of text
#         print(f'  tokens ({len(desc["token_ids"])}): {desc["tokens"][:10]}...')  # first 10 tokens


# if __name__ == '__main__':
#     main()



# # """
# # generate_descriptions_both_backbone.py — Step 0 of Xu et al. classification

# # Generate detailed image descriptions using LLaVA-1.5-7B for all images
# # in the detail_23k subset. These descriptions serve as the text tokens
# # for neuron classification.

# # Supports two backends:
# #   --model_type llava-hf         : HuggingFace llava-hf/llava-1.5-7b-hf
# #   --model_type llava-liuhaotian : Original liuhaotian/llava-v1.5-7b via LLaVA repo
# #                             (requires transformers==4.37.2)

# # Output: generated_descriptions.json — {image_id: {"token_ids": [...], "tokens": [...], "text": "..."}}

# # Usage:
# #     # Full dataset with HF backend
# #     python generate_descriptions_both_backbone.py --model_type llava-hf --start_idx 0 --end_idx 23000

# #     # Full dataset with liuhaotian backend
# #     python generate_descriptions_both_backbone.py --model_type llava-liuhaotian --start_idx 0 --end_idx 23000

# #     # Test on Figure 3 images (6 images, shows token/word format + stage 1 input layout)
# #     python generate_descriptions_both_backbone.py --model_type llava-hf --test_fig3
# #     python generate_descriptions_both_backbone.py --model_type llava-liuhaotian --test_fig3
# # """

# # import argparse                                                           # command-line argument parsing
# # import json                                                               # reading/writing JSON files
# # import os                                                                 # file path manipulation
# # import sys                                                                # modifying the Python import path
# # import time                                                               # timing the generation loop

# # # Add LLaVA repo to path so we can import from llava.* (needed for liuhaotian backend)
# # _PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))  # two levels up from this script
# # _LLAVA_PATH = os.path.join(_PROJECT_ROOT, 'LLaVA')                       # path to cloned LLaVA repo
# # if _LLAVA_PATH not in sys.path:                                           # avoid duplicate entries
# #     sys.path.insert(0, _LLAVA_PATH)                                       # insert at front so llava imports resolve here first

# # import torch                                                              # PyTorch — tensor operations, GPU, inference
# # from PIL import Image                                                     # Pillow — loading image files
# # from tqdm import tqdm                                                     # progress bar for the generation loop


# # def parse_args():
# #     """Parse command-line arguments for dataset paths, generation settings, and sharding."""
# #     p = argparse.ArgumentParser(
# #         description='Generate LLaVA descriptions for detail_23k images (both backends)')

# #     # Backend selection
# #     p.add_argument('--model_type', default='llava-hf', choices=['llava-hf', 'llava-liuhaotian'],
# #                    help='"llava-hf" for llava-hf/llava-1.5-7b-hf, '
# #                         '"llava-liuhaotian" for original llava-v1.5-7b')        # which backend to use

# #     # Model paths
# #     p.add_argument('--hf_id', default='llava-hf/llava-1.5-7b-hf',
# #                    help='HuggingFace model ID (for --model_type llava-hf)')     # HF Hub ID
# #     p.add_argument('--original_model_path', default='liuhaotian/llava-v1.5-7b',
# #                    help='Original LLaVA model path (for --model_type llava-liuhaotian)')  # HF Hub ID or local path

# #     # Data paths
# #     _script_dir = os.path.dirname(os.path.abspath(__file__))              # directory containing this script
# #     _project_root = os.path.abspath(os.path.join(_script_dir, '..', '..'))  # project root (two levels up)
# #     p.add_argument('--detail_23k_path',
# #                    default=os.path.join(_project_root, 'detail_23k.json'),
# #                    help='Path to detail_23k.json (defines which images to use)')
# #     p.add_argument('--coco_img_dir',
# #                    default='/home/projects/bagon/shared/coco2017/images/train2017/',
# #                    help='Path to COCO train2017 images')

# #     # Output
# #     p.add_argument('--output_path',
# #                    default='generated_descriptions/generated_descriptions.json',
# #                    help='Output JSON path')

# #     # Sharding — allows splitting across multiple GPUs/jobs
# #     p.add_argument('--start_idx', type=int, default=0,
# #                    help='Start index in detail_23k list')
# #     p.add_argument('--end_idx', type=int, default=None,
# #                    help='End index in detail_23k list (None = all)')

# #     # Generation parameters
# #     p.add_argument('--min_new_tokens', type=int, default=100,
# #                    help='Min tokens to generate per image')               # forces minimum description length
# #     p.add_argument('--max_new_tokens', type=int, default=550,
# #                    help='Max tokens to generate per image')               # caps maximum description length
# #     p.add_argument('--device', default='0')                               # CUDA GPU index

# #     # Test mode
# #     p.add_argument('--test_fig3', action='store_true',
# #                    help='Run on 6 Figure 3 images only. Shows token/word format '
# #                         'and stage 1 teacher-forcing input layout.')

# #     return p.parse_args()


# # # ═══════════════════════════════════════════════════════════════════
# # # Model loading
# # # ═══════════════════════════════════════════════════════════════════

# # def load_model_hf(hf_id, device):
# #     """Load the HuggingFace LLaVA model.

# #     Uses LlavaForConditionalGeneration from transformers — the HF port
# #     of LLaVA-1.5 that bundles vision encoder + LLM in one model class.

# #     Returns: (model, processor)
# #         - model: LlavaForConditionalGeneration (vision + LLM)
# #         - processor: AutoProcessor (handles both text tokenisation + image preprocessing)
# #     """
# #     from transformers import AutoProcessor, LlavaForConditionalGeneration  # HF classes for LLaVA

# #     processor = AutoProcessor.from_pretrained(hf_id)                      # loads tokenizer + CLIP image processor
# #     model = LlavaForConditionalGeneration.from_pretrained(
# #         hf_id, torch_dtype=torch.float16, low_cpu_mem_usage=True          # float16 to save GPU memory
# #     ).to(device).eval()                                                   # move to GPU, set eval mode

# #     return model, processor


# # def load_model_original(model_path, device):
# #     """Load the original LLaVA model via the cloned LLaVA repo.

# #     Requires the LLaVA repo on sys.path and transformers==4.37.2.

# #     Returns: (model, tokenizer, image_processor)
# #         - model: the LLaVA model (LLaMA + CLIP vision encoder + projection)
# #         - tokenizer: the LLaMA SentencePiece tokenizer
# #         - image_processor: CLIP image processor (resizes/normalises images)
# #     """
# #     from llava.model.builder import load_pretrained_model                 # loads model weights from HF Hub or local
# #     from llava.mm_utils import get_model_name_from_path                   # derives model name string from path

# #     model_name = get_model_name_from_path(model_path)                     # e.g. "llava-v1.5-7b"
# #     tokenizer, model, image_processor, context_len = load_pretrained_model(
# #         model_path, None, model_name, device_map=device,                  # None = no model_base (not a LoRA model)
# #         torch_dtype=torch.float16                                         # half precision to save GPU memory
# #     )
# #     model.eval()                                                          # set to eval mode (disables dropout etc.)


# #     return model, tokenizer, image_processor


# # # ═══════════════════════════════════════════════════════════════════
# # # Prompt building
# # # ═══════════════════════════════════════════════════════════════════

# # def build_prompt_hf():
# #     """Build generation prompt for the HF backend.

# #     HF LLaVA was ported without the system prompt, so the prompt is:
# #         USER: <image>\\nCould you describe the image?\\nASSISTANT:

# #     Returns the formatted prompt string.
# #     """
# #     return "USER: <image>\nCould you describe the image?\nASSISTANT:"    # no system prompt, \\n separators


# # def build_prompt_original():
# #     """Build generation prompt for the liuhaotian backend using conv_templates["v1"].

# #     The v1 template (conv_vicuna_v1) formats the conversation as:
# #         {system_prompt} USER: <image>\\nCould you describe the image? ASSISTANT:

# #     Returns the formatted prompt string.
# #     """
# #     from llava.conversation import conv_templates                         # conversation template registry

# #     conv = conv_templates["v1"].copy()                                    # copy to avoid mutating the global template
# #     conv.append_message(conv.roles[0], "<image>\nCould you describe the image?")  # USER turn with <image> placeholder
# #     conv.append_message(conv.roles[1], None)                              # empty ASSISTANT turn (model will generate)
# #     return conv.get_prompt()                                              # serialise into formatted string


# # # ═══════════════════════════════════════════════════════════════════
# # # Generation
# # # ═══════════════════════════════════════════════════════════════════

# # def generate_description_hf(model, processor, prompt, img, device, max_new_tokens, min_new_tokens):
# #     """Generate a text description for a single image using the HF LLaVA model.

# #     Steps:
# #       1. Use processor to tokenise prompt + preprocess image together
# #       2. Run greedy decoding with model.generate()
# #       3. Extract only the NEW tokens (strip the prompt tokens)
# #       4. Decode into three formats

# #     Returns dict with three keys:
# #         token_ids    : list[int]   — raw token IDs of generated text only
# #         tokens       : list[str]   — subword strings like ["▁d","ough","n","uts"]
# #         text         : str         — human-readable "doughnuts..."
# #     """
# #     # Process text + image together via the HF processor
# #     inputs = processor(                                                   # AutoProcessor handles both modalities
# #         images=img, text=prompt, return_tensors='pt'                      # returns dict with input_ids, pixel_values, etc.
# #     ).to(device)                                                          # move all tensors to GPU

# #     prompt_len = inputs['input_ids'].shape[1]                             # number of tokens in the prompt (including expanded <image>)

# #     with torch.no_grad():                                                 # disable gradient computation (inference only)
# #         output_ids = model.generate(
# #             **inputs,                                                     # unpack input_ids, pixel_values, attention_mask
# #             max_new_tokens=max_new_tokens,                                # maximum number of tokens to generate
# #             min_new_tokens=min_new_tokens,                                # force minimum description length
# #             do_sample=False,                                              # greedy decoding (always pick highest prob token)
# #         )

# #     # HF generate() returns full sequence (prompt + generated), so strip prompt
# #     generated_ids = output_ids[0, prompt_len:]                            # slice off prompt tokens → only new tokens

# #     # Strip EOS token if present at the end
# #     tokenizer = processor.tokenizer                                       # access the underlying tokenizer
# #     if len(generated_ids) > 0 and generated_ids[-1].item() == tokenizer.eos_token_id:  # check last token
# #         generated_ids = generated_ids[:-1]                                # remove EOS

# #     # Three output formats:
# #     token_ids = generated_ids.tolist()                                    # convert tensor to plain Python list of ints
# #     tokens = [tokenizer.decode(tid) for tid in token_ids]                 # decode each token ID individually → subword strings

# #     # Build readable text
# #     text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()  # decode all IDs into one string

# #     # Strip common prefixes (model sometimes starts with a polite preamble)
# #     for prefix in ['Yes, I can describe the image.', 'Yes, I can describe the image',
# #                     'Yes,', 'Yes.']:
# #         if text.startswith(prefix):                                       # check if text starts with this prefix
# #             text = text[len(prefix):].strip()                             # remove prefix and trim whitespace
# #             break                                                         # only strip the first matching prefix

# #     # Strip matching prefix from token lists to keep them aligned with the cleaned text
# #     clean_tokens = [t.lstrip('▁').strip() for t in tokens]                # strip ▁ prefix for matching
# #     PREFIX_TOKENS = [                                                     # known prefix patterns
# #         ['Yes', ',', 'I', 'can', 'describe', 'the', 'image', '.'],       # "Yes, I can describe the image."
# #         ['Yes', ',', 'I', 'can', 'see', 'the', 'image', '.'],            # "Yes, I can see the image."
# #         ['Yes', ','],                                                     # "Yes,"
# #         ['Yes', '.'],                                                     # "Yes."
# #     ]
# #     for prefix_toks in PREFIX_TOKENS:
# #         n = len(prefix_toks)                                              # number of tokens in this prefix pattern
# #         if clean_tokens[:n] == prefix_toks:                               # match against clean_tokens (no ▁)
# #             token_ids = token_ids[n:]                                     # strip same count from all three lists
# #             tokens = tokens[n:]
# #             clean_tokens = clean_tokens[n:]
# #             break                                                         # only strip one prefix

# #     return {"token_ids": token_ids, "tokens": tokens, "text": text}


# # def generate_description_original(model, tokenizer, image_processor, prompt, img, device, max_new_tokens, min_new_tokens):
# #     """Generate a text description for a single image using the original LLaVA model.

# #     Steps:
# #       1. Tokenise the prompt, replacing <image> with IMAGE_TOKEN_INDEX (-200)
# #       2. Preprocess the image using CLIP's image processor (resize to 336x336, normalise)
# #       3. Run greedy decoding with model.generate()
# #       4. Decode the generated token IDs into three formats

# #     Returns dict with three keys:
# #         token_ids    : list[int]   — raw token IDs (used in step 2 classification)
# #         tokens       : list[str]   — subword strings like ["▁d","ough","n","uts"]
# #         text         : str         — human-readable "doughnuts..."
# #     """
# #     from llava.constants import IMAGE_TOKEN_INDEX                         # special token ID (-200) for <image>
# #     from llava.mm_utils import tokenizer_image_token                      # tokenise text, replacing <image> with special ID

# #     # Tokenise the prompt, replacing <image> text with IMAGE_TOKEN_INDEX (-200)
# #     input_ids = tokenizer_image_token(
# #         prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'         # returns a PyTorch tensor
# #     ).unsqueeze(0).to(device)                                             # add batch dim → shape (1, seq_len), move to GPU

# #     # Preprocess image using CLIP processor (resize to 336x336, normalise pixel values)
# #     image_tensor = image_processor.preprocess(
# #         img, return_tensors='pt'                                          # returns dict with 'pixel_values' tensor
# #     )['pixel_values'].half().to(device)                                   # convert to float16, move to GPU → (1, 3, 336, 336)

# #     with torch.no_grad():                                                 # disable gradient computation (inference only)
# #         output_ids = model.generate(
# #             input_ids,                                                    # tokenised prompt with IMAGE_TOKEN_INDEX
# #             images=image_tensor,                                          # preprocessed image passed as keyword arg
# #             max_new_tokens=max_new_tokens,                                # maximum number of tokens to generate
# #             min_new_tokens=min_new_tokens,                                # force minimum description length
# #             do_sample=False,                                              # greedy decoding (always pick highest prob token)
# #         )

# #     # transformers 4.37.2: generate() returns full sequence (prompt + generated)
# #     generated_ids = output_ids[0]                                         # generate() returns only new tokens for original LLaVA model

# #     # Three output formats:
# #     token_ids = generated_ids.tolist()                                    # convert tensor to plain Python list of ints
# #     tokens = [tokenizer.decode(tid) for tid in token_ids]                 # decode each token ID individually → subword strings

# #     # Build readable text: tokenizer.decode() works correctly in transformers 4.37.2
# #     text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()  # decode all IDs into one string, remove <s>/</s>

# #     # Strip common prefixes from v1 conv template (model sometimes starts with a polite preamble)
# #     for prefix in ['Yes, I can describe the image.', 'Yes, I can describe the image',
# #                     'Yes,', 'Yes.']:
# #         if text.startswith(prefix):                                       # check if text starts with this prefix
# #             text = text[len(prefix):].strip()                             # remove prefix and trim whitespace
# #             break                                                         # only strip the first matching prefix

# #     # Strip matching prefix from token lists to keep them aligned with the cleaned text
# #     clean_tokens = [t.lstrip('▁').strip() for t in tokens]                # strip ▁ prefix for matching
# #     PREFIX_TOKENS = [                                                     # known v1 template prefix patterns
# #         ['Yes', ',', 'I', 'can', 'describe', 'the', 'image', '.'],       # "Yes, I can describe the image."
# #         ['Yes', ',', 'I', 'can', 'see', 'the', 'image', '.'],            # "Yes, I can see the image."
# #         ['Yes', ','],                                                     # "Yes,"
# #         ['Yes', '.'],                                                     # "Yes."
# #     ]
# #     for prefix_toks in PREFIX_TOKENS:
# #         n = len(prefix_toks)                                              # number of tokens in this prefix pattern
# #         if clean_tokens[:n] == prefix_toks:                               # match against clean_tokens (no ▁)
# #             token_ids = token_ids[n:]                                     # strip same count from all three lists
# #             tokens = tokens[n:]
# #             clean_tokens = clean_tokens[n:]
# #             break                                                         # only strip one prefix

# #     return {"token_ids": token_ids, "tokens": tokens, "text": text}


# # # ═══════════════════════════════════════════════════════════════════
# # # Figure 3 test — 6 images, shows token/word format + stage 1 input
# # # ═══════════════════════════════════════════════════════════════════

# # FIG3_IMAGES = {                                                           # COCO image IDs from Xu et al. Figure 3
# #     "000000403170": "(a) Visual neuron - airplane/motorcycles",
# #     "000000065793": "(b) Text neuron - teddy bears",
# #     "000000156852": "(c) Multi-modal neuron - kitchen/thumbs up",
# #     "000000323964": "(d) Multi-modal neuron - doughnuts",
# #     "000000276332": "(e) Multi-modal neuron - zebras",
# #     "000000060034": "(f) Multi-modal neuron - fire hydrant/pigeons",
# # }


# # def run_test_fig3(args):
# #     """Generate descriptions for the 6 Figure 3 images.

# #     For each image, prints:
# #       1. Word format  — human-readable decoded text
# #       2. Token format — subword pieces as the tokenizer produces them
# #       3. Clean token format — subword pieces with ▁ stripped
# #       4. Stage 1 input layout — exact sequence fed into LLaVA for
# #          teacher-forcing activation recording:

# #          [BOS] USER: <576 image patches> \\nCould you describe the image?
# #          \\nASSISTANT: <generated_tok_1> <generated_tok_2> ... [EOS]

# #          Positions 3–578   → "visual tokens"  (576 CLIP patches)
# #          Positions after ASSISTANT: → "text tokens" (generated description)
# #     """
# #     device = f'cuda:{args.device}' if args.device.isdigit() else args.device  # e.g. "0" → "cuda:0"

# #     backend_name = args.model_type.upper()                                # "HF" or "LIUHAOTIAN" for display
# #     print(f"\n{'█'*70}")
# #     print(f"  Running {backend_name} backend")
# #     print(f"{'█'*70}")

# #     # ── Load model ──────────────────────────────────────────
# #     if args.model_type == 'llava-hf':
# #         print(f'Loading HF model: {args.hf_id} ...')
# #         model, processor = load_model_hf(args.hf_id, device)             # returns model + processor
# #         prompt = build_prompt_hf()                                        # HF prompt (no system prompt)
# #     else:
# #         print(f'Loading original model: {args.original_model_path} ...')
# #         model, tokenizer, image_proc = load_model_original(              # returns model, tokenizer, image_processor
# #             args.original_model_path, device)
# #         prompt = build_prompt_original()                                  # v1 template prompt (with system prompt)

#     prompt = prompt if 'prompt' in dir() else '(embedded in model.chat API)'
# #     print(f'Prompt: {repr(prompt)}')                                      # show exact prompt with escape chars visible
# #     results = {}

# #     for img_id, label in FIG3_IMAGES.items():                             # iterate over the 6 Figure 3 images
# #         print(f"\n{'═'*70}")
# #         print(f"{label} — COCO ID: {img_id}")
# #         print('═'*70)

# #         img_path = os.path.join(args.coco_img_dir, f"{img_id}.jpg")       # construct full path to COCO image
# #         img = Image.open(img_path).convert('RGB')                         # load image and convert to RGB

# #         # ── Generate description ────────────────────────────
# #         if args.model_type == 'llava-hf':
# #             result = generate_description_hf(                             # HF generation path
# #                 model, processor, prompt, img, device,
# #                 args.max_new_tokens, args.min_new_tokens)
# #         else:
# #             result = generate_description_original(                       # liuhaotian generation path
# #                 model, tokenizer, image_proc, prompt, img, device,
# #                 args.max_new_tokens, args.min_new_tokens)

# #         tokens = result['tokens']                                         # list of subword strings (with ▁)
# #         clean_tokens = [t.lstrip('▁').strip() for t in tokens]            # stripped version for display
# #         text = result['text']                                             # human-readable text

# #         print(f"\n[WORD FORMAT] ({len(text.split())} words):")
# #         print(f"  {text}")
# #         print(f"\n[TOKEN FORMAT] ({len(tokens)} tokens):")
# #         print(f"  {' '.join(tokens)}")
# #         print(f"\n[CLEAN TOKEN FORMAT] ({len(clean_tokens)} tokens):")
# #         print(f"  {' '.join(clean_tokens)}")

# #         # ── Stage 1: Show teacher-forcing input layout ──────
# #         # Approximate layout based on known LLaVA-1.5 structure
# #         desc_start = 10 + 576                                             # ~10 prompt tokens + 576 CLIP patches
# #         n_text_tokens = len(tokens)                                       # number of generated description tokens

# #         print(f"\n[STAGE 1 — TEACHER FORCING INPUT]")
# #         print(f"  Visual token positions:  3–578  (576 CLIP patches)")
# #         print(f"  Text token positions:    ~{desc_start}–{desc_start + n_text_tokens - 1}  "
# #               f"({n_text_tokens} generated tokens)")
# #         print(f"\n  Input sequence order:")
# #         print(f"    [BOS] → USER: → [576 image patches] → "
# #               f"\\nCould you describe the image?\\nASSISTANT: → "
# #               f"[{n_text_tokens} generated tokens]")

# #         results[img_id] = {                                               # store results for this image
# #             "label": label,
# #             "token_ids": result['token_ids'],
# #             "tokens": tokens,
# #             "text": text,
# #         }

# #     # ── Save results ─────────────────────────────────────────
# #     output_path = args.output_path.replace('.json', f'_fig3_{args.model_type}.json')  # append backend name to filename
# #     os.makedirs(os.path.dirname(output_path), exist_ok=True)              # create output directory if needed
# #     with open(output_path, 'w') as f:
# #         json.dump(results, f, indent=2)                                   # write JSON with pretty formatting

# #     print(f"\n{'═'*70}")
# #     print(f"Saved results to {output_path}")
# #     print(f"\nSummary:")
# #     for img_id, r in results.items():                                     # print summary for each image
# #         print(f"  {r['label']}: {len(r['tokens'])} tokens, "
# #               f"{len(r['text'].split())} words")

# #     # ── Free GPU memory ──────────────────────────────────────
# #     if args.model_type == 'llava-hf':
# #         del model, processor                                              # delete HF model references
# #     else:
# #         del model, tokenizer, image_proc                                  # delete liuhaotian model references
# #     torch.cuda.empty_cache()                                              # release GPU memory back to CUDA


# # # ═══════════════════════════════════════════════════════════════════
# # # Main
# # # ═══════════════════════════════════════════════════════════════════

# # def main():
# #     args = parse_args()                                                   # parse command-line arguments

# #     # ── Test mode: run on Figure 3 images only ────────────
# #     if args.test_fig3:                                                    # if --test_fig3 flag was passed
# #         run_test_fig3(args)                                               # run test on 6 images and exit
# #         return

# #     device = f'cuda:{args.device}' if args.device.isdigit() else args.device  # e.g. "0" → "cuda:0", "cpu" → "cpu"

# #     # ─────────────────────────────────────────────────────────
# #     # Load image list from detail_23k.json
# #     # ─────────────────────────────────────────────────────────
# #     print(f'Loading image list from {args.detail_23k_path} ...')
# #     with open(args.detail_23k_path) as f:
# #         detail_data = json.load(f)                                        # load the JSON list of image entries

# #     image_list = []
# #     for item in detail_data:
# #         img_id = item['id']                                               # e.g. "000000323964"
# #         img_path = os.path.join(args.coco_img_dir,
# #                                 os.path.basename(item['image']))          # full path to COCO image
# #         image_list.append((img_id, img_path))                             # list of (id, path) tuples

# #     print(f'Total images in detail_23k: {len(image_list)}')

# #     # Apply sharding — split image list by index range
# #     end_idx = args.end_idx if args.end_idx is not None else len(image_list)  # default to all images
# #     image_list = image_list[args.start_idx:end_idx]                       # slice to requested range
# #     print(f'Processing indices {args.start_idx} to {end_idx} '
# #           f'({len(image_list)} images)')

# #     # ─────────────────────────────────────────────────────────
# #     # Load model
# #     # ─────────────────────────────────────────────────────────
# #     if args.model_type == 'llava-hf':
# #         print(f'Loading HF model: {args.hf_id} ...')
# #         model, processor = load_model_hf(args.hf_id, device)             # HF: model + processor
# #         prompt = build_prompt_hf()                                        # HF prompt (no system prompt)
# #         tokenizer = None                                                  # not used — processor wraps it
# #         image_proc = None                                                 # not used — processor wraps it
# #     else:
# #         print(f'Loading original model: {args.original_model_path} ...')
# #         model, tokenizer, image_proc = load_model_original(              # liuhaotian: model + tokenizer + image_processor
# #             args.original_model_path, device)
# #         prompt = build_prompt_original()                                  # v1 template prompt (with system prompt)
# #         processor = None                                                  # not used — separate tokenizer/image_proc

# #     print(f'Model loaded. Prompt: {repr(prompt[:80])}...')

# #     # ─────────────────────────────────────────────────────────
# #     # Generate descriptions
# #     # ─────────────────────────────────────────────────────────
# #     # Resume support: load existing output if present, skip done images
# #     os.makedirs(os.path.dirname(args.output_path), exist_ok=True)         # create output directory if needed
# #     if os.path.exists(args.output_path):                                  # check if output file already exists
# #         with open(args.output_path) as f:
# #             descriptions = json.load(f)                                   # load previously generated descriptions
# #         print(f'Resuming: loaded {len(descriptions)} existing descriptions')
# #     else:
# #         descriptions = {}                                                 # start fresh

# #     skipped = 0
# #     t0 = time.time()                                                      # start timer

# #     for img_id, img_path in tqdm(image_list, desc='Generating'):          # loop over images with progress bar
# #         # Skip already generated (resume support)
# #         if img_id in descriptions:                                        # already processed in a previous run
# #             continue

# #         # Load image
# #         try:
# #             img = Image.open(img_path).convert('RGB')                     # open and convert to RGB
# #         except Exception as e:
# #             tqdm.write(f'Skip {img_path}: {e}')                           # log error without breaking progress bar
# #             skipped += 1
# #             continue

# #         # Generate description — dispatch to correct backend
# #         if args.model_type == 'llava-hf':
# #             generated_text = generate_description_hf(                     # HF generation path
# #                 model, processor, prompt, img, device,
# #                 args.max_new_tokens, args.min_new_tokens)
# #         else:
# #             generated_text = generate_description_original(               # liuhaotian generation path
# #                 model, tokenizer, image_proc, prompt, img, device,
# #                 args.max_new_tokens, args.min_new_tokens)

# #         descriptions[img_id] = generated_text                             # store result keyed by image ID

# #         # Save periodically to survive preemption (every 50 images)
# #         if len(descriptions) % 50 == 0:
# #             with open(args.output_path, 'w') as f:
# #                 json.dump(descriptions, f, indent=2)                      # checkpoint save

# #     elapsed = time.time() - t0                                            # total generation time

# #     # ─────────────────────────────────────────────────────────
# #     # Save
# #     # ─────────────────────────────────────────────────────────
# #     with open(args.output_path, 'w') as f:
# #         json.dump(descriptions, f, indent=2)                              # final save of all descriptions

# #     print(f'\nDone in {elapsed/60:.1f} min')
# #     print(f'Generated: {len(descriptions)}, Skipped: {skipped}')
# #     print(f'Saved to {args.output_path}')

# #     # Print a few examples for sanity checking
# #     print('\n--- Sample descriptions ---')
# #     for i, (img_id, desc) in enumerate(descriptions.items()):
# #         if i >= 3:                                                        # only show first 3
# #             break
# #         print(f'\n{img_id}: {desc["text"][:150]}...')                     # first 150 chars of text
# #         print(f'  tokens ({len(desc["token_ids"])}): {desc["tokens"][:10]}...')  # first 10 tokens


# # if __name__ == '__main__':
# #     main()