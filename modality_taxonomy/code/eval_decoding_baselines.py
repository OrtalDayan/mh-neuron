#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_decoding_baselines.py — Evaluate three hallucination-mitigation baselines
under a single, unified contrastive-decoding interface, plugged into
run_pipeline.sh's eval conventions.

Baselines
---------
[1] VCD — Visual Contrastive Decoding (CVPR'24)
        weak = forward pass on a *noised* image (DDPM-style Gaussian noise)
[2] ICD — Instruction Contrastive Decoding (ACL'24 Findings)
        weak = forward pass with a *disturbance* prefix injected into the prompt
[3] SID — Self-Introspective Decoding (ICLR'25)
        weak = forward pass keeping only the *least important* vision tokens,
                ranked by the model's own early-layer attention

All three reduce to the same contrastive update:
        logits_final = (1 + alpha) * logits_strong  -  alpha * logits_weak
                       restricted to tokens with  p_strong > beta * max(p_strong)
                       (adaptive plausibility constraint, from VCD §3.3)

The script matches code/eval_chair.py's CLI surface so it slots into STEP 19
(`bash run_pipeline.sh --step baselines --method vcd` — see launcher at bottom).

Currently implemented for: llava-ov  (HF AutoModelForImageTextToText / Vision2Seq).
Other backends raise NotImplementedError with a clear hint.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


# =============================================================================
# Section 1 — Image perturbations (used by VCD's "weak" pass)
# =============================================================================

def diffusion_noise(image: Image.Image, t: int = 500, T: int = 1000,
                    seed: Optional[int] = None) -> Image.Image:
    """Add DDPM forward-diffusion noise at timestep `t` (of total `T`).

    Implements x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε with a *linear* β
    schedule (the VCD paper uses this; cosine is a drop-in alt). At t=500/1000,
    the image is mostly destroyed — exactly the regime VCD relies on so that
    `logits_weak` reflects "language prior given a meaningless image".
    """
    if seed is not None:                                  # reproducibility hook
        torch.manual_seed(seed)

    x0 = torch.from_numpy(                                # H,W,3 uint8 → 3,H,W float in [-1,1]
        __import__("numpy").asarray(image.convert("RGB"))
    ).permute(2, 0, 1).float() / 127.5 - 1.0

    betas = torch.linspace(1e-4, 0.02, T)                 # DDPM default linear β
    alphas = 1.0 - betas                                  # α_t = 1 - β_t
    alpha_bar = torch.cumprod(alphas, dim=0)              # ᾱ_t = ∏_{s≤t} α_s
    a_bar_t = alpha_bar[t - 1]                            # 1-indexed convention

    eps = torch.randn_like(x0)                            # standard Gaussian noise
    xt = a_bar_t.sqrt() * x0 + (1.0 - a_bar_t).sqrt() * eps  # forward-diffusion step

    xt = ((xt.clamp(-1, 1) + 1.0) * 127.5).byte()         # back to uint8 image space
    return Image.fromarray(xt.permute(1, 2, 0).numpy(), mode="RGB")


# =============================================================================
# Section 2 — Backbone wrapper (only abstracts what we need)
# =============================================================================

@dataclass
class VLMBackbone:
    """Thin wrapper so the contrastive loop is model-agnostic.

    Holds the HF model + processor and exposes two operations the decoders need:
      * build_inputs(image, prompt) -> dict of tensors on device
      * one-shot forward(input_ids, ...) returning (logits, past_key_values)
    """
    model: torch.nn.Module                                # HF causal-LM with vision tower
    processor: object                                     # HF Processor (tokenizer + image_processor)
    device: torch.device                                  # cuda / cpu — set in load()
    dtype: torch.dtype                                    # autocast dtype (usually bfloat16)
    model_type: str                                       # 'llava-ov' for now

    def build_inputs(self, image: Image.Image, user_text: str) -> Dict[str, torch.Tensor]:
        """Tokenize + image-encode one (image, question) pair into model inputs."""
        # HF chat template: one user turn with an image placeholder + the question
        conv = [{                                         # standard HF VLM conversation schema
            "role": "user",
            "content": [{"type": "image"}, {"type": "text", "text": user_text}],
        }]
        prompt = self.processor.apply_chat_template(      # turns the convo into a string
            conv, add_generation_prompt=True
        )
        # processor handles image preprocessing + tokenization in one call
        inputs = self.processor(images=image, text=prompt,
                                return_tensors="pt").to(self.device)
        # cast pixel_values to model dtype (vision tower expects matching dtype)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return inputs

    @torch.no_grad()
    def forward(self, **kwargs):
        """Single forward pass with use_cache=True so we can step-by-step decode."""
        return self.model(**kwargs, use_cache=True, output_attentions=False)


def load_backbone(vlm_path: str, model_type: str,
                  device: str = "cuda", dtype=torch.bfloat16) -> VLMBackbone:
    """Load the model + processor. Currently only llava-ov is plumbed end-to-end."""
    if model_type not in ("llava-ov", "llava_ov", "llava-1.5", "llava_15"):
        raise NotImplementedError(                        # explicit so we don't silently break
            f"model_type={model_type!r} not wired here. Implement build_inputs / "
            "weak-side construction for it (see SID's attention extraction)."
        )
    dev = torch.device(device)
    processor = AutoProcessor.from_pretrained(vlm_path)   # tokenizer + image processor
    model = AutoModelForImageTextToText.from_pretrained(  # generic HF VLM loader
        vlm_path, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(dev).eval()
    return VLMBackbone(model=model, processor=processor,
                       device=dev, dtype=dtype, model_type=model_type)


# =============================================================================
# Section 3 — Contrastive-decoding core (shared by all three methods)
# =============================================================================

@dataclass
class ContrastiveConfig:
    """Hyper-params common to VCD / ICD / SID (defaults from the papers)."""
    alpha: float = 1.0                                    # contrast strength
    beta: float = 0.1                                     # adaptive-plausibility cutoff
    max_new_tokens: int = 512                             # CHAIR captions go ~64-100; POPE ~5
    do_sample: bool = False                               # greedy by default (matches papers)


def _contrastive_logits(logits_s: torch.Tensor, logits_w: torch.Tensor,
                        alpha: float, beta: float) -> torch.Tensor:
    """Combine strong/weak logits with the VCD update + adaptive plausibility mask.

    Shapes: both inputs are [B, V]. Returns [B, V] with implausible tokens set
    to -inf so they can never be sampled. This is the *exact* operation used
    by all three baselines — that's why they're one function here.
    """
    probs_s = logits_s.softmax(dim=-1)                    # p_strong  ∈ [0,1]^V
    cutoff = beta * probs_s.max(dim=-1, keepdim=True).values  # adaptive threshold β·max(p)
    plausible = probs_s >= cutoff                         # boolean mask of "valid" tokens

    contrast = (1.0 + alpha) * logits_s - alpha * logits_w  # the VCD/ICD/SID update
    contrast = contrast.masked_fill(~plausible, float("-inf"))  # kill implausible tokens
    return contrast


class WeakBuilder:
    """Strategy interface: each baseline implements `prepare(strong_inputs)`."""

    name: str = "abstract"

    def prepare(self, backbone: VLMBackbone, image: Image.Image,
                user_text: str, strong_inputs: Dict[str, torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """Return the *weak* inputs dict, paired with `strong_inputs`."""
        raise NotImplementedError


# ─────────────────────────── VCD: noised image ──────────────────────────────
class VCDWeak(WeakBuilder):
    name = "vcd"

    def __init__(self, noise_step: int = 500, seed: int = 0):
        self.noise_step = noise_step                      # t in DDPM forward process
        self.seed = seed                                  # makes weak pass reproducible

    def prepare(self, backbone, image, user_text, strong_inputs):
        noised = diffusion_noise(image, t=self.noise_step, seed=self.seed)  # destroy visual info
        return backbone.build_inputs(noised, user_text)   # re-encode through vision tower


# ─────────────────────────── ICD: disturbed prompt ──────────────────────────
class ICDWeak(WeakBuilder):
    name = "icd"

    # Default disturbance from the ICD paper §3.2 ("role-disturbance prefix")
    DISTURB = ("You are a confused object detector. You are likely to "
               "describe objects that do not exist in the image. ")

    def prepare(self, backbone, image, user_text, strong_inputs):
        # Inject the disturbance *before* the actual user question; keep image
        return backbone.build_inputs(image, self.DISTURB + user_text)


# ─────────────────────────── SID: low-importance vision tokens ──────────────
class SIDWeak(WeakBuilder):
    name = "sid"

    def __init__(self, keep_ratio: float = 0.1, attn_layer: int = 2):
        self.keep_ratio = keep_ratio                      # fraction of vision tokens to RETAIN
        self.attn_layer = attn_layer                      # which layer's attention ranks tokens

    @torch.no_grad()
    def prepare(self, backbone, image, user_text, strong_inputs):
        """SID's twist: re-do the forward keeping only the *least-attended*
        vision tokens. We approximate this by *masking out* the highly-attended
        ones in the weak pass via attention_mask zeros at those positions.

        This requires us to know which input positions are vision tokens. We
        get that from the model's image-token id; everything between the first
        and last occurrence of that token in input_ids is the vision span.
        """
        ids = strong_inputs["input_ids"][0]               # [T] — single example
        img_tok_id = backbone.model.config.image_token_index  # HF-standard attribute
        vis_pos = (ids == img_tok_id).nonzero(as_tuple=True)[0]  # positions of vision tokens
        if vis_pos.numel() == 0:                          # safety: no vision span found
            return strong_inputs                          # fall back to identity (no contrast)

        # Run a cheap forward with output_attentions=True at `attn_layer` to
        # get the model's *own* importance ranking of those vision tokens.
        out = backbone.model(**strong_inputs, output_attentions=True, use_cache=False)
        attn = out.attentions[self.attn_layer]            # [B, H, T, T] attention weights
        # Importance = how much the last (prompt-end) token attends to each vision token,
        # averaged over heads. This is SID's "context-aware" criterion (§3.1).
        last_to_vis = attn[0, :, -1, :].mean(dim=0)[vis_pos]  # [N_vis]
        N_vis = vis_pos.numel()
        keep_n = max(1, int(self.keep_ratio * N_vis))     # # tokens we KEEP (the LOW-imp ones)
        # bottomk == lowest-importance == what SID feeds the weak path
        bottom = torch.topk(last_to_vis, keep_n, largest=False).indices
        keep_set = set(vis_pos[bottom].tolist())          # positions to keep visible

        # Build weak inputs by zeroing attention_mask at *all other* vision positions.
        weak = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in strong_inputs.items()}
        attn_mask = weak["attention_mask"].clone()        # [B, T] — 1=keep, 0=mask out
        for p in vis_pos.tolist():
            if p not in keep_set:
                attn_mask[0, p] = 0                       # hide this vision token from attention
        weak["attention_mask"] = attn_mask
        return weak


# =============================================================================
# Section 4 — The greedy contrastive-generation loop (shared by all methods)
# =============================================================================

@torch.no_grad()
def generate_contrastive(backbone: VLMBackbone, image: Image.Image,
                         user_text: str, weak: WeakBuilder,
                         cfg: ContrastiveConfig) -> str:
    """Token-by-token greedy decoding with two parallel forward streams.

    We have to do this manually (HF `.generate()` doesn't expose a hook for
    two-distribution contrastive decoding without subclassing GenerationMixin).
    Both streams maintain their own past_key_values cache; per step we compute
    contrastive logits, pick a token, and append it to BOTH streams' inputs.
    """
    # 1) Build strong + weak inputs (one full prompt-pass each)
    s_in = backbone.build_inputs(image, user_text)        # strong stream inputs
    w_in = weak.prepare(backbone, image, user_text, s_in) # weak stream inputs (method-specific)

    # 2) Prefill: one forward each to populate KV caches with the prompt
    s_out = backbone.forward(**s_in)                      # strong prefill
    w_out = backbone.forward(**w_in)                      # weak prefill
    s_pkv, w_pkv = s_out.past_key_values, w_out.past_key_values

    # Last-token logits from prefill — first decoding step uses these
    logits_s = s_out.logits[:, -1, :]                     # [B, V]
    logits_w = w_out.logits[:, -1, :]                     # [B, V]

    eos = backbone.processor.tokenizer.eos_token_id       # stop condition
    generated: List[int] = []                             # output token ids

    # 3) Generation loop
    for _ in range(cfg.max_new_tokens):
        contrast = _contrastive_logits(logits_s, logits_w, cfg.alpha, cfg.beta)  # fuse
        next_id = (contrast.argmax(dim=-1) if not cfg.do_sample                  # greedy
                   else torch.distributions.Categorical(logits=contrast).sample())
        tok_id = int(next_id.item())
        if tok_id == eos:                                 # natural stop
            break
        generated.append(tok_id)

        # 4) Step both streams forward by the SAME chosen token
        nxt = next_id.unsqueeze(0)                        # [1,1]
        s_step = backbone.model(input_ids=nxt, past_key_values=s_pkv, use_cache=True)
        w_step = backbone.model(input_ids=nxt, past_key_values=w_pkv, use_cache=True)
        s_pkv, w_pkv = s_step.past_key_values, w_step.past_key_values
        logits_s = s_step.logits[:, -1, :]
        logits_w = w_step.logits[:, -1, :]

    # 5) Decode tokens → text
    return backbone.processor.tokenizer.decode(generated, skip_special_tokens=True).strip()


# =============================================================================
# Section 5 — POPE evaluation (yes/no object-existence, 3 splits)
# =============================================================================

def _pope_metrics(records: List[Dict]) -> Dict[str, float]:
    """Compute POPE's standard metrics: acc / prec / rec / F1 / yes_rate."""
    tp = fp = tn = fn = 0
    yes_count = 0
    for r in records:
        # Normalize answer to {"yes","no"}; first lowercase word wins (POPE convention)
        pred = re.search(r"\b(yes|no)\b", r["answer"].lower())
        pred = pred.group(1) if pred else "no"            # default "no" on parse failure
        gold = r["label"].lower().strip()
        if pred == "yes":
            yes_count += 1
            tp += (gold == "yes"); fp += (gold == "no")
        else:
            tn += (gold == "no");  fn += (gold == "yes")
    n = max(1, len(records))                              # avoid div-by-zero
    acc = (tp + tn) / n
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "yes_rate": yes_count / n, "n": n}


def eval_pope(backbone: VLMBackbone, pope_path: str, img_dir: str,
              weak: WeakBuilder, cfg: ContrastiveConfig,
              limit: Optional[int] = None) -> Dict:
    """Run contrastive decoding over a POPE jsonl and return metrics."""
    questions = [json.loads(l) for l in open(pope_path) if l.strip()]
    if limit:
        questions = questions[:limit]                    # for test mode
    records = []
    for q in tqdm(questions, desc=f"POPE[{weak.name}]"):
        img = Image.open(os.path.join(img_dir, q["image"])).convert("RGB")
        # POPE's prompt template: "Is there a <obj> in the image?"
        prompt = q["text"] if "text" in q else q["question"]
        ans = generate_contrastive(backbone, img, prompt, weak,
                                   ContrastiveConfig(**{**cfg.__dict__,
                                                        "max_new_tokens": 8}))
        records.append({"question_id": q.get("question_id"),
                        "image": q["image"],
                        "answer": ans,
                        "label": q["label"]})
    return {"metrics": _pope_metrics(records), "records": records}


# =============================================================================
# Section 6 — CHAIR evaluation (caption hallucination on COCO)
# =============================================================================

def _load_coco_annotations(ann_path: str):
    """Load COCO instances → return (synonyms_for_each_image, all_object_classes).

    The classical CHAIR setup uses MSCOCO's 80-class taxonomy plus a fixed
    synonym list to match free-text caption nouns against ground-truth objects.
    For brevity we use the *category names themselves* (no synonyms). This is
    a small upward bias vs the original metric — for paper-faithful numbers,
    drop in the synonym list from the original CHAIR repo.
    """
    coco = json.load(open(ann_path))
    cat_id_to_name = {c["id"]: c["name"] for c in coco["categories"]}
    img_objs: Dict[int, set] = {}
    for ann in coco["annotations"]:
        img_objs.setdefault(ann["image_id"], set()).add(cat_id_to_name[ann["category_id"]])
    img_file = {im["id"]: im["file_name"] for im in coco["images"]}
    return img_objs, img_file, set(cat_id_to_name.values())


def _chair_scores(caption: str, gt_objs: set, all_objs: set) -> Tuple[int, int, int]:
    """Return (n_hallucinated, n_mentioned, n_gt_recovered) for one caption."""
    cap = " " + caption.lower() + " "                     # pad so we can word-boundary search
    mentioned = {o for o in all_objs if (" " + o + " ") in cap}  # objects the caption names
    hallucinated = mentioned - gt_objs                    # named but not in image GT
    recovered = mentioned & gt_objs
    return len(hallucinated), len(mentioned), len(recovered)


def eval_chair(backbone: VLMBackbone, ann_path: str, img_dir: str,
               weak: WeakBuilder, cfg: ContrastiveConfig,
               n_images: int = 500, seed: int = 0) -> Dict:
    """CHAIR_s (sentence-level) and CHAIR_i (instance-level)."""
    img_objs, img_file, all_objs = _load_coco_annotations(ann_path)
    rng = random.Random(seed)
    ids = sorted(img_objs.keys())                         # deterministic sample
    rng.shuffle(ids)
    ids = ids[:n_images]

    total_hall, total_ment, total_rec, total_gt = 0, 0, 0, 0
    sentence_hits = 0                                     # # captions with ≥1 hallucination
    records = []
    cap_cfg = ContrastiveConfig(**{**cfg.__dict__, "max_new_tokens": 96})  # longer caps for CHAIR

    for iid in tqdm(ids, desc=f"CHAIR[{weak.name}]"):
        img = Image.open(os.path.join(img_dir, img_file[iid])).convert("RGB")
        cap = generate_contrastive(backbone, img,
                                   "Please describe this image in detail.",
                                   weak, cap_cfg)
        h, m, r = _chair_scores(cap, img_objs[iid], all_objs)
        total_hall += h; total_ment += m; total_rec += r
        total_gt += len(img_objs[iid])
        sentence_hits += int(h > 0)
        records.append({"image_id": iid, "caption": cap,
                        "hallucinated": h, "mentioned": m, "gt": len(img_objs[iid])})

    n = max(1, len(ids))
    metrics = {
        "CHAIR_s": sentence_hits / n,                     # fraction of caps with a halluc.
        "CHAIR_i": total_hall / max(1, total_ment),       # per-mention halluc rate
        "recall":  total_rec / max(1, total_gt),          # coverage of GT objects
        "avg_mentioned": total_ment / n,
        "n_images": n,
    }
    return {"metrics": metrics, "records": records}


# =============================================================================
# Section 7 — CLI
# =============================================================================

def _build_weak(method: str) -> WeakBuilder:
    """Map --method string → a WeakBuilder instance with paper-default params."""
    method = method.lower()
    if method == "vcd": return VCDWeak(noise_step=500)
    if method == "icd": return ICDWeak()
    if method == "sid": return SIDWeak(keep_ratio=0.1, attn_layer=2)
    raise ValueError(f"Unknown method {method!r}; choose vcd|icd|sid")


def _dry_run(backbone: VLMBackbone, weak: WeakBuilder, cfg: ContrastiveConfig,
             args) -> None:
    """Run one end-to-end example with verbose diagnostics, then return.

    Exercises every gotcha from the README: image_token_index attribute,
    pixel_values dtype, POPE schema fields, COCO annotations schema, and the
    full two-stream contrastive loop. Cheap (1 example, max_new_tokens=8) so
    it finishes in <30s — surface errors before launching the real run.
    """
    print("\n" + "=" * 60)
    print(f"  DRY RUN — method={args.method}  model={args.model_type}")
    print("=" * 60)

    # ── 1) Verify model exposes image_token_index (SID needs this) ──
    cfg_obj = backbone.model.config                               # HF config object
    img_tok_id = getattr(cfg_obj, "image_token_index",
                         getattr(cfg_obj, "image_token_id", None))  # try both spellings
    print(f"[1/5] image_token_index = {img_tok_id}")
    if img_tok_id is None:
        raise AttributeError(                                     # fail loud, not at SID time
            "Model config has no image_token_index/image_token_id. "
            "SID will not work — inspect model.config and patch SIDWeak.prepare."
        )

    # ── 2) Source one image + prompt from POPE > CHAIR > synthetic, in that order ──
    if args.pope_path:                                            # POPE branch
        q = json.loads(next(l for l in open(args.pope_path) if l.strip()))
        # Schema sanity: papers/repos disagree on field names — surface it now
        prompt = q.get("text") or q.get("question")
        if prompt is None:
            raise KeyError(f"POPE record has neither 'text' nor 'question': {list(q.keys())}")
        img_path = os.path.join(args.pope_img_dir, q["image"])
        print(f"[2/5] POPE sample: image={q['image']}  prompt={prompt!r}  label={q.get('label')}")
        image = Image.open(img_path).convert("RGB")
    elif args.coco_ann_path:                                      # CHAIR branch
        img_objs, img_file, all_objs = _load_coco_annotations(args.coco_ann_path)
        iid = sorted(img_objs.keys())[0]                          # first image
        prompt = "Please describe this image in detail."
        img_path = os.path.join(args.coco_img_dir, img_file[iid])
        print(f"[2/5] CHAIR sample: image_id={iid}  file={img_file[iid]}  gt_objs={img_objs[iid]}")
        image = Image.open(img_path).convert("RGB")
    else:                                                         # synthetic fallback (no data needed)
        prompt = "What is in this image?"
        image = Image.new("RGB", (336, 336), color=(127, 127, 127))  # gray square
        print(f"[2/5] No data paths given — using synthetic 336×336 gray image")

    # ── 3) Build inputs + locate vision span (catches dtype/template issues) ──
    s_in = backbone.build_inputs(image, prompt)
    seq_len = s_in["input_ids"].shape[1]
    vis_pos = (s_in["input_ids"][0] == img_tok_id).nonzero(as_tuple=True)[0]
    print(f"[3/5] strong inputs: seq_len={seq_len}  pixel_dtype={s_in['pixel_values'].dtype}  "
          f"vision_tokens={vis_pos.numel()}  span=[{vis_pos[0].item() if vis_pos.numel() else '∅'}"
          f"..{vis_pos[-1].item() if vis_pos.numel() else '∅'}]")
    if vis_pos.numel() == 0 and args.method == "sid":
        print("      WARNING: no vision tokens found in input — SID will fall back to no-op")

    # ── 4) Build the weak side (method-specific — catches each baseline's failure path) ──
    w_in = weak.prepare(backbone, image, prompt, s_in)
    print(f"[4/5] weak inputs ({weak.name}): seq_len={w_in['input_ids'].shape[1]}  "
          f"attn_mask_kept={int(w_in['attention_mask'].sum().item())}/{w_in['attention_mask'].numel()}")

    # ── 5) Run one full contrastive generation (8 tokens max — fast) ──
    short_cfg = ContrastiveConfig(**{**cfg.__dict__, "max_new_tokens": 8})
    out_text = generate_contrastive(backbone, image, prompt, weak, short_cfg)
    print(f"[5/5] generated (max_new_tokens=8): {out_text!r}")

    print("\n" + "=" * 60)
    print("  DRY RUN OK — safe to launch full eval (drop --dry-run)")
    print("=" * 60 + "\n")


def main():
    p = argparse.ArgumentParser()
    # — args mirror code/eval_chair.py so the pipeline launcher needs no special-casing —
    p.add_argument("--vlm_path", required=True, help="HF hub id or local path")
    p.add_argument("--model_type", default="llava-ov")
    p.add_argument("--method", required=True, choices=["vcd", "icd", "sid"])
    p.add_argument("--output_dir", required=True)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=0.1)
    # POPE
    p.add_argument("--pope_path", default=None, help="POPE jsonl file")
    p.add_argument("--pope_img_dir", default=None)
    p.add_argument("--pope_limit", type=int, default=None)
    # CHAIR
    p.add_argument("--coco_ann_path", default=None, help="instances_val2014.json")
    p.add_argument("--coco_img_dir", default=None)
    p.add_argument("--n_images", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="Load model, run one example end-to-end with diagnostics, exit. "
                        "Uses POPE>CHAIR>synthetic image, whichever is available.")
    args = p.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    weak = _build_weak(args.method)                       # pick the baseline
    cfg = ContrastiveConfig(alpha=args.alpha, beta=args.beta)

    print(f"[{args.method}] loading {args.vlm_path} ({args.model_type})")
    backbone = load_backbone(args.vlm_path, args.model_type)

    if args.dry_run:                                      # smoke-test path — exits before eval loops
        _dry_run(backbone, weak, cfg, args)
        return

    summary = {"method": args.method, "alpha": args.alpha, "beta": args.beta,
               "model": args.vlm_path}

    if args.pope_path:                                    # POPE branch
        print(f"[{args.method}] POPE: {args.pope_path}")
        res = eval_pope(backbone, args.pope_path, args.pope_img_dir,
                        weak, cfg, limit=args.pope_limit)
        (out / "pope_summary.json").write_text(json.dumps(res, indent=2))
        summary["pope"] = res["metrics"]

    if args.coco_ann_path:                                # CHAIR branch
        print(f"[{args.method}] CHAIR: {args.coco_ann_path}")
        res = eval_chair(backbone, args.coco_ann_path, args.coco_img_dir,
                         weak, cfg, n_images=args.n_images, seed=args.seed)
        (out / "chair_summary.json").write_text(json.dumps(res, indent=2))
        summary["chair"] = res["metrics"]

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
