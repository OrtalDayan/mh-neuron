"""check_shapes.py — pre-flight compatibility check for merge_pmbt.py.

For each (VLM, reasoning-LLM) pair you intend to merge, this loads both
config.json files and asserts that the four shape-critical fields agree:

    hidden_size          (per-token embedding width)
    num_hidden_layers    (transformer depth)
    intermediate_size    (FFN width — the "neurons" PMBT classifies)
    vocab_size           (embedding / lm_head rows)

merge_pmbt.py does element-wise weight interpolation on the LLM tower, so any
mismatch in hidden_size / num_hidden_layers / intermediate_size makes the merge
impossible (tensor shapes won't broadcast). A vocab_size mismatch is subtler:
the script *excludes* embed_tokens/lm_head from the merge for some backbones,
so a vocab difference may be survivable — we flag it as a WARNING, not a hard
FAIL, and say so explicitly.

The VLM's LLM tower lives under a nested key in its config. Qwen2.5-VL and
LLaVA store the language-model config under "text_config" (or "llm_config" /
"language_config" depending on the release); a plain LLM stores the fields at
top level. We search both.

Usage
-----
    # Edit the PAIRS list below, or pass pairs on the command line:
    python check_shapes.py
    python check_shapes.py --pair Qwen/Qwen2.5-VL-3B-Instruct Qwen/Qwen2.5-3B-Instruct
    python check_shapes.py --pair <vlm> <llm> --pair <vlm2> <llm2>

Reads only config.json (a few KB each), never downloads weights. Works on a
local path or a HF repo id (the latter needs network + `pip install huggingface_hub`).
"""

import argparse
import json
import os
import sys

# Fields that MUST match for an element-wise tower merge to be valid.
HARD_FIELDS = ["hidden_size", "num_hidden_layers", "intermediate_size"]
# Field that SHOULD match but may be excluded from the merge → warning only.
SOFT_FIELDS = ["vocab_size"]

# Keys under which a VLM may nest its language-model config.
LLM_SUBCONFIG_KEYS = ["text_config", "llm_config", "language_config", "llm"]


# ---------------------------------------------------------------------------
# Default pairs to check. Edit these to match what you'll actually merge.
# Each entry: (label, vlm_path_or_repo, llm_path_or_repo)
# ---------------------------------------------------------------------------
DEFAULT_PAIRS = [
    ("LLaVA-1.5-7b  + MetaMath-7B",
     "llava-hf/llava-1.5-7b-hf", "meta-math/MetaMath-7B-V1.0"),
    ("LLaVA-OV-7B   + Qwen2-Math-7B",
     "llava-hf/llava-onevision-qwen2-7b-ov-hf", "Qwen/Qwen2-Math-7B-Instruct"),
    ("InternVL2.5-8B + InternLM2.5-7B-Chat",
     "OpenGVLab/InternVL2_5-8B", "internlm/internlm2_5-7b-chat"),
    ("Qwen2.5-VL-3B + Qwen2.5-3B-Instruct",
     "Qwen/Qwen2.5-VL-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct"),
]


def load_config(path_or_repo):
    """Return the parsed config.json dict for a local dir or HF repo id."""
    # Local directory containing config.json
    local = os.path.join(path_or_repo, "config.json")
    if os.path.isfile(local):
        with open(local) as f:
            return json.load(f)
    # Local file passed directly
    if os.path.isfile(path_or_repo) and path_or_repo.endswith(".json"):
        with open(path_or_repo) as f:
            return json.load(f)
    # Otherwise treat as a HF repo id and fetch just the config
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise FileNotFoundError(
            f"No local config.json at '{path_or_repo}', and huggingface_hub "
            f"is not installed to fetch it remotely. Either point at a local "
            f"model directory or run: pip install huggingface_hub"
        )
    cfg_path = hf_hub_download(repo_id=path_or_repo, filename="config.json")
    with open(cfg_path) as f:
        return json.load(f)


def extract_llm_fields(cfg):
    """Pull the four shape fields from a config, descending into a nested
    text/llm sub-config if the top level doesn't carry them."""
    def has_all(d):
        # Require ALL hard fields, not just any — a top-level config may carry
        # vocab_size while the real tower dims live nested under text_config.
        return isinstance(d, dict) and all(k in d for k in HARD_FIELDS)

    source = cfg
    used_key = "(top-level)"
    if not has_all(cfg):
        for k in LLM_SUBCONFIG_KEYS:
            if has_all(cfg.get(k, {})):
                source = cfg[k]
                used_key = k
                break

    fields = {}
    for f in HARD_FIELDS:
        fields[f] = source.get(f, None)
    # vocab_size may live only at the top level even when tower dims are nested.
    for f in SOFT_FIELDS:
        fields[f] = source.get(f, cfg.get(f, None))
    return fields, used_key


def check_pair(label, vlm_path, llm_path):
    """Return (ok: bool, lines: list[str]) for one VLM/LLM pair."""
    lines = [f"\n=== {label} ===",
             f"  VLM: {vlm_path}",
             f"  LLM: {llm_path}"]
    try:
        vlm_cfg = load_config(vlm_path)
        llm_cfg = load_config(llm_path)
    except Exception as e:
        lines.append(f"  ERROR loading config: {e}")
        return False, lines

    vlm_fields, vlm_src = extract_llm_fields(vlm_cfg)
    llm_fields, llm_src = extract_llm_fields(llm_cfg)
    lines.append(f"  VLM tower fields read from: {vlm_src}")
    lines.append(f"  LLM fields read from:       {llm_src}")

    hard_ok = True
    soft_ok = True
    header = f"  {'field':<20}{'VLM':>14}{'LLM':>14}   verdict"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for f in HARD_FIELDS + SOFT_FIELDS:
        v, l = vlm_fields[f], llm_fields[f]
        if v is None or l is None:
            verdict = "?? missing"
            if f in HARD_FIELDS:
                hard_ok = False
            else:
                soft_ok = False
        elif v == l:
            verdict = "OK"
        else:
            if f in HARD_FIELDS:
                verdict = "FAIL (must match)"
                hard_ok = False
            else:
                verdict = "WARN (vocab differs)"
                soft_ok = False
        lines.append(f"  {f:<20}{str(v):>14}{str(l):>14}   {verdict}")

    # Derived neuron count, for sanity against the paper's table.
    if vlm_fields["intermediate_size"] and vlm_fields["num_hidden_layers"]:
        n = vlm_fields["intermediate_size"] * vlm_fields["num_hidden_layers"]
        lines.append(f"  → VLM tower neuron count: "
                     f"{vlm_fields['intermediate_size']} x "
                     f"{vlm_fields['num_hidden_layers']} = {n:,}")

    if hard_ok and soft_ok:
        lines.append("  RESULT: MERGE-SAFE (all fields agree)")
    elif hard_ok and not soft_ok:
        lines.append("  RESULT: LIKELY OK — hard fields agree; vocab differs. "
                     "Safe only if embed_tokens/lm_head are excluded from the "
                     "merge (they are for some backbones). Verify before trusting.")
    else:
        lines.append("  RESULT: INCOMPATIBLE — element-wise tower merge will fail.")
    return hard_ok, lines


def main():
    ap = argparse.ArgumentParser(description="Pre-flight shape check for merge_pmbt.py")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("VLM", "LLM"),
                    help="A VLM path/repo and its LLM merge partner. "
                         "Repeatable. If omitted, uses the built-in DEFAULT_PAIRS.")
    args = ap.parse_args()

    if args.pair:
        pairs = [(f"{v}  +  {l}", v, l) for v, l in args.pair]
    else:
        pairs = DEFAULT_PAIRS

    all_hard_ok = True
    for label, vlm, llm in pairs:
        ok, lines = check_pair(label, vlm, llm)
        all_hard_ok = all_hard_ok and ok
        print("\n".join(lines))

    print("\n" + "=" * 60)
    if all_hard_ok:
        print("ALL PAIRS PASS the hard shape check (see per-pair vocab warnings).")
        sys.exit(0)
    else:
        print("AT LEAST ONE PAIR IS INCOMPATIBLE — fix before merging.")
        sys.exit(1)


if __name__ == "__main__":
    main()
