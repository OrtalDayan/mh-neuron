"""
GPU version: percentile comparison of Xu et al.'s T_v=2 threshold across models.

Uses PyTorch on GPU. ~50-100x faster than numpy CPU. Expected runtime: 1-3 min
for 4 models × 6 layers.

Question (same as CPU version): given Xu's per-neuron 0-10 normalization, does
T_v=2 nonetheless correspond to qualitatively different fractions of activated
visual tokens across LLaVA-1.5, LLaVA-OV, InternVL, Qwen?

Memory profile per layer (peak):
  vis_acts on GPU: ~5 GB float16 (LLaVA-OV, the largest)
  validity mask:   ~1 GB bool
  intermediate boolean comparisons: ~1 GB each, freed eagerly
  Total peak:      ~8 GB. Fits any modern GPU.
"""

import numpy as np
import torch
import os, time, sys

# Anchor to project root regardless of cwd.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR) if os.path.basename(SCRIPT_DIR) == "code" else SCRIPT_DIR

MODELS = [
    "llava-1.5-7b",
    "llava-onevision-7b",
    "internvl2.5-8b",
    "qwen2.5-vl-7b",
]

LAYERS_TO_SAMPLE = [0, 5, 10, 15, 20, 25]
THRESHOLD = 2.0
N_T = 4

# Pick GPU. Prefer one explicitly visible via CUDA_VISIBLE_DEVICES; otherwise GPU 0.
if not torch.cuda.is_available():
    print("ERROR: CUDA not available. Use CPU script (percentile_check.py) instead.")
    sys.exit(1)

DEVICE = torch.device("cuda:0")
print(f"Using device: {DEVICE} ({torch.cuda.get_device_name(0)})")
print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")


def load_visual_activations_gpu(model_dir, layer):
    """Load one layer's npz and compute summary statistics entirely on GPU.

    Returns
    -------
    summary : dict of int counts (already moved to CPU as Python ints).
    """
    path = os.path.join(model_dir, f"raw_acts_layer{layer}.npz")
    if not os.path.exists(path):
        return None

    # Load from disk to CPU memory first (numpy), then move to GPU.
    f = np.load(path, allow_pickle=True)
    vis_acts_np = f["vis_acts"]              # (n_neurons, 50, max_tokens) float16
    vis_lengths_np = f["vis_lengths"]        # (n_neurons, 50) int16

    # Move to GPU. Keep float16 to save memory; comparisons work fine in fp16.
    vis_acts = torch.from_numpy(vis_acts_np).to(DEVICE, non_blocking=True)
    vis_lengths = torch.from_numpy(vis_lengths_np.astype(np.int32)).to(DEVICE)

    n_neurons, n_samples, max_tokens = vis_acts.shape

    # Build validity mask on GPU. arange creates the position index;
    # broadcasting against vis_lengths gives the per-(neuron, sample) cutoff.
    pos = torch.arange(max_tokens, device=DEVICE, dtype=torch.int32)
    valid = pos[None, None, :] < vis_lengths[:, :, None]   # bool, same shape as vis_acts

    # Total valid activations
    n_acts = int(valid.sum().item())

    # Threshold counts. Each is one fused GPU kernel; microseconds each.
    # `(vis_acts < x) & valid` produces a bool tensor on GPU; .sum().item()
    # reduces and pulls the scalar back to CPU.
    def count_lt(thresh):
        return int(((vis_acts < thresh) & valid).sum().item())
    def count_gt(thresh):
        return int(((vis_acts > thresh) & valid).sum().item())

    sum_below_T   = count_lt(THRESHOLD)
    sum_below_05  = count_lt(0.5)
    sum_below_1   = count_lt(1.0)
    sum_below_3   = count_lt(3.0)
    sum_below_5   = count_lt(5.0)
    sum_above_9   = count_gt(9.0)
    sum_ge_T      = n_acts - sum_below_T

    # Xu's n_v rule: per-(neuron, sample) count of high-activation tokens
    high_per_sample = ((vis_acts > THRESHOLD) & valid).sum(dim=2)   # (n_neurons, 50) int
    n_pairs = int(high_per_sample.numel())
    sum_pair_ge_n = int((high_per_sample >= N_T).sum().item())

    # Free GPU memory before next layer loads
    del vis_acts, vis_lengths, valid, high_per_sample
    torch.cuda.empty_cache()

    return {
        "n_acts":       n_acts,
        "sum_below_T":  sum_below_T,
        "sum_below_05": sum_below_05,
        "sum_below_1":  sum_below_1,
        "sum_below_3":  sum_below_3,
        "sum_below_5":  sum_below_5,
        "sum_above_9":  sum_above_9,
        "sum_ge_T":     sum_ge_T,
        "n_pairs":      n_pairs,
        "sum_pair_ge_n": sum_pair_ge_n,
    }


def analyse_model(model_name):
    print(f"\n{'='*70}")
    print(f"  {model_name}")
    print('='*70)

    model_dir = os.path.join(PROJECT_ROOT, "results", "3-classify", "full", model_name,
                             "llm_fixed_threshold", "act_pattern_raw")
    if not os.path.isdir(model_dir):
        print(f"  SKIP: {model_dir} not found")
        return None

    totals = {k: 0 for k in [
        "n_acts", "sum_below_T", "sum_below_05", "sum_below_1",
        "sum_below_3", "sum_below_5", "sum_above_9", "sum_ge_T",
        "n_pairs", "sum_pair_ge_n",
    ]}
    layers_processed = []

    for L in LAYERS_TO_SAMPLE:
        t0 = time.time()
        summary = load_visual_activations_gpu(model_dir, L)
        if summary is None:
            print(f"  layer {L}: file missing, skipping")
            continue
        layers_processed.append(L)
        for k in totals:
            totals[k] += summary[k]
        print(f"  layer {L}: {summary['n_acts']:,} valid activations "
              f"(processed in {time.time()-t0:.1f}s)")

    if not layers_processed:
        return None

    n = totals["n_acts"]
    n_pairs = totals["n_pairs"]
    print(f"\n  Aggregated across {len(layers_processed)} layers: "
          f"{n:,} activation values, {n_pairs:,} (neuron, sample) pairs")

    pct_at_T    = totals["sum_below_T"]  / n * 100
    pct_below_1 = totals["sum_below_1"]  / n * 100
    pct_below_3 = totals["sum_below_3"]  / n * 100
    pct_below_5 = totals["sum_below_5"]  / n * 100
    frac_zero   = totals["sum_below_05"] / n * 100
    frac_max    = totals["sum_above_9"]  / n * 100
    frac_ge_T   = totals["sum_ge_T"]     / n * 100
    frac_samples_visactive = totals["sum_pair_ge_n"] / n_pairs * 100

    print(f"\n  --- Threshold T_v=2 analysis ---")
    print(f"  Percentile of T=2 in visual-token activation distribution: {pct_at_T:.2f}%")
    print(f"      (i.e., {pct_at_T:.1f}% of visual tokens have activation < 2)")
    print(f"  Fraction of visual tokens with activation >= 2: {frac_ge_T:.2f}%")
    print(f"\n  --- Sample-level (Xu's n_v=4 rule) ---")
    print(f"  Fraction of (neuron, sample) pairs where >=4 visual tokens activate above 2:")
    print(f"      {frac_samples_visactive:.3f}%")
    print(f"\n  --- Distribution shape ---")
    print(f"  Frac < 0.5 (near-zero):  {frac_zero:.2f}%")
    print(f"  Frac < 1:                {pct_below_1:.2f}%")
    print(f"  Frac < 2 (T_v):          {pct_at_T:.2f}%")
    print(f"  Frac < 3:                {pct_below_3:.2f}%")
    print(f"  Frac < 5:                {pct_below_5:.2f}%")
    print(f"  Frac > 9 (near-max):     {frac_max:.2f}%")

    return {
        "model": model_name,
        "n_acts": n,
        "pct_at_T": pct_at_T,
        "frac_ge_T": frac_ge_T,
        "frac_samples_visactive": frac_samples_visactive,
        "frac_zero": frac_zero,
        "frac_max": frac_max,
        "layers": layers_processed,
    }


def main():
    print("="*70)
    print("Xu et al. T_v=2 threshold — cross-architecture percentile comparison (GPU)")
    print("="*70)
    print(f"Threshold: T_v={THRESHOLD}, n_v={N_T}")
    print(f"Layers sampled per model: {LAYERS_TO_SAMPLE}")

    results = []
    for m in MODELS:
        r = analyse_model(m)
        if r is not None:
            results.append(r)

    # Summary table
    print("\n\n" + "="*70)
    print("SUMMARY TABLE")
    print("="*70)
    print(f"{'Model':<22} {'%<T_v=2':>10} {'%>=T_v=2':>11} {'%(n_v=4 rule)':>14}")
    print("-"*70)
    for r in results:
        print(f"{r['model']:<22} {r['pct_at_T']:>9.2f}% {r['frac_ge_T']:>10.2f}% {r['frac_samples_visactive']:>13.3f}%")

    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    if not results:
        print("No models loaded. Check that PROJECT_ROOT is correct:")
        print(f"  PROJECT_ROOT = {PROJECT_ROOT}")
        print("  Expected to find: results/3-classify/full/<model>/llm_fixed_threshold/act_pattern_raw/")
        return
    pcts_at_T = [r["pct_at_T"] for r in results]
    spread_pct = max(pcts_at_T) - min(pcts_at_T)

    # Also compute the spread on the SAMPLE-LEVEL Xu rule, which is the
    # decision-relevant statistic.
    sample_rule_pcts = [r["frac_samples_visactive"] for r in results]
    spread_sample_rule = max(sample_rule_pcts) - min(sample_rule_pcts)

    print(f"Range of T_v=2 percentile across models: {min(pcts_at_T):.2f}% to {max(pcts_at_T):.2f}%")
    print(f"Spread (max - min): {spread_pct:.2f} percentage points")
    print()
    print(f"Range of Xu n_v=4 sample-rule fraction: {min(sample_rule_pcts):.3f}% to {max(sample_rule_pcts):.3f}%")
    print(f"Spread (max - min): {spread_sample_rule:.3f} percentage points")
    print()
    if spread_pct >= 5 or spread_sample_rule >= 10:
        print(">> SUPPORTS the rebuttal claim that the same threshold maps to")
        print("   architecturally different fractions of activated tokens.")
    elif spread_pct >= 2 or spread_sample_rule >= 5:
        print(">> WEAK support. Effect is real but modest.")
        print("   Could be cited but with measured language.")
    else:
        print(">> DOES NOT support the distribution-shape claim.")
        print("   Recommend keeping the dilemma-only framing in the rebuttal.")


if __name__ == "__main__":
    main()