# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

VLM (vision-language model) interpretability research. The pipeline classifies LLM-FFN neurons inside several VLM backbones into modality categories — **visual / text / multimodal / unknown** — extending Xu et al. (MM 2025) Phases 1–2 with a permutation-based statistical test (PMBT). Classification labels then drive downstream studies: targeted ablation, hallucination scoring, activation steering, and PMBT-guided weight merging.

Two label flavors are produced for every model:
- **FT** — Xu's fixed-threshold (`llm_fixed_threshold/` dirs).
- **PMBT** — permutation test, Otsu threshold, α=0.05 (`llm_permutation/` dirs).

Three hook points are supported (each writes to its own `_gate_up` / `_attn` suffix):
- `gate` (default) — SiLU output of MLP gate_proj.
- `gate_up` — SwiGLU intermediate (`SiLU(gate)*up_proj`), Geva's conceptual equivalent.
- `attn` — attention `o_proj` input; classifies attention heads, not FFN neurons.

## Orchestrator

**Everything goes through `code/run_pipeline.sh`** (~7.7K LOC). The Python scripts are dispatchable units; the shell script handles model resolution, sharding, LSF submission, idempotence, and output paths. Don't invent invocation conventions — re-use the same flags and output directory shape as `run_pipeline.sh`.

Common usage:
```bash
bash code/run_pipeline.sh --step 1 --model-type llava-ov          # one step
bash code/run_pipeline.sh --step all                              # standard chain (steps 1–4, 8)
bash code/run_pipeline.sh --step all_att                          # adds attention (step 9)
bash code/run_pipeline.sh --mode test                             # 6 fig3 images, 2 layers, 1 GPU
bash code/run_pipeline.sh --mode test --local                     # no bsub, runs on this node
bash code/run_pipeline.sh --step 3 --shards 32                    # 32-way sharding for classify
bash code/run_pipeline.sh --step 3 --hook-point gate_up           # alternative hook
bash code/run_pipeline.sh --suffix _v2                             # output to results/.../classification_v2
bash code/run_pipeline.sh --mode test --clean 3 --wipe            # delete results + logs from step 3 on
```

Step numbers ↔ names (see lines 792–826 of `run_pipeline.sh` for the full map):

| # | Name | Script |
|---|------|--------|
| 1 | describe (`gd`) | `generate_descriptions.py` |
| 1c | clean_desc | `clean_descriptions.py` |
| 2 | merge_desc (`merge_gd`) | (in-script merge) |
| 3 | classify (`cn`) | `neuron_modality_statistical.py` |
| 4 | merge_class (`merge_nc`) | `merge_classification.py` |
| 5 | check_collisions | `check_token_collisions.py` |
| 6 | find_fig3 | `find_fig3_neurons.py` |
| 7 | equal_fraction_ablation | `equal_fraction_ablation.py` |
| 8 | visualize | `visualize_neuron_activations.py` |
| 9 | attention (`attn`) | `attention_analysis.py` |
| 10 | statistics (`plot`) | `plot_neuron_statistics.py` + `halluc_score_neurons.py` |
| 11 | vit_analysis | `vit_weight_analysis.py` |
| 12 | layer_plots | `generate_layer_tables_plots.py` |
| 13 | text_inject | `neuron_weight_merge.py` |
| 14 | snrf | `neuron_snrf_merge.py` |
| 15 | srf | `neuron_srf_merge.py` |
| 16 | tune_lambda / 17 select_lambda / 18 compose_layer1 | (in-script) |
| 19 | evaluate | VLMEvalKit (via `modern_vlms/VLMEvalKit_brv/`) |
| 20 | summarize | `summarize_eval_results.py` |
| 21 | weight_diff_rank | `weight_diff_rank.py` |
| 22 | mathverse_ablation | `prepare_mathverse_td.py` + `score_mathverse_local.py` / `gpt_score_mathverse.py` |
| 23 | compare_hooks | `compare_hook_points.py` |
| 24 | ranked_ablation (P4) | `run1_ablation.py` |
| 25 | weight_merge (P5) | `merge_pmbt.py` |
| — | steering / merge_steering / plot_steering | (in-script) |

Cluster execution: LSF (`bsub`) with default queue `waic-risk`. Each GPU job starts at `gmem=80G` and is auto-resubmitted at smaller tiers (`40G`, `10G`) if it stays PEND for `GMEM_WAIT` seconds. Helpers `bsub_tiered`, `is_job_active`, `is_job_pending` enforce idempotence — re-running skips jobs whose output exists or that are currently PEND/RUN.

Baselines (VCD / ICD / SID, for hallucination eval): `bash run_baselines.sh` — uses the same pipeline conventions (env-overridable defaults at the top of the script).

## Python environments

There are **three** uv-managed venvs and the pipeline picks one based on `--model-type`:

| Env | When used | Notes |
|---|---|---|
| `.venv/` (root) | Only `llava-liuhaotian` was originally supported here, **but** `run_pipeline.sh` actually routes that model to `modern_vlms/.venv` too. Effectively a legacy env; pinned to `transformers==4.37.2`, `torch==2.1.2`. The known broken import is `from torch._C import _get_cpp_backtrace` — pipeline checks for this and aborts. |
| `modern_vlms/.venv/` | llava-hf, llava-mistral, llava-llama3, llava-ov, qwen2vl, qwen25vl-7b, qwen25vl-3b, idefics2, llava-liuhaotian | Modern transformers + torch. VLMEvalKit also installed here. |
| `modern_vlms/intervl_env/.venv_internvl/` | `internvl` only | Separate because InternVL pins different versions. |

Critical: when invoking the modern interpreter, `run_pipeline.sh` unsets `VIRTUAL_ENV` and strips `.venv` entries from `PYTHONPATH` so the legacy env's site-packages can't shadow it. Replicate this if you launch Python directly outside the pipeline.

The root `pyproject.toml` declares `llava = { path = "../LLaVA", editable = true }` and `baukit` from GitHub. `../LLaVA` (sibling to this repo) is the cloned LLaVA source.

HF cache: `HF_HOME` defaults to `$WORK_DIR/.cache/huggingface` so all cluster nodes share downloads.

## Repository layout

The repository root is `mh-neuron/`. Most of the pipeline lives inside `modality_taxonomy/`, so pipeline commands shown elsewhere in this file (e.g. `bash code/run_pipeline.sh`) are written **relative to `modality_taxonomy/`** — from the project root, prefix them with `modality_taxonomy/` or `cd` into that subfolder first.

```
mh-neuron/                         # project root
├── LLaVA/                         # Vendored LLaVA source (editable install via root pyproject)
├── modality_taxonomy/             # Main research code (most work happens here)
│   ├── code/                      # Active scripts + orchestrator (also: many *_old*.py kept for reference)
│   ├── data/                      # POPE, CHAIR (val2014 + annotations), MathVerse, TriviaQA, MMLU, VSR, ScienceQA, VQAv2, detail_23k.json
│   ├── results/                   # Per-step outputs, named `{step_num}-{name}/{mode}/{model_name}/...`
│   ├── logs/                      # LSF logs, named `{mode}/{model_name}/{step}_{model_type}{hook}/...`
│   ├── modern_vlms/               # Sibling project tree with its own venvs + VLMEvalKit forks
│   ├── backups/                   # Old artifacts (don't edit)
│   ├── pyproject.toml             # Root deps (transformers==4.37.2 era)
│   └── run_baselines.sh           # VCD/ICD/SID baseline launcher
├── README.md
└── CLAUDE.md                      # This file
```

Result directories follow the step-numbered pattern (e.g. `modality_taxonomy/results/3-classify/full/llava-onevision-7b/llm_permutation/`). The `mode` segment is `test` or `full`. The `model_name` is derived in `run_pipeline.sh` (e.g. `llava-1.5-7b`, `llava-onevision-7b`, `qwen2.5-vl-7b`, `internvl2.5-8b`, `idefics2-8b`).

### Worktrees

This repo uses **git worktrees** for parallel work. There are typically two worktrees on disk:
- `/home/projects/bagon/ortalda/mh-neuron/` — main worktree, often on `pmbt-alpha-0001`
- `/home/projects/bagon/ortalda/mh-neuron-window-2-work/` — sibling worktree on `feature/window-2-work`

Both share the same `.git` and `.gitignore`. Each has its own `.claude/settings.local.json` (gitignored). Edits in one worktree do NOT affect the other on disk — they only meet via git operations (merge, cherry-pick, rebase). When working in one worktree, do not assume the other's uncommitted state matches.

## Key data flow

1. **Descriptions** (step 1) — VLM is asked "Could you describe the image?" on the 23K-image `detail_23k` subset of COCO train2017. Output JSON drives all downstream text-token labelling.
2. **Classification** (step 3) — for each layer, build a Top-N heap of most-activating images per neuron, then teacher-force the model's own descriptions to record per-token activations on those images. Compute both FT proportions `(pv, pt, pm, pu)` and the PMBT permutation null (1000 shuffles, Otsu threshold, α=0.05). Per-shard outputs are merged in step 4.
3. **Ablation / hallucination / steering** (steps 7, 10, 11, 24) — use the merged labels to either ablate or steer neurons of a chosen category, evaluating against POPE / CHAIR / MathVerse / TriviaQA, and report `ΔHallucination`, Fisher enrichment, CETT-diff, etc.
4. **Weight merging** (steps 13–25) — apply PMBT-masked task arithmetic: e.g., inject `(W_math − W_base)` only into text-classified neurons of a VLM (Method 1, "text_inject"), or transplant a stronger VLM's visual neurons (Method 2, "visual_transplant"). Several BRV-style baselines are evaluated in parallel.

Per-model architecture constants (used in `merge_classification.py` MODEL_SPECS):

| model_name | n_layers | neurons/layer |
|---|---|---|
| llava-1.5-7b | 32 | 11008 |
| internvl2.5-8b | 32 | 14336 |
| qwen2.5-vl-7b | 28 | 18944 |
| llava-onevision-7b | 28 | 3584 |
| llava-next-llama3-8b | 32 | 14336 |

## Conventions when modifying code

- **Old files**: many `*_old*.py`, `*_old*.sh`, `.bak.YYYYMMDD_*` files exist. These are reference snapshots — don't edit them and don't assume they reflect current behavior. Active scripts are the unsuffixed names.
- **Don't bypass `run_pipeline.sh`'s output paths.** Downstream steps look in the exact `results/{step_num}-{name}/{mode}/{model_name}/{hook_dir}/` layout (with `_gate_up`, `_attn`, `_iw` suffixes). Reproduce that layout if you add a new step.
- **Idempotence**: every step gate checks for existing outputs (`*.done` markers or final files) and skips PEND/RUN jobs by LSF name. Preserve this pattern — `--clean N` is the supported way to force re-run.
- **Sharding**: classify (step 3) shards by layer (one shard per layer, up to `N_LAYERS=32`). Description generation (step 1) shards by image index. Ablation (steps 7, 24) shards by GPU within each ablation config.
- **`--mode test`** is the smoke test: 6 fig3 images, 2 layers, `top_n=5`, 100 permutations, 1 GPU, no bsub if combined with `--local`. Use it before launching a 32-GPU run.

## Reference

`code/numerical_verification.md` is a checklist of paper-claim → cluster-lookup commands (ECCV #9948 rebuttal). It documents which result JSONs back each numeric claim in the paper — useful when verifying that a change preserved published numbers.
