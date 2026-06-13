# mh-neuron

Code, experiments, and figures for our study of **modality-specialized neurons** in the
LLM backbone of vision-language models (VLMs), and how a single neuron taxonomy can be
reused to diagnose and reduce hallucination.

At the center is **PMBT** (the *Permutation-Based Modality Test*), a non-parametric method
that labels each LLM-backbone neuron as **visual**, **text**, **multimodal**, or
**unknown**. The same labels then drive three downstream applications: enrichment analysis,
activation steering, and taxonomy-guided weight merging.

> **Note:** replace the bracketed placeholders below (title, authors, venue, citation,
> license) before making the repository public.

---

## Demo

https://github.com/user-attachments/assets/00a189e9-3eb8-4182-87ec-56b53f1610e1


Interactive animation of how the PMBT mask is applied across the LLM backbone of a VLM — classification → enrichment → activation steering → weight merging.

▶ **[Live demo](https://ortaldayan.github.io/mh-neuron/modality_taxonomy/pmbt_mask_demo.html)** · [view source](modality_taxonomy/pmbt_mask_demo.html)

The animation's steering numbers (POPE / CHAIRi on LLaVA-OneVision) are real results; the
enrichment odds ratios and merging deltas shown are **illustrative/schematic** and labelled
as such in the demo.

---

## Method

Prior work (Xu et al., *Deciphering Functions of Neurons in Vision-Language Models*)
classifies backbone neurons as visual / text / multimodal using four hand-tuned thresholds
(activation magnitudes and minimum token counts) that lack statistical justification, which
inflates the multimodal category.

PMBT replaces those four thresholds with:
- an **adaptive per-neuron threshold** (Otsu's method), and
- a **non-parametric permutation test** (1,000 permutations, α = 0.05),

yielding a **calibrated p-value per neuron** and a single, principled significance level.

We further define **MH-Neurons** (Modality-specialized Hallucination Neurons): neurons that
are both confidently modality-specific under PMBT *and* identified as hallucination-driving
by causal ablation scoring.

### Three uses of one taxonomy

1. **Enrichment analysis** — cross-reference the labels with hallucination-driving neurons
   and test each modality group for over-representation via an odds ratio.
2. **Activation steering** — scale the activations of a single modality group at inference
   (no retraining, no weight edits). On LLaVA-OneVision, scaling visual-neuron activations
   at α = 0.5 improves POPE (90.2 → 92.0) and lowers CHAIRi (26.1 → 24.3).
3. **Weight merging** — inject a task vector onto a chosen mask (e.g. text neurons only)
   while freezing the others, so a target capability can be added with less collateral
   damage than uniform task-arithmetic merging.

Ablation establishes a **double dissociation**: ablating visual neurons degrades visual
grounding benchmarks (POPE, CHAIR) while preserving text benchmarks (TriviaQA, MMLU), and
vice versa for text neurons.

---

## Models

PMBT is evaluated across several architectures:

| VLM | LLM backbone | Vision encoder |
|---|---|---|
| LLaVA-1.5-7B | LLaMA-2 | CLIP |
| LLaVA-OneVision-7B | Qwen2 | SigLIP |
| InternVL2.5-8B | InternLM2 | InternViT |
| Qwen2.5-VL-7B | Qwen2 | native ViT |

(LLaVA-Next-LLaMA3-8B is additionally used for direct comparison with uniform-merging
baselines.)

---

## Repository layout

```
mh-neuron/
├── modality_taxonomy/
│   ├── code/                 # pipeline scripts (run_pipeline.sh is the entry point)
│   │   └── run_pipeline.sh   # orchestrates all numbered steps
│   ├── pmbt_mask_demo.html   # interactive demo (see Demo above)
│   └── ...                   # figures, result tables, analysis outputs
├── LLaVA/                    # LLaVA dependency (editable install)
└── modern_vlms/              # modern-VLM evaluation (VLMEvalKit, InternVL env)
```

Pipeline outputs follow the convention
`results/{step_num}-{name}/{mode}/{model_name}/...`, which downstream steps rely on.

---

## Usage

`modality_taxonomy/code/run_pipeline.sh` is the single entry point; individual Python
scripts are dispatchable units beneath it. Steps are run by number, for example:

```bash
# Build the neuron taxonomy
bash code/run_pipeline.sh --step 1   # describe: VLM descriptions over COCO images
bash code/run_pipeline.sh --step 2   # merge description shards
bash code/run_pipeline.sh --step 3   # classify: FT + PMBT neuron classification
bash code/run_pipeline.sh --step 4   # merge classification shards
```

The pipeline is organized into phases:

| Phase | Steps | Purpose |
|---|---|---|
| Taxonomy | 1–4 | descriptions → FT + PMBT neuron classification |
| Validation | 5–8 | ablation / double dissociation, visualization, statistics |
| Hallucination | 10–15 | per-neuron hallucination scoring, enrichment, activation steering |
| Weight merging | 16–25 | task-vector injection on PMBT masks, composition, evaluation |
| Analysis | 26–29 | weight-change correlations, effective-rank analysis |

A quick smoke test can be run with `--mode test`. Classification supports two label flavors
(FT and PMBT) and multiple hook points (gate / gate_up / attention).

> Environment setup (Python virtual environments, the editable LLaVA install, and the
> shared Hugging Face cache) is project-specific — see `CLAUDE.md` / `SESSION_HANDOFF.md`
> in the repository for the current setup notes.

---

## Citation

```bibtex
@article{[CITEKEY],
  title   = {[PAPER TITLE]},
  author  = {[AUTHORS]},
  journal = {[VENUE]},
  year    = {[YEAR]}
}
```

## License

[Specify a license — e.g. MIT, Apache-2.0 — before publishing.]
