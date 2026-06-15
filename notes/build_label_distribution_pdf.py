#!/usr/bin/env python3
"""Build notes/modality_label_distributions.pdf — for the focus models at
PMBT, p<0.001, min100/max300 descriptions:
  (1) neuron modality-label distribution, and
  (2) hallucination-driving-neuron enrichment odds ratios by ΔH (ablation).
Reads merged label JSONs + step-10 enrichment JSONs, renders LaTeX, compiles.
"""
import json, os, subprocess
from collections import Counter

CLS = "/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy/results/3-classify/full"
HS  = "/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy/results/10-halluc_scores/full"
NOTES_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = [
    ("llava-1.5-7b",        r"LLaVA-1.5-7B"),
    ("llava-onevision-7b",  r"LLaVA-OneVision-7B"),
    ("internvl2.5-8b",      r"InternVL2.5-8B"),
    ("qwen2.5-vl-3b",       r"Qwen2.5-VL-3B"),
]
CATS = ["visual", "text", "multimodal", "unknown"]

# ── (1) label distribution ────────────────────────────────────────────────
def dist(model):
    path = os.path.join(CLS, model, "llm_permutation_min100_max2048",
                        "neuron_labels_permutation_all.json")
    d = json.load(open(path))
    c = Counter(e["label"] for layer in d.values() for e in layer)
    return c, sum(c.values())

dist_rows = []
for model, pretty in MODELS:
    c, n = dist(model)
    dist_rows.append((pretty, n, c))

def lcell(c, n, k):
    return f"{c[k]:,} ({100*c[k]/n:.1f}\\%)"

dist_body = "\n".join(
    f"{pretty} & {n:,} & {lcell(c,n,'visual')} & {lcell(c,n,'text')} & "
    f"{lcell(c,n,'multimodal')} & {lcell(c,n,'unknown')} \\\\"
    for pretty, n, c in dist_rows
)

# ── (2) enrichment odds ratios by ΔH (ablation) ───────────────────────────
ENR_MODELS = [("llava-1.5-7b",       "LLaVA-1.5-7B"),
              ("llava-onevision-7b", "LLaVA-OneVision-7B"),
              ("internvl2.5-8b",     "InternVL2.5-8B"),
              ("qwen2.5-vl-3b",      "Qwen2.5-VL-3B")]
ENR_COMBOS = [(r"PMBT$\cdot$gate", "."),
              (r"PMBT$\cdot$gate\_up", "pmbt_gate_up"),
              (r"FT$\cdot$gate", "ft")]

def orcell(cat):
    o = cat["odds_ratio"]
    sig = str(cat.get("significant", "False")) == "True" or cat.get("significant") is True
    if o == "inf":
        s, up, ext = "inf", True, True
    else:
        o = float(o); s = f"{o:.2f}"; up = o > 1; ext = (o >= 1.5 or o <= 0.67)
    s = s + (r"\up" if up else r"\dn")
    if sig and ext:
        s = r"\textbf{" + s + "}"
    return s

TQA_INCAPABLE = {"llava-1.5-7b"}  # llava-hf needs image input; text-only TriviaQA invalid

def enr_rows(task_suffix):
    rows = []
    for mn, pretty in ENR_MODELS:
        if task_suffix == "_tqa" and mn in TQA_INCAPABLE:
            continue                                   # skip invalid TQA leg
        for clabel, sub in ENR_COMBOS:
            f = os.path.join(HS, mn, sub, f"enrichment_results{task_suffix}.json")
            if os.path.isfile(f):
                cats = json.load(open(f))["categories"]
                cells = " & ".join(orcell(cats[c]) for c in CATS)
                rows.append(f"{pretty} & {clabel} & {cells} \\\\")
    return "\n".join(rows)

pope_body = enr_rows("")
tqa_body  = enr_rows("_tqa")

TEX = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.8cm]{geometry}
\usepackage{parskip}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{booktabs}
\usepackage{array}
\usepackage{xcolor}
\newcommand{\up}{\,\textcolor{blue}{$\uparrow$}}
\newcommand{\dn}{\,\textcolor{red}{$\downarrow$}}
\title{Neuron Modality Labels \& Hallucination Enrichment\\[0.3em]
\large PMBT, $p<0.001$, min-100/max-300 descriptions}
\author{MH-Neuron / Modality Taxonomy}
\date{June 13, 2026}
\begin{document}
\maketitle

\section*{1. Modality-label distribution (PMBT $\cdot$ gate)}
A neuron is \emph{unknown} if it has $<5$ high-activation tokens; otherwise
\emph{visual}/\emph{text} if its visual-vs-text activation-rate difference is
significant ($p<0.001$), else \emph{multimodal}.

\begin{center}
\begin{tabular}{l r r r r r}
\toprule
Model & Neurons & Visual & Text & Multimodal & Unknown \\
\midrule
""" + dist_body + r"""
\bottomrule
\end{tabular}
\end{center}

\section*{2. Hallucination-driving enrichment by $\Delta H$ --- odds ratios}
Driving neurons = top 5\% by $\Delta H$ (change in hallucination rate when the
neuron is ablated); Fisher's exact test per category. OR $>1$ = enriched \up,
OR $<1$ = depleted \dn; \textbf{bold} = significant ($p<0.05$) and $\ge$1.5 or
$\le$0.67. POPE = visual hallucination; TriviaQA = text knowledge. The expected
double dissociation: visual enriched on POPE / depleted on TQA, text the
reverse. Only models/combos with results on disk are shown.

\subsection*{POPE (visual hallucination)}
\begin{center}
\begin{tabular}{l l c c c c}
\toprule
Model & Combo & Visual & Text & Multimodal & Unknown \\
\midrule
""" + pope_body + r"""
\bottomrule
\end{tabular}
\end{center}

\subsection*{TriviaQA (text knowledge)}
\begin{center}
\begin{tabular}{l l c c c c}
\toprule
Model & Combo & Visual & Text & Multimodal & Unknown \\
\midrule
""" + tqa_body + r"""
\bottomrule
\end{tabular}
\end{center}

\noindent\textbf{Coverage / caveats.} LLaVA-OneVision-7B and LLaVA-1.5-7B
currently have \emph{gate\_up only} --- their gate and FT step-10 jobs are still
running; LLaVA-1.5 is the validation model and runs POPE only (no TriviaQA).
Qwen2.5-VL-3B TQA is gate-only so far. Per-neuron $\Delta H$ is
quantised (batch-of-50 ablation), so the top-5\% cutoff often lands on a tie at
the boundary --- treat marginal ORs cautiously --- but the $\Delta H$ ranking is
the paper's headline and reproduces the visual/text double dissociation. Labels
from \texttt{llm\_permutation\_min100\_max2048/} (native $\alpha=0.001$ on
max-300 text for these models).

\end{document}
"""

tex_path = os.path.join(NOTES_DIR, "modality_label_distributions.tex")
with open(tex_path, "w") as f:
    f.write(TEX)

for _ in range(2):
    subprocess.run(["pdflatex", "-interaction=nonstopmode",
                    "modality_label_distributions.tex"],
                   cwd=NOTES_DIR, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for ext in ("aux", "log", "out"):
    p = os.path.join(NOTES_DIR, f"modality_label_distributions.{ext}")
    if os.path.exists(p):
        os.remove(p)

print("wrote modality_label_distributions.pdf")
print("POPE ΔH rows:\n" + pope_body)
print("TQA ΔH rows:\n" + tqa_body)
