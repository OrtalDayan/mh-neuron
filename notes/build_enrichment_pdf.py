#!/usr/bin/env python3
"""Build notes/enrichment_results.pdf, one section per model.

qwen2.5-vl-3b is a fixed, hand-curated section (its TQA numbers were computed
offline into _tqa_validate/_tqa_scratch, not the standard path). Every other
model is auto-generated from whatever enrichment JSONs currently exist on disk,
so re-running this script picks up combos as they finish. Missing combos/tasks
are rendered as "pending".

Usage:  python3 build_enrichment_pdf.py        # writes .tex and compiles .pdf
"""
import json
import os
import subprocess
import sys

NOTES_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS = ("/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy/"
           "results/10-halluc_scores/full")

# Auto-generated models, in report order. (qwen is the fixed section below.)
AUTO_MODELS = ["internvl2.5-8b", "llava-onevision-7b", "llava-1.5-7b"]

# combo label -> subdirectory under the model dir
COMBOS = [("PMBT $\\cdot$ gate", "."),
          ("PMBT $\\cdot$ gate\\_up", "pmbt_gate_up"),
          ("FT $\\cdot$ gate", "ft")]

CATEGORIES = ["visual", "text", "multimodal", "unknown"]


def load(model, sub, task):
    """Return the categories dict for one (model, combo, task) or None.

    Reads the *dense* 'combined' score (single-neuron ablation ΔH + CETT-diff),
    falling back to the CETT-diff-only file. The plain 'enrichment_results.json'
    (ablation ΔH alone) is intentionally NOT used: per-neuron POPE ablation rarely
    flips a binary answer, so that file is ~99.9% exact zeros and its top-k
    "driving" set degenerates into an argsort index tie-break over the last layers
    (an artifact, not hallucination). See the guard in
    halluc_score_neurons.compute_enrichment.
    """
    if task == "pope":
        candidates = ["enrichment_results_combined.json",
                      "enrichment_results_cett_diff.json"]
    else:
        candidates = ["enrichment_results_combined_tqa.json",
                      "enrichment_results_cett_diff_tqa.json"]
    for name in candidates:
        path = os.path.join(RESULTS, model, sub, name)
        if os.path.isfile(path):
            with open(path) as fh:
                return json.load(fh)
    return None


def fmt_cell(cat):
    """Format one category's odds ratio + p-value the way the qwen tables do."""
    orr = cat["odds_ratio"]
    p = cat["p_value"]
    sig = str(cat.get("significant", "True")) == "True"
    if not sig:
        pstr = f"{p:.1g}, n.s."
    elif p == 0.0 or p < 1e-300:
        pstr = "$\\approx$0"
    elif p < 1e-4:
        pstr = f"{p:.0e}"             # e.g. 3e-246
    else:
        pstr = f"{p:g}"               # e.g. 0.01
    bold = sig and (orr >= 1.5 or orr <= 0.1)
    orstr = f"\\textbf{{{orr:.2f}}}" if bold else f"{orr:.2f}"
    return f"{orstr} ({pstr})"


def arrow(orr):
    return "\\up" if orr > 1 else "\\dn"


def or_table(model, task):
    """POPE/TQA odds-ratio table over the combos that have this task."""
    present = [(lbl, sub, load(model, sub, task)) for lbl, sub in COMBOS]
    present = [(lbl, sub, d) for lbl, sub, d in present if d is not None]
    if not present:
        return None, []
    cols = "l" + "c" * len(present)
    header = " & ".join(["category"] + [lbl for lbl, _, _ in present])
    lines = ["\\begin{center}", f"\\begin{{tabular}}{{{cols}}}", "\\toprule",
             header + " \\\\", "\\midrule"]
    for c in CATEGORIES:
        row = [c]
        for _, _, d in present:
            row.append(fmt_cell(d["categories"][c]))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    return "\n".join(lines), present


def dissociation_table(model):
    """visual/text OR across combos that have BOTH pope and tqa."""
    rows = []
    for lbl, sub in COMBOS:
        p = load(model, sub, "pope")
        t = load(model, sub, "tqa")
        if p is None or t is None:
            continue
        vp = p["categories"]["visual"]["odds_ratio"]
        vt = t["categories"]["visual"]["odds_ratio"]
        tp = p["categories"]["text"]["odds_ratio"]
        tt = t["categories"]["text"]["odds_ratio"]
        rows.append((lbl, vp, vt, tp, tt))
    if not rows:
        return None
    lines = ["\\begin{center}", "\\begin{tabular}{l cc cc}", "\\toprule",
             " & \\multicolumn{2}{c}{\\textbf{visual} OR} & "
             "\\multicolumn{2}{c}{\\textbf{text} OR} \\\\",
             "combo & POPE & TQA & POPE & TQA \\\\", "\\midrule"]
    for lbl, vp, vt, tp, tt in rows:
        lines.append(f"{lbl} & {vp:.2f} {arrow(vp)} & {vt:.2f} {arrow(vt)} & "
                     f"{tp:.2f} {arrow(tp)} & {tt:.2f} {arrow(tt)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{center}"]
    return "\n".join(lines)


def model_section(model):
    out = [f"\\section{{Model: \\texttt{{{tex_escape(model)}}}}}"]
    # status line: which combos/tasks are present
    status = []
    for lbl, sub in COMBOS:
        p = load(model, sub, "pope") is not None
        t = load(model, sub, "tqa") is not None
        tag = lbl.replace("$\\cdot$", "·")
        status.append(f"{tag}: POPE={'y' if p else '--'}, TQA={'y' if t else '--'}")
    out.append("\\emph{Combos available: " + "; ".join(status) + ".}\n")

    pope_tbl, pope_present = or_table(model, "pope")
    if pope_tbl is None:
        out.append("\\emph{No enrichment results on disk yet --- layers still "
                   "running.}")
        return "\n".join(out)

    n_driving = pope_present[0][2]["n_driving"]
    n_total = pope_present[0][2]["n_total"]
    out.append(f"top-$k$ = 5\\%, $n_{{\\text{{driving}}}}={n_driving:,}$ of "
               f"{n_total:,} total neurons (first available combo).\n")

    out.append("\\subsection{POPE (visual task) --- odds ratios}")
    out.append("Hypothesis: \\emph{visual} neurons enriched. $p$ in "
               "parentheses; $\\approx$0 denotes $p<10^{-300}$.")
    out.append(pope_tbl)

    tqa_tbl, _ = or_table(model, "tqa")
    out.append("\\subsection{TriviaQA (text task) --- odds ratios}")
    if tqa_tbl is None:
        out.append("\\emph{Pending --- no TQA enrichment for any completed "
                   "combo yet.}")
    else:
        out.append("Hypothesis: \\emph{text} neurons enriched.")
        out.append(tqa_tbl)

    diss = dissociation_table(model)
    out.append("\\subsection{Double dissociation}")
    if diss is None:
        out.append("\\emph{Pending --- needs at least one combo with both POPE "
                   "and TriviaQA enrichment.}")
    else:
        out.append("\\up{} = enriched (OR$>$1), \\dn{} = depleted (OR$<$1). "
                   "Only combos with both tasks scored are shown.")
        out.append(diss)
    return "\n".join(out)


def tex_escape(s):
    return s.replace("_", "\\_")


PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{parskip}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{array}
\usepackage{xcolor}
\usepackage{titlesec}
\titleformat{\section}{\large\bfseries}{\thesection}{0.6em}{}
\titleformat{\subsection}{\normalsize\bfseries}{\thesubsection}{0.6em}{}
\newcommand{\up}{\textcolor{blue}{$\uparrow$}}
\newcommand{\dn}{\textcolor{red}{$\downarrow$}}

\title{Enrichment Analysis Results --- All Models\\[0.3em]
\large Modality labels vs.\ hallucination-driving neurons}
\author{MH-Neuron / Modality Taxonomy (step 10, halluc\_score)}
\date{June 13, 2026}

\begin{document}
\maketitle

\section*{Scope and status}
One section per model. Sections are regenerated from the enrichment JSONs on
disk by \texttt{build\_\allowbreak enrichment\_\allowbreak pdf.py}; re-running it picks up new combos as
their cluster jobs finish. \texttt{qwen2.5-vl-3b} is complete; the other models
are filled in incrementally and any combo not yet on disk is marked pending.

\textbf{What the numbers mean.} For each modality category (visual / text /
multimodal / unknown) we ablate the top 5\% of neurons by $\Delta$Hallucination
and run Fisher's exact test on whether that category is over-represented among
those hallucination-driving neurons. The \emph{odds ratio} (OR) is the headline:
OR $>1$ = enriched, OR $<1$ = depleted. Two tasks are scored: \textbf{POPE}
(visual object hallucination) and \textbf{TriviaQA} (text knowledge errors).
"""

# ---- Fixed, hand-curated qwen2.5-vl-3b section (TQA computed offline) --------
QWEN_SECTION = r"""
\section{Model: \texttt{qwen2.5-vl-3b}}
\emph{Complete --- all three label/hook combos.} All combos: top-$k$ = 5\%,
$n_{\text{driving}}=19{,}814$ of $396{,}288$ total neurons.

\subsection{POPE (visual task) --- odds ratios}
Hypothesis: \emph{visual} neurons should be enriched (ablating them blinds the
model). $p$-values in parentheses; $\approx$0 denotes $p<10^{-300}$.

\begin{center}
\begin{tabular}{lccc}
\toprule
category & PMBT $\cdot$ gate & PMBT $\cdot$ gate\_up & FT $\cdot$ gate \\
\midrule
visual     & \textbf{2.02} ($\approx$0)      & \textbf{1.38} (7e-106) & 1.18 (5e-25) \\
text       & 0.81 (3e-38)        & 0.91 (4e-9)            & 0.78 (1e-35) \\
multimodal & 1.04 (0.09, n.s.)   & 0.93 (8e-4)            & 0.93 (2e-5)  \\
unknown    & \textbf{0.03} ($\approx$0)      & \textbf{0.02} ($\approx$0)         & 1.06 (8e-5)  \\
\bottomrule
\end{tabular}
\end{center}

\subsection{TriviaQA (text task) --- odds ratios}
Hypothesis: \emph{text} neurons should be enriched (ablating them erases
knowledge). These were computed from the merged $\Delta H_{\text{TQA}}$ scores
(see Caveats).

\begin{center}
\begin{tabular}{lccc}
\toprule
category & PMBT $\cdot$ gate & PMBT $\cdot$ gate\_up & FT $\cdot$ gate \\
\midrule
visual     & 0.88 (3e-19)        & 1.40 (1e-115) & 1.48 (7e-140) \\
text       & \textbf{1.21} (1e-36)  & 0.86 (7e-23)  & 1.05 (0.01)   \\
multimodal & 0.85 (8e-11)        & 0.99 (0.6, n.s.) & 1.10 (1e-7) \\
unknown    & 1.03 (0.2, n.s.)    & 0.08 ($\approx$0) & 0.58 (1e-232) \\
\bottomrule
\end{tabular}
\end{center}

\subsection{The double dissociation (the key result)}
The dual-source design predicts a diagonal flip: visual neurons enriched on
POPE but \emph{not} TriviaQA, text neurons enriched on TriviaQA but \emph{not}
POPE. \up{} = enriched (OR$>$1), \dn{} = depleted (OR$<$1).

\begin{center}
\begin{tabular}{l cc cc}
\toprule
 & \multicolumn{2}{c}{\textbf{visual} OR} & \multicolumn{2}{c}{\textbf{text} OR} \\
combo & POPE & TQA & POPE & TQA \\
\midrule
\textbf{PMBT $\cdot$ gate}    & 2.02 \up & 0.88 \dn & 0.81 \dn & 1.21 \up \\
PMBT $\cdot$ gate\_up         & 1.38 \up & 1.40 \up & 0.91 \dn & 0.86 \dn \\
FT $\cdot$ gate               & 1.18 \up & 1.48 \up & 0.78 \dn & 1.05 \up \\
\bottomrule
\end{tabular}
\end{center}

\textbf{Only PMBT $\cdot$ gate shows the clean flip} (visual \up\dn{},
text \dn\up). The other two do not: under PMBT gate\_up and FT gate, visual
neurons are enriched for \emph{both} tasks, so the modality axis is not
separated.

\subsection{Interpretation}

\textbf{Taxonomy: PMBT $\gg$ FT (same gate hook, POPE).}
PMBT concentrates visual function far harder (OR \textbf{2.02} vs FT's 1.18) and
\textbf{purges the ``unknown'' bucket to 0.03}, whereas FT leaves unknown
slightly \emph{enriched} at \textbf{1.06}. So FT's catch-all category still hides
hallucination-driving neurons that PMBT correctly reclassifies as visual. This is
the functional-validity argument for the permutation-test labels.

\textbf{Hook: gate $>$ gate\_up for separation.}
Both PMBT hooks share the qualitative signature (visual enriched, unknown
purged $\approx$0.02--0.03), confirming the effect under an \emph{independent}
labeling. But the \texttt{gate} hook gives sharper visual enrichment (2.02 vs
1.38) \emph{and} is the only one that yields the double dissociation; the
\texttt{gate\_up} intermediate captures visual function but does not separate
the text axis (visual enriched on TriviaQA too, OR 1.40).

\textbf{Headline.}
On identical neurons, ablation scores, and statistics, \textbf{PMBT labels at the
gate hook are the only configuration that (i) maximises visual enrichment, (ii)
empties the ``unknown'' bucket, and (iii) produces the visual/text double
dissociation across POPE and TriviaQA.} FT labels achieve none of these cleanly.

\subsection{Caveats}
\begin{itemize}
\item \textbf{TriviaQA enrichment was computed offline.} The 10C aggregation
historically wrote only the POPE \texttt{enrichment\_results.json}; the merged
$\Delta H_{\text{TQA}}$ existed (\texttt{ablation\_scores\_tqa.json}) but was not
run through enrichment. The TQA numbers here were produced by feeding that file
to the same \texttt{compute\_enrichment} routine. A pipeline fix (forwarding the
TQA scores so \texttt{enrichment\_results\_tqa.json} is emitted automatically)
has been applied for the remaining models.
\item Multimodal results are mostly non-significant or weak and are not load-bearing.
\item ``gate'' and ``gate\_up'' label \emph{different} neuron populations
(different hook point), so cross-hook agreement is corroboration by an
independent labeling, not a same-neuron comparison.
\end{itemize}
"""


def main():
    parts = [PREAMBLE, QWEN_SECTION]
    for m in AUTO_MODELS:
        parts.append(model_section(m))
    parts.append("\n\\end{document}\n")
    tex = "\n".join(parts)

    tex_path = os.path.join(NOTES_DIR, "enrichment_results.tex")
    with open(tex_path, "w") as fh:
        fh.write(tex)

    for _ in range(2):
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode",
                            "enrichment_results.tex"], cwd=NOTES_DIR,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.stderr.write(r.stdout.decode(errors="replace")[-2000:])
        sys.exit("pdflatex failed")
    for ext in ("aux", "log", "out"):
        f = os.path.join(NOTES_DIR, f"enrichment_results.{ext}")
        if os.path.exists(f):
            os.remove(f)
    print("wrote enrichment_results.pdf")


if __name__ == "__main__":
    main()
