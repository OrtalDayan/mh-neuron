#!/usr/bin/env python3
"""Build notes/enrichment_families.pdf — the spec's family-A/B enrichment report.

Family A = BINARY (causal "hallucination-driving"): POPE binary hallucination ΔH
+ TriviaQA factual error rate ΔErr_TQA. Degenerate by construction (single-batch
ablation rarely flips a discrete answer → ~all-zero → top-k is an argsort
tie-break), so shown with the flag, NOT as a valid ranking.

Family B = DENSE (degeneracy-robust "answer-influence"): POPE soft P(yes) +
TriviaQA 1-P_gold. Continuous → tie-free. Computed by the running dense campaign;
populated here once .../dense/enrichment_results*.json land.

Re-run this script to refresh as results land.
"""
import json, os, numpy as np, subprocess

HS = "/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy/results/10-halluc_scores/full"
CLS = "/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy/results/3-classify/full"
NOTES = os.path.dirname(os.path.abspath(__file__))

MODELS = [("llava-onevision-7b","LLaVA-OneVision-7B"),
          ("internvl2.5-8b","InternVL2.5-8B"),
          ("qwen2.5-vl-3b","Qwen2.5-VL-3B"),
          ("llava-1.5-7b","LLaVA-1.5-7B")]
COMBOS = [(r"PMBT$\cdot$gate","."),(r"PMBT$\cdot$gate\_up","pmbt_gate_up"),(r"FT$\cdot$gate","ft")]
CATS = ["visual","text","multimodal","unknown"]
TQA_INCAPABLE = {"llava-1.5-7b"}

def orcell(c):
    o=c["odds_ratio"]; sig=str(c.get("significant","False"))=="True" or c.get("significant") is True
    if o=="inf": s,up,ext="inf",True,True
    else: o=float(o); s=f"{o:.2f}"; up=o>1; ext=(o>=1.5 or o<=0.67)
    s=s+(r"\up" if up else r"\dn")
    if sig and o!="inf" and ext: s=r"\textbf{"+s+"}"
    return s

def load(model, sub, task_suffix):
    f=os.path.join(HS,model,sub,f"enrichment_results{task_suffix}.json")
    if not os.path.isfile(f): return None
    return json.load(open(f))["categories"]

def family_table(task_suffix, family):
    rows=[]
    for mn,pretty in MODELS:
        for clabel,sub in COMBOS:
            if task_suffix=="_tqa" and mn in TQA_INCAPABLE: continue
            cats=load(mn,sub,task_suffix)
            if cats is None: continue
            cells=" & ".join(orcell(cats[c]) for c in CATS)
            flag = r"\textsc{deg}" if family=="A" else "ok"
            rows.append(f"{pretty} & {clabel} & {cells} & {flag} \\\\")
    return "\n".join(rows) if rows else r"\multicolumn{7}{c}{\emph{none on disk yet}} \\"

# ── sample-size / pool composition (spec output #4) ──
def sizes(model):
    import collections
    pf=os.path.join(HS,model,"contrastive_pope.jsonl")
    tf=os.path.join(HS,model,"contrastive_triviaqa.jsonl")
    pope_no=pope_n=tqa_n=tqa_c=tqa_i=0
    if os.path.isfile(pf):
        rows=[json.loads(l) for l in open(pf) if l.strip()]; pope_n=len(rows)
        pope_no=sum(1 for r in rows if (r.get('label') or r.get('answer','')).strip().lower()=='no')
    if os.path.isfile(tf):
        rows=[json.loads(l) for l in open(tf) if l.strip()]; tqa_n=len(rows)
        tqa_c=sum(1 for r in rows if r.get('contrastive_label')=='correct')
        tqa_i=tqa_n-tqa_c
    return pope_n,pope_no,tqa_n,tqa_c,tqa_i

size_rows=[]
for mn,pretty in MODELS:
    pn,pno,tn,tc,ti=sizes(mn)
    if pn or tn:
        step = f"{1/max(tn,1):.4f}" if tn else "--"
        size_rows.append(f"{pretty} & {pno} & {tn} & {tc}/{ti} & {step} \\\\")
size_body="\n".join(size_rows) if size_rows else r"\multicolumn{5}{c}{--} \\"

# dense coverage status
DENSE_DIRS=[(m,s) for m,_ in MODELS for s in (".","pmbt_gate_up")]
dense_done=sum(1 for m,s in DENSE_DIRS if os.path.isfile(os.path.join(HS,m,"dense",("" if s=="." else s),"enrichment_results.json")))

TEX = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=1.6cm]{geometry}\usepackage{parskip}\usepackage[T1]{fontenc}
\usepackage{lmodern}\usepackage{booktabs}\usepackage{array}\usepackage{xcolor}
\newcommand{\up}{\,\textcolor{blue}{$\uparrow$}}\newcommand{\dn}{\,\textcolor{red}{$\downarrow$}}
\title{Enrichment: Family A (binary) vs Family B (dense)\\[0.3em]
\large PMBT, $p<0.001$, max-300 descriptions}
\author{MH-Neuron / Modality Taxonomy}\date{June 15, 2026}
\begin{document}\maketitle

\noindent\textbf{Two metric families, reported side by side (not mixed).}
\emph{Family A — binary} preserves the paper's causal ``hallucination-driving''
meaning: POPE hallucination $\Delta H$ (false-yes on gt=no) and TriviaQA factual
error rate $\Delta\mathrm{Err}_{\mathrm{TQA}}$. Single-batch ablation rarely flips
a discrete answer, so these arrays are $\sim$all-zero and the top-5\% set is an
\textsc{argsort} tie-break --- \textbf{flagged \textsc{deg}; not a valid ranking},
shown only for continuity. \emph{Family B — dense} (soft $P(\text{yes})$;
$1-P_{\text{gold}}$) is continuous/tie-robust and is the \emph{primary} result;
it is being computed by a running ablation campaign (""" + f"{dense_done}/8 dense combos landed" + r""").
OR$>$1 enriched \up, $<$1 depleted \dn; \textbf{bold} = significant \& $\ge$1.5/$\le$0.67.

\section*{Family A --- POPE binary hallucination $\Delta H$}
\begin{center}\small\begin{tabular}{l l c c c c c}
\toprule Model & Combo & Visual & Text & Multimodal & Unknown & Guard \\ \midrule
""" + family_table("","A") + r"""
\bottomrule\end{tabular}\end{center}

\section*{Family A --- TriviaQA binary factual error $\Delta\mathrm{Err}_{\mathrm{TQA}}$}
\begin{center}\small\begin{tabular}{l l c c c c c}
\toprule Model & Combo & Visual & Text & Multimodal & Unknown & Guard \\ \midrule
""" + family_table("_tqa","A") + r"""
\bottomrule\end{tabular}\end{center}
\noindent\footnotesize(LLaVA-1.5 omitted from TQA --- text-only TriviaQA is invalid for that backend.)\normalsize

\section*{Family B --- DENSE (POPE soft $P(\text{yes})$ / TQA $1-P_{\text{gold}}$)}
\emph{Pending the running dense campaign} (\texttt{...\,/dense/} per-layer ablation
$\to$ \texttt{10C\_*\_dn} enrichment). Validated live: dense $\Delta H$ is
continuous and distinct per batch (40 distinct values, both signs, 0 zeros) ---
i.e. the guard will read \emph{not} degenerate. Re-run this script to populate
the dense tables once the results land.

\section*{TriviaQA sample-size \& pool composition (spec output \#4)}
Effective $n$: POPE scores gt=no only; TQA scores the full pool. The TQA pool is
$\sim$half POPE's and coarsely quantized (step $=1/n$), so a flagged binary TQA
side is ``ranking probably invalid'', not ``slightly noisy''.
\begin{center}\small\begin{tabular}{l c c c c}
\toprule Model & POPE $n$ (gt=no) & TQA $n$ & TQA corr/incorr & TQA bin step \\ \midrule
""" + size_body + r"""
\bottomrule\end{tabular}\end{center}

\end{document}
"""
open(os.path.join(NOTES,"enrichment_families.tex"),"w").write(TEX)
for _ in range(2):
    subprocess.run(["pdflatex","-interaction=nonstopmode","enrichment_families.tex"],
                   cwd=NOTES, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for ext in ("aux","log","out"):
    p=os.path.join(NOTES,f"enrichment_families.{ext}")
    if os.path.exists(p): os.remove(p)
print(f"wrote enrichment_families.pdf  (dense combos landed: {dense_done}/8)")
