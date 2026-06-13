#!/usr/bin/env python3
"""Generate the filled LaTeX down_proj table from the score CSVs on disk.

Reads eval results from results/25-merge/{model}/{math_llm}/{config}/eval/
for each of the 3 (VLM, math_LLM) pairs, and writes a filled version of
downproj_table.tex with all TBD placeholders replaced.

Usage:
    python3 code/generate_downproj_table.py > paper/downproj_table.tex
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Reuse the existing extractor for score lookup
import extract_merge_results as E

# ─── Configuration ───────────────────────────────────────────────────
MODELS = [
    ('LLaVA-Next',  'results/25-merge/llava-next-llama3-8b/dart-prop',
     'pmbt_t0.7_v1.0_m1.0',  'uniform_a0.8'),   # winner full-PMBT config, winner uniform
    ('Idefics2',    'results/25-merge/idefics2-8b/mammoth1',
     'pmbt_t0.8_v1.0_m0.9',  'uniform_a0.9'),
    ('Qwen2-VL',    'results/25-merge/qwen2-vl-7b/qwen2-math',
     'pmbt_t0.9_v1.0_m1.0_o1.0', 'uniform_a0.9'),
]

DOWNPROJ_SUFFIX = '_mlponly_a1.0_o1.0_downproj'

# Canonical 23 triplets from the sweep
TRIPLETS = [
    (0.3, 1.0, 0.9),  (0.3, 1.0, 1.0),
    (0.5, 1.0, 0.9),  (0.5, 1.0, 1.0),
    (0.6, 1.0, 0.9),  (0.6, 1.0, 1.0),
    (0.65, 1.0, 0.9), (0.65, 1.0, 1.0),
    (0.7, 0.9, 0.9),  (0.7, 0.95, 0.9),
    (0.7, 1.0, 0.5),  (0.7, 1.0, 0.7),  (0.7, 1.0, 0.9),  (0.7, 1.0, 1.0),
    (0.75, 1.0, 0.9), (0.75, 1.0, 1.0),
    (0.8, 1.0, 0.8),  (0.8, 1.0, 0.9),  (0.8, 1.0, 1.0),
    (0.85, 1.0, 0.9), (0.85, 1.0, 1.0),
    (0.9, 1.0, 0.9),  (0.9, 1.0, 1.0),
]


def _collect(base, subdir):
    """Return a dict of benchmark→score for a config, or None if not found."""
    full = os.path.join(base, subdir)
    if not os.path.isdir(full):
        return None
    r = E._collect_row(base, [subdir])
    r['__mm__'] = E._math_mean(r)
    r['__mm_partial__'] = E._math_mean_partial(r)
    return r


def _fmt(v, digits=1):
    if v is None:
        return '—'
    return f'{v:.{digits}f}'


def _triplet_to_cfg(t, v, m):
    """(0.7, 1.0, 1.0) → 'pmbt_t0.7_v1.0_m1.0'"""
    return f'pmbt_t{t}_v{v}_m{m}'


def _downproj_cfg(t, v, m):
    return _triplet_to_cfg(t, v, m) + DOWNPROJ_SUFFIX


def _full_cfg_candidates(t, v, m):
    """Candidates for matching full-PMBT config."""
    base = _triplet_to_cfg(t, v, m)
    return [base, base + '_o1.0', base + '_o0.9']


def _find_full(base_path, t, v, m):
    """Return the first existing full-PMBT config dir at triplet (t, v, m)."""
    for cand in _full_cfg_candidates(t, v, m):
        if os.path.isdir(os.path.join(base_path, cand)):
            return cand
    return None


def _curated_math(row):
    """Return (MathVista, MathVerse_avg, MMStar, DynaMath, MathVision, MM-Math, MathMn-10)."""
    if row is None:
        return tuple([None] * 7)
    mv  = row.get('MathVista')
    mve = [row.get(f'MVe_{s}') for s in
           ('Text_Dominant', 'Text_Lite', 'Vision_Intensive', 'Vision_Dominant', 'Vision_Only')]
    mve_vals = [x for x in mve if x is not None]
    mve_avg = sum(mve_vals) / len(mve_vals) if mve_vals else None
    mms = row.get('MMStar')
    dm  = row.get('DynaMath')
    mvis = row.get('MathVision')
    mmm = row.get('MM-Math')
    mm_mean = row.get('__mm__') or row.get('__mm_partial__')
    return (mv, mve_avg, mms, dm, mvis, mmm, mm_mean)


def _render_main_row(label_prefix, label, row, is_downproj=False):
    """Render one row of the main table."""
    mv, mve, mms, dm, mvis, mmm, mn = _curated_math(row)
    vals = [_fmt(v, 1) for v in (mv, mve, mms, dm, mvis, mmm, mn)]
    if is_downproj:
        # Bold the downproj row numbers when available
        vals = [f'\\textbf{{{v}}}' if v != '—' else v for v in vals]
        lbl = f'\\quad+ {label}'
    else:
        lbl = label
    return f'{label_prefix}& {lbl:<32s} & ' + ' & '.join(f'{v:>6s}' for v in vals) + ' \\\\\n'


def build_main_table():
    """Main table — 3 models × 4 rows each."""
    out = []
    out.append(r'\begin{table*}[t]')
    out.append(r'\centering')
    out.append(r'\small')
    out.append(r'\caption{\textbf{Down\_proj-only weight merging is sufficient to transfer math capability.}')
    out.append(r'For each VLM, we compare the baseline, BRV uniform merging, full PMBT-guided merging,')
    out.append(r'and our new down\_proj-only variant — which merges \emph{only the $W_{\text{down}}$')
    out.append(r'columns} of PMBT-labeled MLP neurons (leaving $W_{\text{gate}}$, $W_{\text{up}}$, all')
    out.append(r'attention weights, layernorms, and biases at their VLM baseline values). Down\_proj-only')
    out.append(r'uses the same PMBT $(\alpha_{\text{text}}, \alpha_{\text{visual}}, \alpha_{\text{multi}})$')
    out.append(r'triplet as the PMBT winner. All numbers are accuracy (\%). \emph{MathMn} = mean of 10')
    out.append(r'math benchmarks (6 shown + MathVerse-TD/TL/VI/VD/VO averaged).}')
    out.append(r'\label{tab:downproj-main}')
    out.append(r'\begin{tabular}{llcccccc r}')
    out.append(r'\toprule')
    out.append(r' &  & \multicolumn{6}{c}{\textbf{Math benchmarks}} & \textbf{MathMn} \\')
    out.append(r' \cmidrule(lr){3-8}')
    out.append(r'\textbf{Model} & \textbf{Method} & MathVista & MathVerse-Avg & MMStar & DynaMath & MathVision & MM-Math &       \\')
    out.append(r'\midrule')

    for model_name, base, full_name, uni_name in MODELS:
        # Parse the winner triplet from full_name
        m = re.match(r'pmbt_t([\d.]+)_v([\d.]+)_m([\d.]+)', full_name)
        if not m:
            continue
        t, v, m_val = float(m.group(1)), float(m.group(2)), float(m.group(3))
        dp_name = _downproj_cfg(t, v, m_val)

        bl    = _collect(base, 'baseline')
        uni   = _collect(base, uni_name)
        full  = _collect(base, full_name)
        dp    = _collect(base, dp_name)

        prefix = r'\multirow{4}{*}{\textbf{' + model_name + r'}} '
        out.append(_render_main_row(prefix,              'Baseline',                        bl).rstrip())
        out.append(_render_main_row('                    ', f'Uniform ($\\alpha$={uni_name.replace("uniform_a", "")})', uni).rstrip())
        pmbt_lbl = f't{t}/v{v}/m{m_val}'
        out.append(_render_main_row('                    ', f'PMBT {pmbt_lbl}', full).rstrip())
        out.append(_render_main_row('                    ', 'down\\_proj only', dp, is_downproj=True).rstrip())
        out.append(r'\midrule')

    # Remove trailing midrule, replace with bottomrule
    if out[-1] == r'\midrule':
        out[-1] = r'\bottomrule'
    else:
        out.append(r'\bottomrule')
    out.append(r'\end{tabular}')
    out.append(r'\end{table*}')
    return '\n'.join(out)


def build_sweep_table():
    """Appendix table — 23 triplets × 3 models."""
    out = []
    out.append('\n\n')
    out.append(r'\begin{table*}[t]')
    out.append(r'\centering')
    out.append(r'\small')
    out.append(r'\caption{\textbf{Down\_proj-only vs. full PMBT across the canonical 23-triplet sweep.}')
    out.append(r'Each row shows MathMn (mean of 10 math benchmarks) for the down\_proj-only variant')
    out.append(r'at that $(\alpha_t, \alpha_v, \alpha_m)$ triplet, alongside the matching full-PMBT')
    out.append(r'number and their difference. Positive $\Delta$ means down\_proj-only is better.}')
    out.append(r'\label{tab:downproj-sweep}')
    out.append(r'\begin{tabular}{l l rrr rrr rrr}')
    out.append(r'\toprule')
    out.append(r' &  & \multicolumn{3}{c}{\textbf{LLaVA-Next}} & \multicolumn{3}{c}{\textbf{Idefics2}} & \multicolumn{3}{c}{\textbf{Qwen2-VL}} \\')
    out.append(r' \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11}')
    out.append(r'\textbf{$\alpha_t$} & \textbf{($\alpha_v$, $\alpha_m$)} & DP & Full & $\Delta$ & DP & Full & $\Delta$ & DP & Full & $\Delta$ \\')
    out.append(r'\midrule')

    per_model_deltas = {m[0]: [] for m in MODELS}

    for t, v, m_val in TRIPLETS:
        row_cells = []
        for model_name, base, _, _ in MODELS:
            dp_name = _downproj_cfg(t, v, m_val)
            dp = _collect(base, dp_name)
            full_name = _find_full(base, t, v, m_val)
            full = _collect(base, full_name) if full_name else None

            dp_mn = dp.get('__mm__') if dp else None
            if dp_mn is None and dp:
                dp_mn = dp.get('__mm_partial__')
            full_mn = full.get('__mm__') if full else None
            if full_mn is None and full:
                full_mn = full.get('__mm_partial__')

            if dp_mn is not None and full_mn is not None:
                delta = dp_mn - full_mn
                per_model_deltas[model_name].append(delta)
                row_cells.extend([_fmt(dp_mn, 1), _fmt(full_mn, 1), f'{delta:+.2f}'])
            else:
                row_cells.extend([_fmt(dp_mn, 1), _fmt(full_mn, 1), '—'])

        line = f'{t:<5} & ({v}, {m_val}) & ' + ' & '.join(f'{c:>6s}' for c in row_cells) + r' \\'
        out.append(line)

    out.append(r'\midrule')

    # Mean Δ and win count per model
    mean_row = r'\multicolumn{2}{l}{\textbf{Mean $\Delta$}} '
    wins_row = r'\multicolumn{2}{l}{\textbf{\# DP $>$ full}} '
    for model_name, _, _, _ in MODELS:
        deltas = per_model_deltas[model_name]
        if deltas:
            mean = sum(deltas) / len(deltas)
            wins = sum(1 for d in deltas if d > 0.01)
            mean_row += f'&  &  & \\textbf{{{mean:+.2f}}} '
            wins_row += f'&  &  & \\textbf{{{wins}/{len(deltas)}}} '
        else:
            mean_row += r'&  &  & — '
            wins_row += r'&  &  & — '
    mean_row += r'\\'
    wins_row += r'\\'
    out.append(mean_row)
    out.append(wins_row)

    out.append(r'\bottomrule')
    out.append(r'\end{tabular}')
    out.append(r'\end{table*}')
    return '\n'.join(out)


if __name__ == '__main__':
    print('% Auto-generated by code/generate_downproj_table.py')
    print('% Requires: \\usepackage{booktabs, multirow}')
    print()
    print(build_main_table())
    print(build_sweep_table())
