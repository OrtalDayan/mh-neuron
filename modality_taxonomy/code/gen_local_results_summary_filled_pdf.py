"""Filled copy of the uniform-sweep template.

Same layout as gen_local_results_summary_template_pdf.py, but Local cells are
filled from the on-disk coverage CSV (extract_merge_coverage.py output) for
modes: baseline, uniform_a0.9, uniform_a0.8, uniform_a0.7. Paper cells use the
BRV-published values (baseline + uniform-0.9 only, for the 3 BRV models). Gap =
Local-Paper; Δ = (+Uniform α) - Baseline. Empty where no score on disk.

Writes a NEW file; does not overwrite the template or the original summary.
"""
import csv
import math
from reportlab.lib.pagesizes import TABLOID, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.lib.enums import TA_JUSTIFY

COV_CSV = "/tmp/cov_fresh.csv"
OUT = ("/home/projects/bagon/ortalda/myrepo-weight_merging/modality_taxonomy/"
       "notes/local_results_summary_uniform_sweep_filled.pdf")

# ── Colors / styles (identical to source generators) ──
C_YELLOW = colors.HexColor("#FFF59D"); C_ORANGE = colors.HexColor("#FFCC80")
C_PINK   = colors.HexColor("#F8BBD0"); C_RED    = colors.HexColor("#EF9A9A")
C_HEADER = colors.HexColor("#2a4d69"); C_SUBHDR = colors.HexColor("#5c7a93")
C_ALT    = colors.HexColor("#f2f5f8")

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                     fontSize=10, leading=13, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceAfter=10, spaceBefore=18)
note = ParagraphStyle("note", parent=styles["BodyText"], fontName="Helvetica-Oblique",
                     fontSize=9, leading=11, spaceAfter=4, textColor=colors.grey)
cell_left_small = ParagraphStyle("cell_left_small", parent=styles["BodyText"],
                                fontName="Helvetica", fontSize=7.5, leading=9, alignment=0)


def fmt(v):  return f"{v:.2f}" if v is not None else "—"
def fmt_d(d):
    if d is None: return "—"
    return f"{'+' if d >= 0 else ''}{d:.2f}"
def color_for_delta(delta, perception, threshold):
    if delta is None: return None
    favorable = delta > 0; big = abs(delta) >= threshold
    if big and favorable: return C_YELLOW
    if big and not favorable: return C_RED
    return C_ORANGE if perception else C_PINK


# ── Load coverage: best (latest-mtime) score per (model, benchmark, mode) ──
best = {}   # (cov_model, cov_bench, mode) -> (score, mtime)
with open(COV_CSV) as f:
    for r in csv.DictReader(f):
        if not r["score"]:
            continue
        try:
            sc = float(r["score"]); mt = int(r["mtime"])
        except (ValueError, KeyError):
            continue
        key = (r["model"], r["benchmark"], r["mode"])
        if key not in best or mt > best[key][1]:
            best[key] = (sc, mt)

BENCH_COV = {
    "MathVista-Overall": "MathVista", "MathVista-General (NEW)": "MathVista-General",
    "MathVista-Math (NEW)": "MathVista-Math",
    "MathVerse-T-D": "MathVerse-T-D", "MathVerse-T-L": "MathVerse-T-L",
    "MathVerse-V-I": "MathVerse-V-I", "MathVerse-V-D": "MathVerse-V-D",
    "MathVerse-V-O": "MathVerse-V-O", "MathVision": "MathVision",
    "MMStar-Overall": "MMStar", "MMStar-Math": "MMStar-Math",
    "DynaMath-Average": "DynaMath", "DynaMath-Worst (NEW)": "DynaMath-Worst",
}
MV_SPLITS = ["MathVerse-T-D", "MathVerse-T-L", "MathVerse-V-I", "MathVerse-V-D", "MathVerse-V-O"]

def localval(cm, bdisp, mode):
    """Local score for (cov_model, display benchmark, mode); MathVerse-Overall = mean of 5 splits."""
    if bdisp == "MathVerse-Overall":
        vals = [best.get((cm, s, mode)) for s in MV_SPLITS]
        vals = [v[0] for v in vals if v]
        return sum(vals) / len(vals) if len(vals) == 5 else None
    rec = best.get((cm, BENCH_COV[bdisp], mode))
    return rec[0] if rec else None


# ── BRV-published Paper values (baseline, uniform_0.9); only the 3 BRV models ──
PAPER = {
    "llava-next-llama3-8b": {
        "MathVista-Overall": (37.4, 38.0), "MathVista-General (NEW)": (51.7, 48.7),
        "MathVista-Math (NEW)": (25.4, 28.9), "MathVerse-Overall": (20.1, 23.7),
        "MathVerse-T-D": (25.9, 30.7), "MathVerse-T-L": (20.8, 24.8),
        "MathVerse-V-I": (21.1, 25.5), "MathVerse-V-D": (16.5, 19.8),
        "MathVerse-V-O": (16.0, 17.4), "MathVision": (13.8, 14.8),
        "MMStar-Overall": (43.8, 43.6), "MMStar-Math": (30.0, 33.6),
        "DynaMath-Average": (22.7, 24.5), "DynaMath-Worst (NEW)": (None, None),
    },
    "idefics2-8b": {
        "MathVista-Overall": (51.8, 53.0), "MathVista-General (NEW)": (57.0, 58.5),
        "MathVista-Math (NEW)": (47.4, 48.3), "MathVerse-Overall": (19.4, 20.4),
        "MathVerse-T-D": (24.4, 26.0), "MathVerse-T-L": (21.3, 22.5),
        "MathVerse-V-I": (20.7, 21.3), "MathVerse-V-D": (19.7, 19.8),
        "MathVerse-V-O": (11.0, 12.1), "MathVision": (17.1, 16.8),
        "MMStar-Overall": (49.5, 48.3), "MMStar-Math": (39.6, 40.8),
        "DynaMath-Average": (21.8, 23.2), "DynaMath-Worst (NEW)": (None, None),
    },
    "qwen2-vl-7b": {
        "MathVista-Overall": (61.2, 60.2), "MathVista-General (NEW)": (69.6, 68.0),
        "MathVista-Math (NEW)": (54.1, 53.5), "MathVerse-Overall": (31.8, 31.9),
        "MathVerse-T-D": (35.9, 37.1), "MathVerse-T-L": (31.4, 31.7),
        "MathVerse-V-I": (31.5, 31.5), "MathVerse-V-D": (33.1, 32.5),
        "MathVerse-V-O": (26.9, 26.7), "MathVision": (21.1, 21.7),
        "MMStar-Overall": (59.9, 59.5), "MMStar-Math": (59.2, 58.4),
        "DynaMath-Average": (34.4, 35.0), "DynaMath-Worst (NEW)": (None, None),
    },
}
def paper(cm, bdisp, idx):  # idx 0 = baseline, 1 = uniform_0.9
    return PAPER.get(cm, {}).get(bdisp, (None, None))[idx]


BENCH_ROWS = [
    ("MathVista-Overall", "both"), ("MathVista-General (NEW)", "perception"),
    ("MathVista-Math (NEW)", "reasoning"), ("MathVerse-Overall", "reasoning"),
    ("MathVerse-T-D", "reasoning"), ("MathVerse-T-L", "reasoning"),
    ("MathVerse-V-I", "reasoning"), ("MathVerse-V-D", "reasoning"),
    ("MathVerse-V-O", "reasoning"), ("MathVision", "reasoning"),
    ("MMStar-Overall", "both"), ("MMStar-Math", "reasoning"),
    ("DynaMath-Average", "reasoning"), ("DynaMath-Worst (NEW)", "reasoning"),
]
# ── Score direction per benchmark. Web-verified: all 14 are accuracy metrics
# (MathVista, MathVerse, MathVision, MMStar, DynaMath — incl. DynaMath worst-case
# accuracy), where a higher score is better, so every benchmark gets an up arrow. ──
BENCH_ARROW = {b: "↑" for b, _ in BENCH_ROWS}

def bench_label(bdisp):
    """Benchmark name with a directional arrow (↑ = higher score is better)."""
    return f"{bdisp} {BENCH_ARROW.get(bdisp, '')}"
BRV_N = {
    "MathVista-Overall": "1000", "MathVista-General (NEW)": "460",
    "MathVista-Math (NEW)": "540", "MathVerse-Overall": "788*",
    "MathVerse-T-D": "788", "MathVerse-T-L": "788", "MathVerse-V-I": "788",
    "MathVerse-V-D": "788", "MathVerse-V-O": "788", "MathVision": "304",
    "MMStar-Overall": "1500", "MMStar-Math": "250",
    "DynaMath-Average": "5010", "DynaMath-Worst (NEW)": "501",
}

# ── Per-benchmark noise floor (web-derived). Every benchmark is an accuracy
# metric, so its finite-sample / run-to-run noise sets how big a change must be
# to count as real. Two distinct numbers, both from the binomial 95% normal
# approximation (worst case p=0.5) on each benchmark's own item count n:
#   NOISE_SINGLE = 1.96*sqrt(0.25/n)  — 95% wobble on ONE measured score.
#   REAL_DELTA   = 1.96*sqrt(0.50/n)  — min gap between TWO configs to clear
#       noise; ~sqrt(2) larger because a difference stacks two noisy scores.
# Empirical run-to-run variance from the literature (~1-2pp for greedy decoding,
# larger with GPT-judge answer extraction) sits at or below these sampling
# floors. n comes from each dataset's own paper; references per benchmark below.
BENCH_REF = {
    "MathVista-Overall":      "MathVista, arXiv:2310.02255",
    "MathVista-General (NEW)": "MathVista, arXiv:2310.02255",
    "MathVista-Math (NEW)":   "MathVista, arXiv:2310.02255",
    "MathVerse-Overall":      "MathVerse, arXiv:2403.14624",
    "MathVerse-T-D":          "MathVerse, arXiv:2403.14624",
    "MathVerse-T-L":          "MathVerse, arXiv:2403.14624",
    "MathVerse-V-I":          "MathVerse, arXiv:2403.14624",
    "MathVerse-V-D":          "MathVerse, arXiv:2403.14624",
    "MathVerse-V-O":          "MathVerse, arXiv:2403.14624",
    "MathVision":             "MATH-Vision, arXiv:2402.14804",
    "MMStar-Overall":         "MMStar, arXiv:2403.20330",
    "MMStar-Math":            "MMStar, arXiv:2403.20330",
    "DynaMath-Average":       "DynaMath, arXiv:2411.00836 (rep. var. ~1-2pp)",
    "DynaMath-Worst (NEW)":   "DynaMath, arXiv:2411.00836 (rep. var. ~1-2pp)",
}

def _bn(bdisp):  # item count n parsed from the BRV n column ("788*" -> 788)
    return int("".join(ch for ch in BRV_N.get(bdisp, "") if ch.isdigit()) or "0")

def noise_single(bdisp):
    n = _bn(bdisp); return 1.96 * math.sqrt(0.25 / n) * 100 if n else None

def real_delta(bdisp):
    n = _bn(bdisp); return 1.96 * math.sqrt(0.50 / n) * 100 if n else None

def fmt_thr(bdisp):  # compact inline cell, e.g. "≥4.4"
    rd = real_delta(bdisp); return f"≥{rd:.1f}" if rd is not None else "—"

# (display label, cov_model)
MODELS = [
    ("LLaVA-Next-LLaMA3-8B (Dart-Prop)",                   "llava-next-llama3-8b"),
    ("Idefics2-8B (MAmmoTH-1)",                            "idefics2-8b"),
    ("Qwen2-VL-7B-Instruct (Qwen2-Math-7B)",              "qwen2-vl-7b"),
    ("LLaVA-1.5-7B (MetaMath-7B-V1.0)",                   "llava-1.5-7b"),
    ("LLaVA-Next-Mistral-7B (Dart-Math-Mistral-Uniform)", "llava-next-mistral-7b"),
    ("LLaVA-OneVision-7B (Qwen2-Math-7B)",                "llava-onevision-7b"),
    ("InternVL2.5-8B (InternLM2-Math-Plus-7B)",           "internvl2.5-8b"),
    ("Qwen2.5-VL-7B (Qwen2.5-Math-7B)",                   "qwen2.5-vl-7b"),
    ("Qwen2.5-VL-3B (Qwen2.5-3B-DAPO-math-reasoning)",    "qwen2.5-vl-3b"),
]

# ── Document ──
doc = SimpleDocTemplate(OUT, pagesize=landscape(TABLOID),
    leftMargin=0.35*inch, rightMargin=0.35*inch,
    topMargin=0.45*inch, bottomMargin=0.45*inch,
    title="Local Results Summary: Uniform-Sweep (filled)")
story = []
story.append(Paragraph("Local Results Summary &mdash; Uniform Sweep 0.9 / 0.8 / 0.7 (filled)", h1))
story.append(Paragraph(
    "Filled from on-disk step-25 evals (coverage extract). <b>Local</b> = our measured "
    "score for baseline / +Uniform 0.9 / 0.8 / 0.7; <b>Paper</b> = BRV-published (baseline "
    "&amp; uniform-0.9 only, for the 3 BRV models); <b>Gap</b> = Local &minus; Paper; "
    "each <b>&Delta;</b> = (+Uniform &alpha;) &minus; Baseline. &mdash; means no score on "
    "disk yet (eval still in flight). Table 2: previously-blocked models, now resolved.",
    body))
story.append(Paragraph("All values, row-by-row (consolidated reference) — filled", h2))
story.append(Paragraph(
    "&Delta; Local cells colored by the perception/reasoning rule, thresholded at each "
    "benchmark's Real &Delta; (pp) noise floor (yellow = favorable &ge; floor, red = "
    "unfavorable &ge; floor; orange/pink = below the floor, i.e. within noise). Red Gap "
    "cells = |Local &minus; Paper| &ge; that same floor. "
    "MathVerse-Overall is the mean of its five split rows (only when all five are present). "
    "<b>Real &Delta; (pp)</b> = smallest baseline&rarr;config change that beats the 95% "
    "sampling-noise floor for that benchmark's item count n (1.96&middot;&radic;(0.5/n)); "
    "any &Delta; below it is statistical noise. Per-benchmark noise and references: final table.",
    note))

# ── Color legend: what each cell background means (swatch + meaning) ──
_legend_rows = [
    ["", Paragraph("<b>Yellow</b> &mdash; favorable &Delta; that beats the benchmark's Real &Delta; (pp) "
                   "noise floor (a real improvement).", cell_left_small)],
    ["", Paragraph("<b>Red</b> &mdash; unfavorable &Delta; that beats the floor (a real regression); also a "
                   "Gap cell where |Local &minus; Paper| &ge; the floor.", cell_left_small)],
    ["", Paragraph("<b>Orange</b> &mdash; &Delta; below the floor on a perception benchmark (within sampling "
                   "noise).", cell_left_small)],
    ["", Paragraph("<b>Pink</b> &mdash; &Delta; below the floor on a reasoning / both benchmark (within "
                   "sampling noise).", cell_left_small)],
]
_legtbl = Table(_legend_rows, colWidths=[0.3*inch, 6.7*inch])
_legtbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (0, 0), C_YELLOW), ("BACKGROUND", (0, 1), (0, 1), C_RED),
    ("BACKGROUND", (0, 2), (0, 2), C_ORANGE), ("BACKGROUND", (0, 3), (0, 3), C_PINK),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
]))
story.append(_legtbl)
story.append(Spacer(1, 6))

# ── Per-benchmark Real Δ (pp) floor + source (2-up to save vertical space) ──
story.append(Paragraph("Real &Delta; (pp) noise floor and source per benchmark "
                       "(the threshold used for cell coloring):", note))
_rd_hdr = ParagraphStyle("rd_hdr", parent=cell_left_small, fontName="Helvetica-Bold")
_rd_cells = [(bench_label(b), fmt_thr(b), BENCH_REF.get(b, "—")) for b, _ in BENCH_ROWS]
_rd_rows = [[Paragraph("Benchmark", _rd_hdr), Paragraph("Real &Delta; (pp)", _rd_hdr),
             Paragraph("Reference", _rd_hdr)] * 2]
for k in range(0, len(_rd_cells), 2):
    left = _rd_cells[k]
    right = _rd_cells[k + 1] if k + 1 < len(_rd_cells) else ("", "", "")
    _rd_rows.append([
        Paragraph(left[0], cell_left_small), left[1], Paragraph(left[2], cell_left_small),
        Paragraph(right[0], cell_left_small), right[1], Paragraph(right[2], cell_left_small),
    ])
_rdw = [1.25*inch, 0.65*inch, 2.55*inch]
_rdtbl = Table(_rd_rows, colWidths=_rdw * 2)
_rdtbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), C_SUBHDR), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("ALIGN", (1, 0), (1, -1), "CENTER"), ("ALIGN", (4, 0), (4, -1), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("LINEAFTER", (2, 0), (2, -1), 1.0, colors.grey),
    ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
] + [("BACKGROUND", (0, _r), (-1, _r), C_ALT) for _r in range(2, len(_rd_rows), 2)]))
story.append(_rdtbl)
story.append(Spacer(1, 6))

# Uniform 0.8/0.7 have no BRV-published value, so their Paper (and the derived
# Gap) are always empty — show Local only. Same for the Δ 0.8/0.7 Paper subcols.
GROUPS = [("Baseline", 3), ("+Uniform 0.9", 3), ("+Uniform 0.8", 1),
          ("+Uniform 0.7", 1), ("Δ 0.9 vs base", 2), ("Δ 0.8", 1),
          ("Δ 0.7", 1)]
hdr1 = ["", "", "", "", "", ""]
for title, span in GROUPS:
    hdr1 += [title] + [""] * (span - 1)
hdr2 = ["Model", "Benchmark", "n (Local)", "n (BRV)", "Type", "Real Δ (pp)"]
for _t, span in GROUPS:
    hdr2 += {3: ["Local", "Paper", "Gap"], 2: ["Local", "Paper"], 1: ["Local"]}[span]
NCOL = len(hdr2)

data = [hdr1, hdr2]
style_cmds = [
    ("BACKGROUND", (6, 0), (-1, 0), C_HEADER), ("TEXTCOLOR", (6, 0), (-1, 0), colors.white),
    ("FONTNAME", (6, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (6, 0), (-1, 0), 8.5),
    ("ALIGN", (0, 0), (-1, 1), "CENTER"), ("VALIGN", (0, 0), (-1, 1), "MIDDLE"),
    ("BACKGROUND", (0, 0), (5, 0), colors.white),
    ("LINEABOVE", (0, 0), (5, 0), 0, colors.white), ("LINEBELOW", (0, 0), (5, 0), 0, colors.white),
    ("LINEBEFORE", (0, 0), (0, 0), 0, colors.white), ("LINEAFTER", (5, 0), (5, 0), 0, colors.white),
    ("BACKGROUND", (0, 1), (-1, 1), C_SUBHDR), ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
    ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"), ("FONTSIZE", (0, 1), (-1, 1), 8),
    ("FONTSIZE", (0, 2), (-1, -1), 7.5), ("ALIGN", (2, 2), (-1, -1), "CENTER"),
    ("VALIGN", (0, 2), (-1, -1), "MIDDLE"), ("GRID", (0, 1), (-1, -1), 0.4, colors.grey),
    ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]
col = 6
for _t, span in GROUPS:
    style_cmds.append(("SPAN", (col, 0), (col + span - 1, 0)))
    col += span
# Identity block is now 6 cols (… Type, Real Δ), so the value groups start at
# col 6: Baseline=6, Uni09=9, Uni08=12, Uni07=13, d09=14, d08=16, d07=17.
GAP_B_COL, GAP_U9_COL = 8, 11
D09_COL, D08_COL, D07_COL = 14, 16, 17

i = 2
for label, cm in MODELS:
    prev = None
    for bdisp, btype in BENCH_ROWS:
        perc = (btype == "perception")
        lb  = localval(cm, bdisp, "baseline")
        l9  = localval(cm, bdisp, "uniform_a0.9")
        l8  = localval(cm, bdisp, "uniform_a0.8")
        l7  = localval(cm, bdisp, "uniform_a0.7")
        pb  = paper(cm, bdisp, 0); p9 = paper(cm, bdisp, 1)
        gap_b = (lb - pb) if (lb is not None and pb is not None) else None
        gap_9 = (l9 - p9) if (l9 is not None and p9 is not None) else None
        d9 = (l9 - lb) if (lb is not None and l9 is not None) else None
        d8 = (l8 - lb) if (lb is not None and l8 is not None) else None
        d7 = (l7 - lb) if (lb is not None and l7 is not None) else None
        d9p = (p9 - pb) if (pb is not None and p9 is not None) else None
        model_cell = Paragraph(label, cell_left_small) if label != prev else ""
        prev = label
        n_val = BRV_N.get(bdisp, "—")
        data.append([
            model_cell, Paragraph(bench_label(bdisp), cell_left_small), n_val, n_val, btype,
            fmt_thr(bdisp),
            fmt(lb), fmt(pb), fmt_d(gap_b),
            fmt(l9), fmt(p9), fmt_d(gap_9),
            fmt(l8),
            fmt(l7),
            fmt_d(d9), fmt_d(d9p),
            fmt_d(d8),
            fmt_d(d7),
        ])
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), C_ALT))
        rd = real_delta(bdisp)
        for ccol, dval in ((D09_COL, d9), (D08_COL, d8), (D07_COL, d7)):
            cd = color_for_delta(dval, perc, rd)
            if cd is not None:
                style_cmds.append(("BACKGROUND", (ccol, i), (ccol, i), cd))
        if gap_b is not None and abs(gap_b) >= rd:
            style_cmds.append(("BACKGROUND", (GAP_B_COL, i), (GAP_B_COL, i), C_RED))
        if gap_9 is not None and abs(gap_9) >= rd:
            style_cmds.append(("BACKGROUND", (GAP_U9_COL, i), (GAP_U9_COL, i), C_RED))
        i += 1

_idw = [1.30*inch, 1.30*inch, 0.50*inch, 0.50*inch, 0.55*inch, 0.52*inch]
_valw = [0.66*inch] * (NCOL - 6)
tbl = Table(data, colWidths=_idw + _valw, repeatRows=2)
tbl.setStyle(TableStyle(style_cmds))
story.append(tbl)
story.append(Spacer(1, 8))

# ── Tables 3a-3c: masked alpha sweep (t=0.3..0.8) per hook/method variant ──
ALPHAS_MASKED = ["0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9"]

def alpha_sweep_table(title, sub, mode_fn):
    story.append(Paragraph(title, h2))
    story.append(Paragraph(sub, note))
    hdr1 = ["", "", "", "", "Base"]
    for a in ALPHAS_MASKED: hdr1 += ["α=" + a, ""]
    hdr2 = ["Model", "Benchmark", "Type", "Real Δ (pp)", "Local"]
    for _a in ALPHAS_MASKED: hdr2 += ["Local", "Δ"]
    data = [hdr1, hdr2]
    style = [
        ("BACKGROUND", (4,0), (-1,0), C_HEADER), ("TEXTCOLOR", (4,0), (-1,0), colors.white),
        ("FONTNAME", (4,0), (-1,0), "Helvetica-Bold"), ("FONTSIZE", (4,0), (-1,0), 8.5),
        ("ALIGN", (0,0), (-1,1), "CENTER"), ("VALIGN", (0,0), (-1,1), "MIDDLE"),
        ("BACKGROUND", (0,0), (3,0), colors.white),
        ("LINEABOVE", (0,0), (3,0), 0, colors.white), ("LINEBELOW", (0,0), (3,0), 0, colors.white),
        ("LINEBEFORE", (0,0), (0,0), 0, colors.white), ("LINEAFTER", (3,0), (3,0), 0, colors.white),
        ("BACKGROUND", (0,1), (-1,1), C_SUBHDR), ("TEXTCOLOR", (0,1), (-1,1), colors.white),
        ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"), ("FONTSIZE", (0,1), (-1,1), 8),
        ("FONTSIZE", (0,2), (-1,-1), 7.5), ("ALIGN", (2,2), (-1,-1), "CENTER"),
        ("VALIGN", (0,2), (-1,-1), "MIDDLE"), ("GRID", (0,1), (-1,-1), 0.4, colors.grey),
        ("LEFTPADDING", (0,0), (-1,-1), 2), ("RIGHTPADDING", (0,0), (-1,-1), 2),
        ("TOPPADDING", (0,0), (-1,-1), 3), ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]
    col = 5
    for _a in ALPHAS_MASKED:
        style.append(("SPAN", (col,0), (col+1,0))); col += 2
    i = 2
    for label, cm in MODELS:
        prev = None
        for bdisp, btype in BENCH_ROWS:
            perc = (btype == "perception")
            lb = localval(cm, bdisp, "baseline")
            row = [Paragraph(label, cell_left_small) if label != prev else "",
                   Paragraph(bench_label(bdisp), cell_left_small), btype,
                   fmt_thr(bdisp), fmt(lb)]
            prev = label
            rd = real_delta(bdisp)
            dcol = 6
            for a in ALPHAS_MASKED:
                v = localval(cm, bdisp, mode_fn(a))
                d = (v - lb) if (v is not None and lb is not None) else None
                row += [fmt(v), fmt_d(d)]
                cd = color_for_delta(d, perc, rd)
                if cd is not None:
                    style.append(("BACKGROUND", (dcol, i), (dcol, i), cd))
                dcol += 2
            data.append(row)
            if i % 2 == 0:
                style.append(("BACKGROUND", (0,i), (-1,i), C_ALT))
            i += 1
    widths = [1.30*inch, 1.25*inch, 0.60*inch, 0.50*inch, 0.55*inch] + [0.52*inch, 0.50*inch] * len(ALPHAS_MASKED)
    t = Table(data, colWidths=widths, repeatRows=2)
    t.setStyle(TableStyle(style))
    story.append(t); story.append(Spacer(1, 8))

_sweep_sub = ("Masked task-arithmetic on text-classified MLP neurons (v=m=1.0, mlp-only). "
              "Local = measured score; &Delta; = vs the model's Baseline, colored at each benchmark's "
              "Real &Delta; (pp) noise floor (yellow = favorable &ge; floor, red = unfavorable &ge; floor; "
              "orange/pink = below the floor). &mdash; = no score on disk. MathVerse-Overall = mean of 5 splits. "
              "Real &Delta; (pp) = min change beating the 95% noise floor (see final-table references); "
              "treat any &Delta; below it as noise.")
story.append(PageBreak())
alpha_sweep_table("Masked alpha sweep — PMBT gate_up (pmbt_t*_v1.0_m1.0_mlponly)", _sweep_sub,
                  lambda a: f"pmbt_t{a}_v1.0_m1.0_mlponly")
story.append(PageBreak())
alpha_sweep_table("Masked alpha sweep — PMBT gate (pmbt_t*_v1.0_m1.0_mlponly_hgate)", _sweep_sub,
                  lambda a: f"pmbt_t{a}_v1.0_m1.0_mlponly_hgate")
story.append(PageBreak())
alpha_sweep_table("Masked alpha sweep — FT gate (ft_pmbt_t*_v1.0_m1.0_mlponly_hgate)", _sweep_sub,
                  lambda a: f"ft_pmbt_t{a}_v1.0_m1.0_mlponly_hgate")

# ── Noise reference table: per-benchmark sampling floor + sources ──
story.append(PageBreak())
story.append(Paragraph("Per-benchmark noise floor &amp; references", h2))
story.append(Paragraph(
    "How big a score change has to be before it is real, per benchmark. Both numbers come from "
    "the binomial 95% normal approximation (worst case p=0.5) on each benchmark's item count n. "
    "<b>Noise &plusmn;pp (single score)</b> = 95% wobble on <i>one</i> measured accuracy "
    "(1.96&middot;&radic;(p(1&minus;p)/n)). <b>Real &Delta; pp (config change)</b> = smallest gap "
    "between <i>two</i> configs that beats noise (1.96&middot;&radic;(2p(1&minus;p)/n)); it is "
    "&asymp;&radic;2 larger because a difference stacks the noise of two scores. These are "
    "conservative (independent-sample); same-item paired comparison (McNemar) is somewhat tighter. "
    "Published run-to-run variance (DynaMath: ~1&ndash;2pp over 5 repeats; multi-run VLM studies: "
    "~1.7&ndash;4.2pp std; GPT-judge extraction adds noise for MathVista/MathVerse) sits at or "
    "below these floors. Reference column = the dataset paper (which fixes n) plus any measured "
    "variance source. MathVerse-Overall averages 5 splits, so its effective floor is somewhat "
    "below the per-split value shown.",
    note))
_hw = ParagraphStyle("nz_hdr", parent=styles["BodyText"], fontName="Helvetica-Bold",
                     fontSize=8, leading=9.5, alignment=1, textColor=colors.white)
_nz_rows = [[Paragraph("Benchmark", _hw), Paragraph("n", _hw),
             Paragraph("Noise &plusmn;pp<br/>(single score, 95%)", _hw),
             Paragraph("Real &Delta; pp<br/>(config change, 95%)", _hw),
             Paragraph("Reference (defines n; + variance source)", _hw)]]
for _b, _t in BENCH_ROWS:
    _ns = noise_single(_b); _rd = real_delta(_b)
    _nz_rows.append([
        Paragraph(bench_label(_b), cell_left_small), BRV_N.get(_b, "—"),
        f"±{_ns:.1f}" if _ns is not None else "—",
        f"{_rd:.1f}" if _rd is not None else "—",
        Paragraph(BENCH_REF.get(_b, "—"), cell_left_small),
    ])
nztbl = Table(_nz_rows, colWidths=[1.6*inch, 0.6*inch, 1.5*inch, 1.5*inch, 3.6*inch])
nztbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), C_HEADER), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 8),
    ("ALIGN", (1, 0), (3, -1), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ("FONTSIZE", (0, 1), (-1, -1), 7.5), ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
    ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
] + [("BACKGROUND", (0, _r), (-1, _r), C_ALT) for _r in range(2, len(_nz_rows), 2)]))
story.append(nztbl)
story.append(Spacer(1, 8))

# ── Table 2: previously-blocked models, now resolved (same as template) ──
story.append(PageBreak())
story.append(Paragraph("Previously-blocked models — now resolved", h2))
story.append(Paragraph(
    "All four models that completed the 9-model enrichment-target set previously had no "
    "step-25 score CSVs. As of this update <b>every one now has on-disk eval data</b>, so "
    "their blockers are resolved. CSV counts are score/acc files under "
    "<code>results/25-merge/&lt;model&gt;/</code> at update time.",
    body))
blocked_rows = [
    ["Model", "Auto-routed math LLM", "Prior blocker", "Current status"],
    [Paragraph("LLaVA-OneVision-7B", cell_left_small), Paragraph("Qwen2-Math-7B (HF)", cell_left_small),
     Paragraph("VLMEvalKit_brv venv missing the <i>llava</i> Python package; eval jobs died with "
               "<code>No module named 'llava'</code> (also hit LLaVA-1.5-7B).", cell_left_small),
     Paragraph("<b>Resolved.</b> 24 merges, 402 score/acc CSVs on disk. "
               "(Handoff: <code>notes/vlmevalkit_handoff_llava_next_install.md</code>.)", cell_left_small)],
    [Paragraph("Qwen2.5-VL-7B", cell_left_small), Paragraph("Qwen2.5-Math-7B (HF)", cell_left_small),
     Paragraph("(1) <code>hf-xet</code> Rust panic mid-download of Qwen2.5-Math-7B; (2) eval "
               "<code>KeyError: 'Qwen2.5-VL-7B-Instruct'</code> (missing from VLMEvalKit "
               "<code>supported_VLM</code>).", cell_left_small),
     Paragraph("<b>Resolved.</b> 24 merges, 357 score/acc CSVs on disk. "
               "(Handoff: <code>notes/vlmevalkit_handoff_qwen25vl_7b_registry.md</code>.)", cell_left_small)],
    [Paragraph("InternVL2.5-8B", cell_left_small), Paragraph("internlm2-math-plus-7b (HF)", cell_left_small),
     Paragraph("Merge once killed by TERM_THREADLIMIT + <code>TypeError: dtype</code> (both fixed); "
               "eval failed with <code>'InternLM2ForCausalLM' object has no attribute 'generate'</code> "
               "(GenerationMixin not inherited).", cell_left_small),
     Paragraph("<b>Resolved.</b> 24 merges, 449 score/acc CSVs on disk. "
               "(Handoff: <code>notes/internvl_handoff_generation_mixin.md</code>.)", cell_left_small)],
    [Paragraph("Qwen2.5-VL-3B", cell_left_small), Paragraph("Qwen2.5-3B-DAPO-math-reasoning", cell_left_small),
     Paragraph("Hard-blocked (run_pipeline.sh exit 1): official Qwen2.5-Math ships only 1.5B/7B (28L), "
               "shape-incompatible with the 36-layer Qwen2.5-3B backbone, so no math donor was wired in.",
               cell_left_small),
     Paragraph("<b>Resolved.</b> Donor <code>jaygala24/Qwen2.5-3B-DAPO-math-reasoning</code> "
               "(36-layer, shape-compatible) added to step 25; 4 merges, 74 score/acc CSVs on disk.",
               cell_left_small)],
]
btbl = Table(blocked_rows, colWidths=[1.4*inch, 1.7*inch, 3.6*inch, 3.4*inch])
btbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), C_HEADER), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTSIZE", (0, 0), (-1, 0), 9),
    ("ALIGN", (0, 0), (-1, 0), "CENTER"), ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("BACKGROUND", (0, 1), (-1, 1), C_ALT), ("BACKGROUND", (0, 3), (-1, 3), C_ALT),
]))
story.append(btbl)
story.append(Spacer(1, 8))

# ── Methods note + references (HallusionBench evaluation) ──
story.append(PageBreak())
story.append(Paragraph("Methods note — HallusionBench evaluation", h2))
story.append(Paragraph(
    "HallusionBench is evaluated with <b>VLMEvalKit</b> (<code>image_yorn</code>), reporting the "
    "standard triplet of metrics &mdash; <b>aAcc</b> (per-question accuracy), <b>fAcc</b> "
    "(per-figure: all questions on a figure correct), and <b>qAcc</b> (per-question-pair: the "
    "original and the edited/control question both correct) &mdash; overall and per VS/VD category. "
    "Yes/No answers are obtained by <b>GPT-assisted extraction</b>: rule-based matching first, with a "
    "GPT judge (gpt-4o-mini) resolving the answers that rule-matching leaves ambiguous. This matches "
    "the protocol of the original HallusionBench paper (Guan et&nbsp;al., CVPR&nbsp;2024) and the "
    "OpenVLM leaderboard, which also use GPT-assisted yes/no extraction with the same aAcc/fAcc/qAcc "
    "metrics. (We switched from VLMEvalKit&rsquo;s rule-only <i>exact_matching</i> fallback to "
    "GPT-assisted extraction so our numbers are comparable to those works.)",
    body))
story.append(Paragraph("References", h2))
story.append(Paragraph(
    "&bull; Guan et&nbsp;al., <i>HallusionBench: An Advanced Diagnostic Suite for Entangled Language "
    "Hallucination &amp; Visual Illusion in Large Vision-Language Models</i>, CVPR 2024. "
    "<a href=\"https://arxiv.org/abs/2310.14566\" color=\"blue\">arXiv:2310.14566</a> "
    "(defines aAcc/fAcc/qAcc and GPT-4 yes/no extraction).",
    body))
story.append(Paragraph(
    "&bull; Duan et&nbsp;al., <i>VLMEvalKit: An Open-Source Toolkit for Evaluating Large "
    "Multi-Modality Models</i>, ACM MM 2024. "
    "<a href=\"https://arxiv.org/abs/2407.11691\" color=\"blue\">arXiv:2407.11691</a> "
    "(the evaluation framework used here; GPT-assisted vs exact-matching extraction). "
    "Powers the <a href=\"https://huggingface.co/spaces/opencompass/open_vlm_leaderboard\" color=\"blue\">"
    "OpenVLM leaderboard</a>.",
    body))

doc.build(story)
print(f"Wrote {OUT}")
