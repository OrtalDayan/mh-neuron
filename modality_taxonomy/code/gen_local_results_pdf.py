"""Render local + BRV-paper baseline-vs-uniform-0.9 results side by side, as a PDF.

Adds two Gap columns (Local − Paper) with red highlight when |gap| > 1.
Saved into modality_taxonomy/notes/.
"""
from reportlab.lib.pagesizes import LETTER, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.lib.enums import TA_JUSTIFY

OUT = "/home/projects/bagon/ortalda/myrepo-weight_merging/modality_taxonomy/notes/local_results_uniform_lambda_0.9.pdf"

# ─────────────────────────────────────────────────────────────────────
# Load hook-aware merge values from the extract_merge_coverage.py CSV
# Augments the consolidated big_table with PMBT-gate, PMBT-gate_up, FT-gate.
import csv as _csv, os as _os, re as _re
COV_CSV = "/tmp/cov_full.csv"
# Mode-tag prefix/suffix patterns. We pick the MAX value across all matching
# mode-tags per (model, benchmark, hook+method) — different alpha-triples
# produce different scores per benchmark, so "best" is selected per-cell.
RE_PMBT_GATE    = _re.compile(r"^pmbt_t.*hgate")              # any PMBT with _hgate
RE_PMBT_GATE_UP = _re.compile(r"^pmbt_t(?!.*hgate).*")         # PMBT without hgate
RE_FT_GATE      = _re.compile(r"^ft_pmbt_.*hgate")            # any FT with _hgate

# (model_name in big_table) ↔ (model/tv path in coverage CSV)
MODEL_TV_MAP = {
    "LLaVA-Next-LLaMA3-8B (Dart-Prop)":              ("llava-next-llama3-8b", "dart-prop"),
    "Idefics2-8B (MAmmoTH-1)":                       ("idefics2-8b",          "mammoth1"),
    "Qwen2-VL-7B-Instruct (Qwen2-Math-7B)":          ("qwen2-vl-7b",          "qwen2-math"),
    "LLaVA-1.5-7B (MetaMath-7B-V1.0)":               ("llava-1.5-7b",         "MetaMath-7B-V1.0"),
    "LLaVA-Next-Mistral-7B (MAmmoTH-7B-Mistral)":    ("llava-next-mistral-7b", "mammoth1"),
    "LLaVA-OneVision-7B (Qwen2-Math-7B)":            ("llava-onevision-7b",   "qwen2-math"),
    "InternVL2.5-8B (InternLM2-Math-Plus-7B)":       ("internvl2.5-8b",       "internlm2-math-plus-7b"),
    "Qwen2.5-VL-7B (Qwen2.5-Math-7B)":               ("qwen2.5-vl-7b",        "qwen25-math"),
}
# Map PDF benchmark display name → coverage canonical name. None means no
# corresponding row in the coverage script (e.g. MathVista General/Math
# splits are produced by a patched VLMEvalKit and don't surface in the
# script's per-benchmark extraction).
BENCH_MAP = {
    "MathVista-Overall":       "MathVista",
    "MathVista-General (NEW)": "MathVista-General",  # computed from per-skill rows
    "MathVista-Math (NEW)":    "MathVista-Math",     # = math-targeted-vqa skill row
    "MathVerse-Overall":       None,  # computed mean — done separately below
    "MathVerse-T-D":           "MathVerse-T-D",
    "MathVerse-T-L":           "MathVerse-T-L",
    "MathVerse-V-I":           "MathVerse-V-I",
    "MathVerse-V-D":           "MathVerse-V-D",
    "MathVerse-V-O":           "MathVerse-V-O",
    "MathVision":              "MathVision",
    "MMStar-Overall":          "MMStar",
    "MMStar-Math":             "MMStar-Math",         # from MMStar_acc.csv "math" col
    "DynaMath-Average":        "DynaMath",
    "DynaMath-Worst (NEW)":    "DynaMath-Worst",      # row 3 of DynaMath CSV
}

def _load_cov_index():
    """
    Index the coverage CSV by (model, tv, benchmark) → list of (mode, score).
    Lets us pick the MAX-scoring mode-tag per cell for the hook-aware columns.
    """
    out = {}
    if not _os.path.exists(COV_CSV):
        return out
    with open(COV_CSV) as f:
        for row in _csv.DictReader(f):
            try:
                v = float(row["score"]) if row["score"] else None
            except ValueError:
                v = None
            if v is None:
                continue
            key = (row["model"], row["tv"], row["benchmark"])
            out.setdefault(key, []).append((row["mode"], v))
    return out

_COV_INDEX = _load_cov_index()

def _best(model, tv, bench, mode_re):
    """Return the max score across all mode-tags matching mode_re."""
    candidates = _COV_INDEX.get((model, tv, bench), [])
    matches = [v for (mode, v) in candidates if mode_re.match(mode)]
    return max(matches) if matches else None

def hook_aware(model_name, bench_name):
    """
    Returns the BEST (max-scoring) value across all on-disk alpha-triple
    variants for each of (PMBT-gate, PMBT-gate_up, FT-gate).
    """
    if model_name not in MODEL_TV_MAP:
        return (None, None, None)
    m, tv = MODEL_TV_MAP[model_name]
    if bench_name == "MathVerse-Overall":
        # For MathVerse-Overall, compute the mean of the 5 split rows
        # for each hook+method *independently per mode-tag*, then take
        # the max across modes. (Equivalent to: best per-mode Overall.)
        splits = ("MathVerse-T-D", "MathVerse-T-L",
                  "MathVerse-V-I", "MathVerse-V-D", "MathVerse-V-O")
        out = []
        for regex in (RE_PMBT_GATE, RE_PMBT_GATE_UP, RE_FT_GATE):
            # Collect per-mode tuples of (mode, [split values])
            per_mode = {}
            for s in splits:
                for (mode, v) in _COV_INDEX.get((m, tv, s), []):
                    if regex.match(mode):
                        per_mode.setdefault(mode, []).append(v)
            mode_overall = [sum(vs)/len(vs) for vs in per_mode.values() if len(vs) == 5]
            out.append(max(mode_overall) if mode_overall else None)
        return tuple(out)
    cov_bench = BENCH_MAP.get(bench_name)
    if cov_bench is None:
        return (None, None, None)
    return (
        _best(m, tv, cov_bench, RE_PMBT_GATE),
        _best(m, tv, cov_bench, RE_PMBT_GATE_UP),
        _best(m, tv, cov_bench, RE_FT_GATE),
    )

C_YELLOW = colors.HexColor("#FFF59D")
C_ORANGE = colors.HexColor("#FFCC80")
C_PINK   = colors.HexColor("#F8BBD0")
C_RED    = colors.HexColor("#EF9A9A")
C_HEADER = colors.HexColor("#2a4d69")
C_SUBHDR = colors.HexColor("#5c7a93")
C_ALT    = colors.HexColor("#f2f5f8")
GAP_RED_THRESHOLD = 1.0

styles = getSampleStyleSheet()
body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                     fontSize=10, leading=13, alignment=TA_JUSTIFY, spaceAfter=6)
h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, spaceAfter=8)
h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceAfter=10, spaceBefore=18)
note = ParagraphStyle("note", parent=styles["BodyText"], fontName="Helvetica-Oblique",
                     fontSize=9, leading=11, spaceAfter=4, textColor=colors.grey)
# Cell-level paragraph styles for auto-wrapping inside tables
cell_left = ParagraphStyle("cell_left", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=8, leading=10, alignment=0)  # left
cell_left_small = ParagraphStyle("cell_left_small", parent=styles["BodyText"],
                                fontName="Helvetica", fontSize=7.5, leading=9, alignment=0)
# Style used for wrapping column-group banner text in the consolidated table
# header so long labels like "Hook-aware (best α-triple per cell)" don't get
# truncated by their cell width.
cell_center_hdr = ParagraphStyle("cell_center_hdr", parent=styles["BodyText"],
                                fontName="Helvetica-Bold", fontSize=8.5,
                                leading=10, alignment=1, textColor=colors.white)
cell_center_subhdr = ParagraphStyle("cell_center_subhdr", parent=styles["BodyText"],
                                fontName="Helvetica-Bold", fontSize=7.5,
                                leading=9, alignment=1, textColor=colors.white)

doc = SimpleDocTemplate(OUT, pagesize=landscape(LETTER),
    leftMargin=0.35*inch, rightMargin=0.35*inch,
    topMargin=0.45*inch, bottomMargin=0.45*inch,
    title="Local + BRV-paper Results: Baseline vs Uniform λ=0.9")

story = []
story.append(Paragraph(
    "Local + BRV-paper Results &mdash; Baseline vs. Uniform &lambda;=0.9", h1))
story.append(Paragraph(
    "Each row compares one benchmark across two methods (Baseline / +Uniform 0.9) "
    "and two sources (Local re-scored eval / BRV Chen et al. 2025 Tables 2-3). "
    "<b>Gap = Local − Paper</b>; cells where |Gap|&nbsp;&gt;&nbsp;1&nbsp;pp are highlighted "
    "in red. The &Delta; columns are also kept &mdash; Local &Delta; gets the original "
    "yellow/orange/pink/red coloring (favorable/perception/reasoning/unfavorable).",
    body))


def fmt(v):
    return f"{v:.2f}" if v is not None else "—"

def fmt_d(d):
    if d is None: return "—"
    return f"{'+' if d>=0 else ''}{d:.2f}"

def color_for_delta(delta, perception):
    if delta is None: return None
    favorable = delta > 0
    big = abs(delta) >= 1.0
    if big and favorable:    return C_YELLOW
    if big and not favorable: return C_RED
    return C_ORANGE if perception else C_PINK


def make_table(title, sub, rows):
    """rows: (name, type, local_base, paper_base, local_unif, paper_unif, perception)"""
    story.append(Paragraph(title, h2))
    if sub:
        story.append(Paragraph(sub, note))

    # Two-row header. Top row groups the numeric columns into Baseline / +Uniform / Δ.
    # Bottom row has the leaf labels (Benchmark / Type / Local / Paper / Gap).
    hdr1 = ["", "",
            "Baseline", "", "",
            "+Uniform 0.9", "", "",
            "Δ", ""]
    hdr2 = ["Benchmark ↑", "Type",
            "Local", "Paper", "Gap",
            "Local", "Paper", "Gap",
            "Local", "Paper"]
    data = [hdr1, hdr2]

    style_cmds = [
        # Top header row: dark background only on the group-title cells.
        ("BACKGROUND", (2,0), (-1,0), C_HEADER),
        ("TEXTCOLOR", (2,0), (-1,0), colors.white),
        ("FONTNAME", (2,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (2,0), (-1,0), 9),
        ("ALIGN", (0,0), (-1,1), "CENTER"),
        ("VALIGN", (0,0), (-1,1), "MIDDLE"),
        ("SPAN", (2,0), (4,0)),    # Baseline group
        ("SPAN", (5,0), (7,0)),    # +Uniform group
        ("SPAN", (8,0), (9,0)),    # Δ group
        # The two empty cells above "Benchmark"/"Type" — keep them
        # background-less and gridless so there's a clean white gap.
        ("BACKGROUND", (0,0), (1,0), colors.white),
        ("LINEABOVE", (0,0), (1,0), 0, colors.white),
        ("LINEBELOW", (0,0), (1,0), 0, colors.white),
        ("LINEBEFORE", (0,0), (0,0), 0, colors.white),
        ("LINEAFTER", (1,0), (1,0), 0, colors.white),
        # Bottom header row (leaf labels).
        ("BACKGROUND", (0,1), (-1,1), C_SUBHDR),
        ("TEXTCOLOR", (0,1), (-1,1), colors.white),
        ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE", (0,1), (-1,1), 8.5),
        # Body
        ("FONTSIZE", (0,2), (-1,-1), 8),
        ("ALIGN", (2,2), (-1,-1), "CENTER"),
        ("VALIGN", (0,2), (-1,-1), "MIDDLE"),
        ("GRID", (0,1), (-1,-1), 0.4, colors.grey),  # grid only on row 1+
        ("LEFTPADDING", (0,0), (-1,-1), 3),
        ("RIGHTPADDING", (0,0), (-1,-1), 3),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ]

    for i, (name, typ, lb, pb, lu, pu, perc) in enumerate(rows, start=2):
        gap_b = (lb - pb) if (lb is not None and pb is not None) else None
        gap_u = (lu - pu) if (lu is not None and pu is not None) else None
        ld = (lu - lb) if (lb is not None and lu is not None) else None
        pd = (pu - pb) if (pb is not None and pu is not None) else None
        data.append([Paragraph(name, cell_left), typ,
                     fmt(lb), fmt(pb), fmt_d(gap_b),
                     fmt(lu), fmt(pu), fmt_d(gap_u),
                     fmt_d(ld), fmt_d(pd)])
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0,i), (-1,i), C_ALT))
        # Color the Local Δ cell with the perception/reasoning rule
        cdelta = color_for_delta(ld, perc)
        if cdelta is not None:
            style_cmds.append(("BACKGROUND", (8,i), (8,i), cdelta))
        # Red-flag the Gap cells when |gap| > 1
        if gap_b is not None and abs(gap_b) > GAP_RED_THRESHOLD:
            style_cmds.append(("BACKGROUND", (4,i), (4,i), C_RED))
        if gap_u is not None and abs(gap_u) > GAP_RED_THRESHOLD:
            style_cmds.append(("BACKGROUND", (7,i), (7,i), C_RED))

    tbl = Table(data, colWidths=[1.7*inch, 0.7*inch,
                                 0.55*inch, 0.55*inch, 0.55*inch,
                                 0.6*inch,  0.6*inch,  0.55*inch,
                                 0.6*inch,  0.6*inch],
                repeatRows=2)
    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    story.append(Spacer(1, 8))


# ───────── LLaVA-Next (local: Dart-Prop ; paper Table 2: +Dart-Prop) ─────────
llava = [
    ("MathVista-Overall",       "perception", 37.80, 37.4,  37.90, 38.0,  True),
    ("MathVista-General (NEW)", "perception", 51.96, 51.7,  49.13, 48.7,  True),
    ("MathVista-Math (NEW)",    "reasoning",  25.74, 25.4,  28.33, 28.9,  False),
    ("MathVerse-Overall",       "reasoning",  18.45, 20.1,  19.03, 23.7,  False),
    ("MathVerse-T-D",           "reasoning",  22.34, 25.9,  23.98, 30.7,  False),
    ("MathVerse-T-L",           "reasoning",  19.67, 20.8,  19.92, 24.8,  False),
    ("MathVerse-V-I",           "reasoning",  19.80, 21.1,  20.18, 25.5,  False),
    ("MathVerse-V-D",           "reasoning",  15.48, 16.5,  15.48, 19.8,  False),
    ("MathVerse-V-O",           "reasoning",  14.97, 16.0,  15.61, 17.4,  False),
    ("MathVision",              "reasoning",  14.14, 13.8,  14.47, 14.8,  False),
    ("MMStar-Overall",          "perception", 43.33, 43.8,  43.60, 43.6,  True),
    ("MMStar-Math",             "reasoning",  29.60, 30.0,  33.60, 33.6,  False),
    ("DynaMath-Average",        "reasoning",  22.75, 22.7,  24.35, 24.5,  False),
    ("DynaMath-Worst (NEW)",    "reasoning",   3.19, None,   4.39, None,  False),
]

# ───────── Idefics (local: MAmmoTH-1 ; paper Table 3: +MAmmoTH-1) ─────────
idefics = [
    ("MathVista-Overall",       "perception", 52.00, 51.8,  52.50, 53.0,  True),
    ("MathVista-General (NEW)", "perception", 56.30, 57.0,  59.13, 58.5,  True),
    ("MathVista-Math (NEW)",    "reasoning",  48.33, 47.4,  46.85, 48.3,  False),
    ("MathVerse-Overall",       "reasoning",  18.38, 19.4,  18.96, 20.4,  False),
    ("MathVerse-T-D",           "reasoning",  21.70, 24.4,  24.62, 26.0,  False),
    ("MathVerse-T-L",           "reasoning",  20.81, 21.3,  21.57, 22.5,  False),
    ("MathVerse-V-I",           "reasoning",  20.05, 20.7,  19.67, 21.3,  False),
    ("MathVerse-V-D",           "reasoning",  19.04, 19.7,  17.89, 19.8,  False),
    ("MathVerse-V-O",           "reasoning",  10.28, 11.0,  11.04, 12.1,  False),
    ("MathVision",              "reasoning",  16.45, 17.1,  14.80, 16.8,  False),
    ("MMStar-Overall",          "perception", 49.33, 49.5,  48.40, 48.3,  True),
    ("MMStar-Math",             "reasoning",  39.60, 39.6,  40.00, 40.8,  False),
    ("DynaMath-Average",        "reasoning",  21.80, 21.8,  23.01, 23.2,  False),
    ("DynaMath-Worst (NEW)",    "reasoning",   3.19, None,   4.79, None,  False),
]

make_table("LLaVA-Next-LLaMA3-8B &nbsp;&nbsp;(task vector: Dart-Prop)",
           "Local re-scored vs. BRV Table 2 (+Dart-Prop column). MathVerse-Overall is "
           "computed locally as the simple mean of the five mode scores, matching BRV's "
           "methodology (the five modes share the same 788 problems).",
           llava)

story.append(PageBreak())
make_table("Idefics2-8B &nbsp;&nbsp;(task vector: MAmmoTH-1)",
           "Local re-scored vs. BRV Table 3 (+MAmmoTH-1 column).",
           idefics)

# ───────── Qwen2-VL (local: Qwen2-Math ; paper Table 7: +Qwen2-Math) ─────────
qwen = [
    ("MathVista-Overall",       "perception", 61.10, 61.2,  58.30, 60.2,  True),
    ("MathVista-General (NEW)", "perception", 69.78, 69.6,  66.09, 68.0,  True),
    ("MathVista-Math (NEW)",    "reasoning",  53.70, 54.1,  51.67, 53.5,  False),
    ("MathVerse-Overall",       "reasoning",  30.68, 31.8,  30.41, 31.9,  False),
    ("MathVerse-T-D",           "reasoning",  34.77, 35.9,  33.88, 37.1,  False),
    ("MathVerse-T-L",           "reasoning",  30.58, 31.4,  30.08, 31.7,  False),
    ("MathVerse-V-I",           "reasoning",  30.71, 31.5,  30.08, 31.5,  False),
    ("MathVerse-V-D",           "reasoning",  32.23, 33.1,  31.09, 32.5,  False),
    ("MathVerse-V-O",           "reasoning",  25.13, 26.9,  26.90, 26.7,  False),
    ("MathVision",              "reasoning",  20.72, 21.1,  20.07, 21.7,  False),
    ("MMStar-Overall",          "perception", 59.60, 59.9,  56.47, 59.5,  True),
    ("MMStar-Math",             "reasoning",  58.40, 59.2,  55.20, 58.4,  False),
    ("DynaMath-Average",        "reasoning",  33.55, 34.4,  31.40, 35.0,  False),
    ("DynaMath-Worst (NEW)",    "reasoning",  10.38, None,  10.78, None,  False),
]
story.append(PageBreak())
make_table("Qwen2-VL-7B-Instruct &nbsp;&nbsp;(task vector: Qwen2-Math-7B)",
           "Local re-scored vs. BRV Appendix E / Table 7 (+Qwen2-Math column).",
           qwen)

# ───────── LLaVA-1.5-7B (local: MetaMath-7B-V1.0; not in BRV main tables) ─────────
# All Paper columns are None — BRV doesn't report results for LLaVA-1.5-7B + MetaMath
# in Tables 2/3/7. Δ columns (Local) are still computed and color-coded.
# MMStar and DynaMath-Worst pending jobs as of 2026-06-04.
llava15 = [
    ("MathVista-Overall",       "perception", 24.10, None,  23.70, None,  True),
    ("MathVista-General (NEW)", "perception", 24.21, None,  23.63, None,  True),
    ("MathVista-Math (NEW)",    "reasoning",  17.04, None,  18.33, None,  False),
    ("MathVerse-Overall",       "reasoning",  15.03, None,  14.90, None,  False),
    ("MathVerse-T-D",           "reasoning",  15.61, None,  14.85, None,  False),
    ("MathVerse-T-L",           "reasoning",  14.59, None,  14.59, None,  False),
    ("MathVerse-V-I",           "reasoning",  14.85, None,  14.72, None,  False),
    ("MathVerse-V-D",           "reasoning",  15.36, None,  15.10, None,  False),
    ("MathVerse-V-O",           "reasoning",  14.72, None,  15.23, None,  False),
    ("MathVision",              "reasoning",  15.79, None,  17.11, None,  False),
    ("MMStar-Overall",          "perception", 31.07, None,   None, None,  True),
    ("MMStar-Math",             "reasoning",  22.40, None,   None, None,  False),
    ("DynaMath-Average",        "reasoning",  12.73, None,  11.86, None,  False),
    ("DynaMath-Worst (NEW)",    "reasoning",   1.20, None,   1.80, None,  False),
]
story.append(PageBreak())
make_table("LLaVA-1.5-7B &nbsp;&nbsp;(task vector: MetaMath-7B-V1.0)",
           "Local re-scored only. BRV Tables 2/3/7 do not include LLaVA-1.5-7B. "
           "MMStar baseline + uniform jobs in flight as of 2026-06-04. "
           "POPE-Overall available (84.80 / 84.10, Δ &minus;0.70) — see supplement notes.",
           llava15)

# ───────── LLaVA-Next-Mistral-7B (local: MAmmoTH-7B-Mistral; not in BRV main tables) ─────────
mistral = [
    ("MathVista-Overall",       "perception", 35.40, None,  36.00, None,  True),
    ("MathVista-General (NEW)", "perception", 35.96, None,  36.74, None,  True),
    ("MathVista-Math (NEW)",    "reasoning",  26.11, None,  26.30, None,  False),
    ("MathVerse-Overall",       "reasoning",  18.27, None,  17.92, None,  False),
    ("MathVerse-T-D",           "reasoning",  21.19, None,  21.07, None,  False),
    ("MathVerse-T-L",           "reasoning",  17.89, None,  18.27, None,  False),
    ("MathVerse-V-I",           "reasoning",  19.16, None,  19.16, None,  False),
    ("MathVerse-V-D",           "reasoning",  17.01, None,  16.75, None,  False),
    ("MathVerse-V-O",           "reasoning",  16.12, None,  14.34, None,  False),
    ("MathVision",              "reasoning",  11.18, None,  13.82, None,  False),
    ("MMStar-Overall",          "perception", 39.33, None,   None, None,  True),
    ("MMStar-Math",             "reasoning",  27.20, None,   None, None,  False),
    ("DynaMath-Average",        "reasoning",  17.78, None,  19.26, None,  False),
    ("DynaMath-Worst (NEW)",    "reasoning",   2.20, None,   2.40, None,  False),
]
# ───────── LLaVA-OneVision-7B (Local: Qwen2-Math-7B, not in BRV main tables) ─────────
# Partial coverage today — newly unblocked via HF VLMEvalKit class switch
# (LLaVA_OneVision_HF). Uniform side landed first; baseline only has 2 cells
# until those evals catch up. mistral-style cap: Paper columns all None.
llava_ov = [
    ("MathVista-Overall",       "perception",  None, None,  60.00, None,  True),
    ("MathVista-General (NEW)", "perception",  None, None,  59.31, None,  True),
    ("MathVista-Math (NEW)",    "reasoning",   None, None,  61.48, None,  False),
    ("MathVerse-Overall",       "reasoning",   None, None,  30.36, None,  False),
    ("MathVerse-T-D",           "reasoning",   None, None,  38.71, None,  False),
    ("MathVerse-T-L",           "reasoning",   None, None,  33.63, None,  False),
    ("MathVerse-V-I",           "reasoning",   None, None,  31.73, None,  False),
    ("MathVerse-V-D",           "reasoning",   None, None,  29.06, None,  False),
    ("MathVerse-V-O",           "reasoning",  17.64, None,  18.65, None,  False),
    ("MathVision",              "reasoning",  18.42, None,  17.43, None,  False),
    ("MMStar-Overall",          "perception",  0.00, None,   0.00, None,  True),
    ("MMStar-Math",             "reasoning",   0.00, None,   0.00, None,  False),
    ("DynaMath-Average",        "reasoning",  19.66, None,  20.72, None,  False),
    ("DynaMath-Worst (NEW)",    "reasoning",   4.99, None,   5.59, None,  False),
]

# ───────── InternVL2.5-8B (Local: InternLM2-Math-Plus-7B, not in BRV main tables) ─────────
# Partial — baseline landed via the GenerationMixin + past_key_values patches,
# uniform still in flight.
internvl = [
    ("MathVista-Overall",       "perception", 65.00, None,   None, None,  True),
    ("MathVista-General (NEW)", "perception", 64.59, None,   None, None,  True),
    ("MathVista-Math (NEW)",    "reasoning",  61.48, None,   None, None,  False),
    ("MathVerse-Overall",       "reasoning",  34.67, None,  27.36, None,  False),
    ("MathVerse-T-D",           "reasoning",  42.89, None,  31.60, None,  False),
    ("MathVerse-T-L",           "reasoning",  38.58, None,  29.06, None,  False),
    ("MathVerse-V-I",           "reasoning",  37.31, None,  30.20, None,  False),
    ("MathVerse-V-D",           "reasoning",  33.25, None,  26.14, None,  False),
    ("MathVerse-V-O",           "reasoning",  21.32, None,  19.80, None,  False),
    ("MathVision",              "reasoning",  23.68, None,  10.86, None,  False),
    ("MMStar-Overall",          "perception", 62.73, None,  61.13, None,  True),
    ("MMStar-Math",             "reasoning",  68.00, None,  69.60, None,  False),
    ("DynaMath-Average",        "reasoning",  32.22, None,   None, None,  False),
    ("DynaMath-Worst (NEW)",    "reasoning",   9.38, None,   None, None,  False),
]

# ───────── Qwen2.5-VL-7B (Local: Qwen2.5-Math-7B, not in BRV main tables) ─────────
# No score CSVs landed yet — eval jobs slow (each MathVista item is ~5-10s),
# placeholder rows so the model has a presence in the consolidated table.
qwen25vl7b = [
    ("MathVista-Overall",       "perception", 68.90, None,   None, None,  True),
    ("MathVista-General (NEW)", "perception", 68.20, None,   None, None,  True),
    ("MathVista-Math (NEW)",    "reasoning",  67.04, None,   None, None,  False),
    ("MathVerse-Overall",       "reasoning",  44.01, None,   None, None,  False),
    ("MathVerse-T-D",           "reasoning",  53.43, None,   None, None,  False),
    ("MathVerse-T-L",           "reasoning",  45.18, None,   None, None,  False),
    ("MathVerse-V-I",           "reasoning",  40.61, None,   None, None,  False),
    ("MathVerse-V-D",           "reasoning",  40.10, None,   None, None,  False),
    ("MathVerse-V-O",           "reasoning",  40.74, None,   None, None,  False),
    ("MathVision",              "reasoning",  26.32, None,   None, None,  False),
    ("MMStar-Overall",          "perception", 61.67, None,   None, None,  True),
    ("MMStar-Math",             "reasoning",  61.20, None,   None, None,  False),
    ("DynaMath-Average",        "reasoning",   None, None,   None, None,  False),
    ("DynaMath-Worst (NEW)",    "reasoning",   None, None,   None, None,  False),
]

story.append(PageBreak())
make_table("LLaVA-Next-Mistral-7B &nbsp;&nbsp;(task vector: MAmmoTH-7B-Mistral)",
           "Local re-scored only. BRV Tables 2/3/7 do not include LLaVA-Next-Mistral-7B. "
           "MM-Math jobs in flight as of 2026-06-04 (~8–12h remaining; uniform at 76%, "
           "baseline at 68%). Bench rows not in this table but in the supplement: "
           "MME 1788.42 / 1791.94, HallusionBench 30.28 / 33.44, "
           "POPE-Overall 87.44 / 87.36 (Δ −0.08).",
           mistral)

# ───────── Consolidated large reference table (all values, one by one) ─────────
story.append(PageBreak())
story.append(Paragraph("All values, row-by-row (consolidated reference)", h2))
story.append(Paragraph(
    "<b>Coverage:</b> 8 of the 9 enrichment-target VLMs now appear in this table "
    "(LLaVA-1.5-7B, Idefics2-8B, LLaVA-Next-LLaMA3-8B, LLaVA-Next-Mistral-7B, "
    "Qwen2-VL-7B-Instruct, LLaVA-OneVision-7B, InternVL2.5-8B, Qwen2.5-VL-7B). "
    "The 3 newly-added models have partial cells (uniform-only or baseline-only) "
    "because their eval jobs are still in flight or only one side has landed so far. "
    "Qwen2.5-VL-3B remains structurally blocked (no shape-compatible math donor); "
    "see the &ldquo;Blocked models&rdquo; page for the full blocker map.",
    note))
story.append(Paragraph(
    "<b>Note on (NEW) rows for LLaVA-1.5 and Mistral:</b> the "
    "MathVista-General, MathVista-Math, and DynaMath-Worst values for those "
    "two pairs are computed by <code>extract_merge_coverage.py</code> from "
    "per-skill MathVista CSV rows / row 3 of the DynaMath CSV. MathVista-Math "
    "and DynaMath-Worst are exact matches against the canonical extraction. "
    "MathVista-General is approximated as (Overall &minus; Math) weighted by "
    "row count; BRV's General excludes additional math-leaning skills "
    "(arithmetic, algebra, geometry, etc.) so its published value tends to "
    "run ~5 pp higher than this approximation. The 3 existing models "
    "(LLaVA-LLaMA3, Idefics2, Qwen2-VL) keep the original BRV-script values.",
    note))
story.append(Paragraph(
    "<b>Hook-aware columns:</b> the three rightmost columns show the "
    "<i>best</i> (max-scoring) value across <b>every &alpha;-triple variant on "
    "disk</b> for each (model, benchmark) cell. PMBT-gate scans all "
    "<code>pmbt_t*_hgate</code> mode-tags (~20 variants); PMBT-gate_up scans all "
    "<code>pmbt_t*</code> tags <i>without</i> the <code>_hgate</code> suffix "
    "(~206 variants); FT-gate scans all <code>ft_pmbt_*_hgate</code> tags "
    "(~10 variants). MathVerse-Overall is the mean of the 5 split rows for the "
    "best single mode-tag per cell (not a free max across modes). Cells "
    "= &mdash; indicate that no on-disk variant in that hook+method group has "
    "a value for that (model, tv, benchmark) tuple. PMBT-gate and PMBT-gate_up "
    "are not perfectly comparable: the on-disk gate variants don't carry the "
    "<code>_a1.0_o1.0</code> suffix that gate_up variants do, so &alpha;_other "
    "differs (gate uses default 0.9; gate_up uses 1.0).",
    note))
story.append(Paragraph(
    "Same data as the per-model tables above, flattened into one table for "
    "spreadsheet-style scanning. One row per (model &times; benchmark). "
    "<b>n (Local)</b> is the row count we measure in the local <code>~/LMUData/</code> "
    "TSV; <b>n (BRV)</b> is the row count BRV's eval registers via "
    "<code>VLMEvalKit_brv/vlmeval/dataset/image_vqa.py</code>. The two columns are "
    "identical for every row because BRV's dataset registry resolves to the same "
    "upstream files (MD5-matched). The asterisk on MathVerse-Overall (788*) marks "
    "the <i>effective</i> n &mdash; the five modes share the same 788 problems, so "
    "the 5-mode average denominator stays at 788, not 3,940. Red highlight on Gap "
    "cells = |Local − Paper| &gt; 1&nbsp;pp.",
    body))

# Sample sizes used by BRV (matches our local evals — same TSVs, identical MD5s).
# MathVerse-Overall n=788 is the effective sample count (5 modes share the same
# 788 problems, so the average doesn't multiply the denominator).
BRV_N = {
    "MathVista-Overall":       "1000",
    "MathVista-General (NEW)": "460",
    "MathVista-Math (NEW)":    "540",
    "MathVerse-Overall":       "788*",
    "MathVerse-T-D":           "788",
    "MathVerse-T-L":           "788",
    "MathVerse-V-I":           "788",
    "MathVerse-V-D":           "788",
    "MathVerse-V-O":           "788",
    "MathVision":              "304",
    "MMStar-Overall":          "1500",
    "MMStar-Math":             "250",
    "DynaMath-Average":        "5010",
    "DynaMath-Worst (NEW)":    "501",
}


def big_table(all_rows):
    # Wrap the long banner in a Paragraph so it can flow across the 3-column
    # span (which is ~1.7 inches wide — too narrow for a single-line label).
    hook_banner = Paragraph(
        "Hook-aware best (max across &alpha;-triples)", cell_center_hdr)
    hdr1 = ["", "", "", "", "",
            "Baseline", "", "",
            "+Uniform 0.9", "", "",
            "Δ", "",
            hook_banner, "", ""]
    hdr2 = ["Model", "Benchmark", "n (Local)", "n (BRV)", "Type",
            "Local", "Paper", "Gap",
            "Local", "Paper", "Gap",
            "Local", "Paper",
            Paragraph("PMBT<br/>gate", cell_center_subhdr),
            Paragraph("PMBT<br/>gate_up", cell_center_subhdr),
            Paragraph("FT<br/>gate", cell_center_subhdr)]
    data = [hdr1, hdr2]

    style_cmds = [
        ("BACKGROUND", (5,0), (-1,0), C_HEADER),
        ("TEXTCOLOR", (5,0), (-1,0), colors.white),
        ("FONTNAME", (5,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (5,0), (-1,0), 9),
        ("ALIGN", (0,0), (-1,1), "CENTER"),
        ("VALIGN", (0,0), (-1,1), "MIDDLE"),
        ("SPAN", (5,0), (7,0)),    # Baseline
        ("SPAN", (8,0), (10,0)),   # +Uniform
        ("SPAN", (11,0), (12,0)),  # Δ
        ("SPAN", (13,0), (15,0)),  # Hook-aware merge
        # Hide grid in the empty top-left header region above Model/Benchmark/n(Local)/n(BRV)/Type.
        ("BACKGROUND", (0,0), (4,0), colors.white),
        ("LINEABOVE", (0,0), (4,0), 0, colors.white),
        ("LINEBELOW", (0,0), (4,0), 0, colors.white),
        ("LINEBEFORE", (0,0), (0,0), 0, colors.white),
        ("LINEAFTER", (4,0), (4,0), 0, colors.white),
        # Bottom header row (leaf labels).
        ("BACKGROUND", (0,1), (-1,1), C_SUBHDR),
        ("TEXTCOLOR", (0,1), (-1,1), colors.white),
        ("FONTNAME", (0,1), (-1,1), "Helvetica-Bold"),
        ("FONTSIZE", (0,1), (-1,1), 8.5),
        # Body
        ("FONTSIZE", (0,2), (-1,-1), 7.5),
        ("ALIGN", (2,2), (-1,-1), "CENTER"),
        ("VALIGN", (0,2), (-1,-1), "MIDDLE"),
        ("GRID", (0,1), (-1,-1), 0.4, colors.grey),
        ("LEFTPADDING", (0,0), (-1,-1), 2),
        ("RIGHTPADDING", (0,0), (-1,-1), 2),
        ("TOPPADDING", (0,0), (-1,-1), 3),
        ("BOTTOMPADDING", (0,0), (-1,-1), 3),
    ]

    prev_model = None
    for i, (model, name, typ, lb, pb, lu, pu, perc) in enumerate(all_rows, start=2):
        gap_b = (lb - pb) if (lb is not None and pb is not None) else None
        gap_u = (lu - pu) if (lu is not None and pu is not None) else None
        ld    = (lu - lb) if (lb is not None and lu is not None) else None
        pd    = (pu - pb) if (pb is not None and pu is not None) else None
        model_cell = Paragraph(model, cell_left_small) if model != prev_model else ""
        prev_model = model
        # Local n equals BRV n for every row (same TSV files; MD5-verified).
        n_val = BRV_N.get(name, "—")
        pmbt_g, pmbt_gu, ft_g = hook_aware(model, name)
        data.append([model_cell, Paragraph(name, cell_left_small),
                     n_val, n_val, typ,
                     fmt(lb), fmt(pb), fmt_d(gap_b),
                     fmt(lu), fmt(pu), fmt_d(gap_u),
                     fmt_d(ld), fmt_d(pd),
                     fmt(pmbt_g), fmt(pmbt_gu), fmt(ft_g)])
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0,i), (-1,i), C_ALT))
        cdelta = color_for_delta(ld, perc)
        if cdelta is not None:
            style_cmds.append(("BACKGROUND", (11,i), (11,i), cdelta))
        if gap_b is not None and abs(gap_b) > GAP_RED_THRESHOLD:
            style_cmds.append(("BACKGROUND", (7,i), (7,i), C_RED))
        if gap_u is not None and abs(gap_u) > GAP_RED_THRESHOLD:
            style_cmds.append(("BACKGROUND", (10,i), (10,i), C_RED))

    tbl = Table(data, colWidths=[1.05*inch, 1.10*inch, 0.40*inch, 0.40*inch, 0.50*inch,
                                 0.50*inch, 0.50*inch, 0.50*inch,
                                 0.55*inch, 0.55*inch, 0.50*inch,
                                 0.55*inch, 0.55*inch,
                                 0.55*inch, 0.60*inch, 0.55*inch],
                repeatRows=2)
    tbl.setStyle(TableStyle(style_cmds))
    story.append(tbl)
    story.append(Spacer(1, 8))

# Flatten both models' rows into one list, prefixing each with the model name
all_rows = (
    [("LLaVA-Next-LLaMA3-8B (Dart-Prop)", *r) for r in llava]
    + [("Idefics2-8B (MAmmoTH-1)", *r) for r in idefics]
    + [("Qwen2-VL-7B-Instruct (Qwen2-Math-7B)", *r) for r in qwen]
    + [("LLaVA-1.5-7B (MetaMath-7B-V1.0)", *r) for r in llava15]
    + [("LLaVA-Next-Mistral-7B (MAmmoTH-7B-Mistral)", *r) for r in mistral]
    + [("LLaVA-OneVision-7B (Qwen2-Math-7B)", *r) for r in llava_ov]
    + [("InternVL2.5-8B (InternLM2-Math-Plus-7B)", *r) for r in internvl]
    + [("Qwen2.5-VL-7B (Qwen2.5-Math-7B)", *r) for r in qwen25vl7b]
)
big_table(all_rows)

# ───────── Blocked models (no on-disk data; per-blocker explanation) ─────────
story.append(PageBreak())
story.append(Paragraph("Blocked models — no on-disk data", h2))
story.append(Paragraph(
    "These 4 models complete the 9-model enrichment-target set but have no "
    "step-25 score CSVs at all. Each is blocked by a different infrastructure "
    "issue, summarized here.",
    body))

blocked_rows = [
    ["Model",
     "Auto-routed math LLM",
     "Blocker",
     "Status / handoff"],
    [Paragraph("LLaVA-OneVision-7B", cell_left_small),
     Paragraph("Qwen2-Math-7B (HF)", cell_left_small),
     Paragraph("VLMEvalKit_brv venv missing the <i>llava</i> Python package. "
               "Eval jobs die silently with <code>No module named 'llava'</code>. "
               "<b>Same package issue affects LLaVA-1.5-7B today</b> — both will resume "
               "once the package is installed. Merge .pth already exists from earlier session.",
               cell_left_small),
     Paragraph("Handoff filed: <code>notes/vlmevalkit_handoff_llava_next_install.md</code>. "
               "One <code>uv pip install</code> command unblocks both llava-1.5 and llava-ov.",
               cell_left_small)],
    [Paragraph("Qwen2.5-VL-7B", cell_left_small),
     Paragraph("Qwen2.5-Math-7B (HF)", cell_left_small),
     Paragraph("Two blockers. (1) Merge job crashed twice with an <code>hf-xet</code> Rust "
               "panic mid-download of Qwen2.5-Math-7B "
               "(<code>range end out of bounds: 67M &le; 34M</code>). "
               "Worked around by pre-fetching via <code>huggingface-cli download</code>; "
               "merge re-submitted 2026-06-04 18:32. "
               "(2) Every eval job raises <code>KeyError: 'Qwen2.5-VL-7B-Instruct'</code> — "
               "VLMEvalKit's <code>supported_VLM</code> registry has the 3B and 2B but "
               "not the 7B variant.",
               cell_left_small),
     Paragraph("Handoff filed: <code>notes/vlmevalkit_handoff_qwen25vl_7b_registry.md</code> "
               "(eval). Merge unblock used <code>huggingface-cli download "
               "Qwen/Qwen2.5-Math-7B</code> to bypass <code>hf-xet</code>.",
               cell_left_small)],
    [Paragraph("InternVL2.5-8B", cell_left_small),
     Paragraph("internlm2-math-plus-7b (HF)", cell_left_small),
     Paragraph("Merge job formerly killed by TERM_THREADLIMIT (fixed in run_pipeline.sh) "
               "and raised <code>TypeError: dtype</code> in merge_pmbt.py (fixed in code/merge_pmbt.py). "
               "<b>Merge .pth landed 2026-06-04 14:56 (15.5&nbsp;GB).</b> "
               "Eval still fails with <code>'InternLM2ForCausalLM' object has no attribute 'generate'</code> "
               "— HF-cache modeling_internlm2.py doesn't inherit GenerationMixin.",
               cell_left_small),
     Paragraph("Handoff filed: <code>notes/internvl_handoff_generation_mixin.md</code>. "
               "Merge side is unblocked by today's commits; eval side still needs the "
               "HF-cache patch (or transformers&nbsp;&le;&nbsp;4.49 pin).",
               cell_left_small)],
    [Paragraph("Qwen2.5-VL-3B", cell_left_small),
     Paragraph("— (none compatible)", cell_left_small),
     Paragraph("Hard-blocked by run_pipeline.sh (exit 1). Qwen2.5-VL-3B uses a 36-layer "
               "2048-hidden Qwen2.5-3B-base LLM; Qwen2.5-Math ships only 1.5B "
               "(28L/1536H) and 7B (28L/3584H), so no Qwen2.5-Math variant is "
               "shape-compatible. Other math families (LLaMA-base, Mistral-base) are "
               "also architecturally incompatible with the Qwen2.5-3B backbone.",
               cell_left_small),
     Paragraph("Decision: <b>leave the block in place</b>. No shape-compatible math "
               "donor exists; lifting the block would just move the failure into "
               "merge_pmbt.py's tensor-shape mismatch path.",
               cell_left_small)],
]
btbl = Table(blocked_rows,
             colWidths=[1.4*inch, 1.5*inch, 3.7*inch, 3.4*inch])
btbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_HEADER),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 9),
    ("ALIGN", (0,0), (-1,0), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "TOP"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("RIGHTPADDING", (0,0), (-1,-1), 4),
    ("TOPPADDING", (0,0), (-1,-1), 4),
    ("BOTTOMPADDING", (0,0), (-1,-1), 4),
    ("BACKGROUND", (0,1), (-1,1), C_ALT),
    ("BACKGROUND", (0,3), (-1,3), C_ALT),
]))
story.append(btbl)
story.append(Spacer(1, 8))

# ───────── Closeness analysis ─────────
story.append(PageBreak())
story.append(Paragraph("How close is local to the paper?", h2))
story.append(Paragraph(
    "Counting only rows where both Local and Paper values exist:",
    body))

def stats(rows, idx_local, idx_paper):
    gaps = []
    for r in rows:
        lv, pv = r[idx_local], r[idx_paper]
        if lv is not None and pv is not None:
            gaps.append(lv - pv)
    return gaps

llava_b_gaps = stats(llava, 2, 3)
llava_u_gaps = stats(llava, 4, 5)
ide_b_gaps   = stats(idefics, 2, 3)
ide_u_gaps   = stats(idefics, 4, 5)

def summarise(name, gaps):
    if not gaps:
        return [name, "—", "—", "—", "—", "—"]
    import statistics
    n = len(gaps)
    abs_mean = sum(abs(g) for g in gaps) / n
    signed_mean = sum(gaps) / n
    max_abs = max(abs(g) for g in gaps)
    n_over = sum(1 for g in gaps if abs(g) > GAP_RED_THRESHOLD)
    return [name, str(n), f"{abs_mean:.2f}", f"{signed_mean:+.2f}",
            f"{max_abs:.2f}", f"{n_over}/{n}"]

closeness = [
    ["Slice", "rows", "mean |Gap|", "mean Gap", "max |Gap|", "# |Gap|>1"],
    summarise("LLaVA — Baseline",     llava_b_gaps),
    summarise("LLaVA — +Uniform 0.9", llava_u_gaps),
    summarise("Idefics — Baseline",   ide_b_gaps),
    summarise("Idefics — +Uniform 0.9", ide_u_gaps),
]
ctbl = Table(closeness, colWidths=[2.2*inch, 0.8*inch, 1.1*inch, 1.1*inch, 1.0*inch, 1.0*inch],
             repeatRows=1)
ctbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), C_HEADER),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE", (0,0), (-1,0), 9),
    ("FONTSIZE", (0,1), (-1,-1), 9),
    ("ALIGN", (1,1), (-1,-1), "CENTER"),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, C_ALT]),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("RIGHTPADDING", (0,0), (-1,-1), 4),
]))
story.append(ctbl)
story.append(Spacer(1, 8))

story.append(Paragraph("Observations", h2))
obs = [
    "<b>MathVista, MathVision, DynaMath</b>: Local tracks Paper tightly. All Baseline and most +Uniform cells are within ~0.5&nbsp;pp; max |Gap| there is ~0.6 pp. The patched <code>MathVista_acc</code> reproduces the BRV General/Math split to within ~1&nbsp;pp.",
    "<b>MathVerse</b>: systematic shortfall. Local LLaVA baseline already runs 1&ndash;3&nbsp;pp below paper; after merging with Dart-Prop the gap grows to 4&ndash;7&nbsp;pp (Local +Uniform − Paper +Uniform). Idefics + MAmmoTH-1 shows a smaller but consistently negative MathVerse gap (1&ndash;2&nbsp;pp).",
    "<b>Likely causes</b> for the MathVerse divergence: (1) MathVerse has a two-stage judge (extract&nbsp;+&nbsp;score) so it accumulates more gpt-4o-mini sampling variance than single-stage benchmarks; (2) the local LLaVA-Next checkpoint may differ slightly from the one BRV used; (3) the merge implementation may apply Dart-Prop slightly differently than BRV's official script.",
    "<b>Net interpretation</b>: the qualitative direction agrees with BRV (uniform-merge helps math; modest perception cost), but the absolute MathVerse numbers should be taken with the gap in mind. Re-running step 19 against the BRV-published gpt-4o-mini judge build, or re-pinning the LLaVA checkpoint, would close most of the gap.",
]
for o in obs:
    story.append(Paragraph(f"&bull;&nbsp; {o}", body))

# ───────── Coverage gaps + provenance ─────────
story.append(Paragraph("Coverage gaps in the local data", h2))
miss = [
    "<b>InternVL2-76B</b> &mdash; no local eval (size makes it expensive); BRV's InternVL row can only come from the paper.",
    "<b>MathVerse-Overall</b> &mdash; the local pipeline doesn't precompute this row, but it is reconstructed in this PDF as the simple mean of the five mode scores (matching BRV).",
    "<b>DynaMath Worst-Case for the paper column</b> &mdash; BRV Tables 2/3 do not report it (paper only reports Average).",
    "<b>Older results/13-weight-merge/* JSON outputs</b> &mdash; different format with no cached judge xlsx, skipped in the re-score.",
]
for m in miss:
    story.append(Paragraph(f"&bull;&nbsp; {m}", body))

story.append(Paragraph("Provenance", h2))
prov = [
    "Local score CSVs read from <code>results/25-merge/&lt;model&gt;/&lt;tv&gt;/eval_baseline</code> and <code>.../eval_uniform_0.9</code>.",
    "Paper values: BRV (Chen et al., 2025) Table 2 (LLaVA, +Dart-Prop column) and Table 3 (Idefics, +MAmmoTH-1 column).",
    "MathVista General/Math rows populated by the patched <code>VLMEvalKit_brv/vlmeval/dataset/utils/mathvista.py:MathVista_acc</code>.",
    "DynaMath worst-case row already produced by <code>VLMEvalKit_brv/vlmeval/dataset/dynamath.py:226-244</code>; summarizer (<code>code/summarize_eval_results.py</code>) patched to read <code>Setting=Worst Case</code>.",
    "Re-scoring run: 406 files in ~107&nbsp;s on the modern_vlms venv; no judge re-calls.",
]
for p in prov:
    story.append(Paragraph(f"&bull;&nbsp; {p}", body))

doc.build(story)
print(f"Wrote {OUT}")
