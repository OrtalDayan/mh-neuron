#!/usr/bin/env python3
"""Render a standalone PDF of all BASELINE + UNIFORM step-25 eval results.

Reads the flat coverage CSV produced by extract_merge_coverage.py
(columns: model,tv,mode,benchmark,score,csv_path,mtime) and emits one
landscape table per mode (baseline, uniform_a0.7/0.8/0.85/0.9):
rows = models, columns = benchmarks. Missing cells show as "—".
"""
import csv
import sys
from collections import defaultdict
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

CSV = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cov_fresh.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else (
    "/home/projects/bagon/ortalda/myrepo-weight_merging/modality_taxonomy/"
    "notes/local_results_baseline_uniform.pdf")

# Fixed display orders
MODES = [
    ("baseline",      "Baseline (unmerged VLM)"),
    ("uniform_a0.9",  "Uniform merge α=0.9"),
    ("uniform_a0.85", "Uniform merge α=0.85"),
    ("uniform_a0.8",  "Uniform merge α=0.8"),
    ("uniform_a0.7",  "Uniform merge α=0.7"),
]
MODELS = [
    "llava-1.5-7b", "llava-next-mistral-7b", "llava-next-llama3-8b",
    "llava-onevision-7b", "qwen2-vl-7b", "qwen2.5-vl-7b", "qwen2.5-vl-3b",
    "idefics2-8b", "internvl2.5-8b",
]
# benchmark -> short column label, in display order
BENCHES = [
    ("MathVista", "MVista"), ("MathVista-General", "MVi-Gen"),
    ("MathVista-Math", "MVi-Mth"), ("MathVerse-T-D", "MV-TD"),
    ("MathVerse-T-L", "MV-TL"), ("MathVerse-V-I", "MV-VI"),
    ("MathVerse-V-D", "MV-VD"), ("MathVerse-V-O", "MV-VO"),
    ("MathVision", "MVision"), ("MMStar", "MMStar"),
    ("MMStar-Math", "MMS-Mth"), ("DynaMath", "Dyna"),
    ("DynaMath-Worst", "Dyna-W"), ("MME", "MME"), ("POPE", "POPE"),
    ("HallusionBench", "Hallu"), ("TriviaQA", "Trivia"),
]

# (mode, model, benchmark) -> score string
data = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        data[(r["mode"], r["model"], r["benchmark"])] = r["score"]


def fmt(s):
    if s is None:
        return "—"
    try:
        return f"{float(s):.1f}"
    except (ValueError, TypeError):
        return str(s)


styles = getSampleStyleSheet()
doc = SimpleDocTemplate(OUT, pagesize=landscape(letter),
                        leftMargin=0.3 * inch, rightMargin=0.3 * inch,
                        topMargin=0.4 * inch, bottomMargin=0.4 * inch)
flow = [Paragraph("Baseline & Uniform-Merge Results (step 25)", styles["Title"]),
        Paragraph("Scores per model × benchmark. — = not evaluated. "
                  "Generated from on-disk eval CSVs.", styles["Normal"]),
        Spacer(1, 0.15 * inch)]

short_models = {m: m.replace("llava-next-", "ll-").replace("llava-", "llava")
                for m in MODELS}

for mode, title in MODES:
    # skip a mode entirely absent from the data
    if not any((mode, m, b) in data for m in MODELS for b, _ in BENCHES):
        continue
    header = ["model"] + [lbl for _, lbl in BENCHES]
    rows = [header]
    for m in MODELS:
        row = [short_models[m]]
        for b, _ in BENCHES:
            row.append(fmt(data.get((mode, m, b))))
        rows.append(row)
    col_w = [1.25 * inch] + [(10.0 / len(BENCHES)) * 0.62 * inch] * len(BENCHES)
    t = Table(rows, colWidths=col_w, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 6),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor("#f2f4f5")]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bdc3c7")),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    flow.append(Paragraph(title, styles["Heading3"]))
    flow.append(t)
    flow.append(Spacer(1, 0.22 * inch))

doc.build(flow)
print(f"Wrote {OUT}")
