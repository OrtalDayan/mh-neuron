"""Template copy of local_results_summary.pdf for the uniform-alpha sweep.

Derived from gen_local_results_summary_pdf.py. Two tables:
  Table 1 — consolidated reference, ALL score cells blank (fill-in template),
            full Baseline/Paper/Gap structure plus +Uniform 0.9 / 0.8 / 0.7
            groups and a Δ(vs Baseline) group per uniform alpha. All 9
            enrichment-target models.
  Table 2 — previously-blocked models, updated to current state (all now have
            on-disk data → resolved).

Writes a NEW file; does not overwrite local_results_summary.pdf.
"""
from reportlab.lib.pagesizes import TABLOID, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)
from reportlab.lib.enums import TA_JUSTIFY

OUT = ("/home/projects/bagon/ortalda/myrepo-weight_merging/modality_taxonomy/"
       "notes/local_results_summary_uniform_sweep_template.pdf")

# ── Colors / styles (identical to the source generator) ──
C_HEADER = colors.HexColor("#2a4d69")
C_SUBHDR = colors.HexColor("#5c7a93")
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

doc = SimpleDocTemplate(OUT, pagesize=landscape(TABLOID),
    leftMargin=0.35*inch, rightMargin=0.35*inch,
    topMargin=0.45*inch, bottomMargin=0.45*inch,
    title="Local Results Summary: Uniform-Sweep Template")

story = []
story.append(Paragraph(
    "Local Results Summary &mdash; Uniform-Sweep Template (0.9 / 0.8 / 0.7)", h1))
story.append(Paragraph(
    "Blank fill-in template derived from <code>local_results_summary.pdf</code>. "
    "Table 1 is the consolidated &ldquo;All values, row-by-row&rdquo; reference with "
    "<b>all score cells intentionally empty</b> &mdash; one row per (model &times; "
    "benchmark), to be filled as the uniform &alpha; = 0.9 / 0.8 / 0.7 evals land. "
    "Table 2 lists the four models that were previously blocked, updated to the "
    "current state (all now have on-disk data).",
    body))

# ── Shared row definitions (same 14 benchmark rows + Type as the source) ──
BENCH_ROWS = [
    ("MathVista-Overall",       "perception"),
    ("MathVista-General (NEW)", "perception"),
    ("MathVista-Math (NEW)",    "reasoning"),
    ("MathVerse-Overall",       "reasoning"),
    ("MathVerse-T-D",           "reasoning"),
    ("MathVerse-T-L",           "reasoning"),
    ("MathVerse-V-I",           "reasoning"),
    ("MathVerse-V-D",           "reasoning"),
    ("MathVerse-V-O",           "reasoning"),
    ("MathVision",              "reasoning"),
    ("MMStar-Overall",          "perception"),
    ("MMStar-Math",             "reasoning"),
    ("DynaMath-Average",        "reasoning"),
    ("DynaMath-Worst (NEW)",    "reasoning"),
]
BRV_N = {
    "MathVista-Overall": "1000", "MathVista-General (NEW)": "460",
    "MathVista-Math (NEW)": "540", "MathVerse-Overall": "788*",
    "MathVerse-T-D": "788", "MathVerse-T-L": "788", "MathVerse-V-I": "788",
    "MathVerse-V-D": "788", "MathVerse-V-O": "788", "MathVision": "304",
    "MMStar-Overall": "1500", "MMStar-Math": "250",
    "DynaMath-Average": "5010", "DynaMath-Worst (NEW)": "501",
}
# All 9 enrichment-target models (donor shown in parens = step-25 uniform donor).
MODELS = [
    "LLaVA-Next-LLaMA3-8B (Dart-Prop)",
    "Idefics2-8B (MAmmoTH-1)",
    "Qwen2-VL-7B-Instruct (Qwen2-Math-7B)",
    "LLaVA-1.5-7B (MetaMath-7B-V1.0)",
    "LLaVA-Next-Mistral-7B (Dart-Math-Mistral-Uniform)",
    "LLaVA-OneVision-7B (Qwen2-Math-7B)",
    "InternVL2.5-8B (InternLM2-Math-Plus-7B)",
    "Qwen2.5-VL-7B (Qwen2.5-Math-7B)",
    "Qwen2.5-VL-3B (Qwen2.5-3B-DAPO-math-reasoning)",
]

# ── Table 1: consolidated template, all score cells blank ──
story.append(Paragraph("All values, row-by-row (consolidated reference) — blank template", h2))
story.append(Paragraph(
    "One row per (model &times; benchmark). Score cells are intentionally empty: "
    "fill <b>Local</b> from each step-25 eval, <b>Paper</b> from BRV where reported, "
    "<b>Gap</b> = Local &minus; Paper, and each <b>&Delta;</b> group = (+Uniform &alpha;) "
    "&minus; Baseline. <b>n (Local) = n (BRV)</b> for every row (same MD5-matched TSVs); "
    "MathVerse-Overall 788* is the effective n (five modes share 788 problems).",
    note))

# Header: group banner row + leaf row.
# Uniform 0.8/0.7 have no BRV-published value → drop their (always-empty) Paper
# and Gap columns; same for the Δ 0.8/0.7 Paper subcols.
GROUPS = [("Baseline", 3), ("+Uniform 0.9", 3), ("+Uniform 0.8", 1),
          ("+Uniform 0.7", 1), ("Δ 0.9 vs base", 2), ("Δ 0.8", 1),
          ("Δ 0.7", 1)]
hdr1 = ["", "", "", "", ""]
for title, span in GROUPS:
    hdr1 += [title] + [""] * (span - 1)
hdr2 = ["Model", "Benchmark", "n (Local)", "n (BRV)", "Type"]
for _title, span in GROUPS:
    hdr2 += {3: ["Local", "Paper", "Gap"], 2: ["Local", "Paper"], 1: ["Local"]}[span]
NCOL = len(hdr2)

data = [hdr1, hdr2]
style_cmds = [
    ("BACKGROUND", (5, 0), (-1, 0), C_HEADER),
    ("TEXTCOLOR", (5, 0), (-1, 0), colors.white),
    ("FONTNAME", (5, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (5, 0), (-1, 0), 8.5),
    ("ALIGN", (0, 0), (-1, 1), "CENTER"),
    ("VALIGN", (0, 0), (-1, 1), "MIDDLE"),
    # white gap above the id columns
    ("BACKGROUND", (0, 0), (4, 0), colors.white),
    ("LINEABOVE", (0, 0), (4, 0), 0, colors.white),
    ("LINEBELOW", (0, 0), (4, 0), 0, colors.white),
    ("LINEBEFORE", (0, 0), (0, 0), 0, colors.white),
    ("LINEAFTER", (4, 0), (4, 0), 0, colors.white),
    ("BACKGROUND", (0, 1), (-1, 1), C_SUBHDR),
    ("TEXTCOLOR", (0, 1), (-1, 1), colors.white),
    ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
    ("FONTSIZE", (0, 1), (-1, 1), 8),
    ("FONTSIZE", (0, 2), (-1, -1), 7.5),
    ("ALIGN", (2, 2), (-1, -1), "CENTER"),
    ("VALIGN", (0, 2), (-1, -1), "MIDDLE"),
    ("GRID", (0, 1), (-1, -1), 0.4, colors.grey),
    ("LEFTPADDING", (0, 0), (-1, -1), 2),
    ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ("TOPPADDING", (0, 0), (-1, -1), 3),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
]
# group-banner SPANs
col = 5
for _title, span in GROUPS:
    style_cmds.append(("SPAN", (col, 0), (col + span - 1, 0)))
    col += span

i = 2
for model in MODELS:
    prev_model = None
    for bname, btype in BENCH_ROWS:
        model_cell = Paragraph(model, cell_left_small) if model != prev_model else ""
        prev_model = model
        n_val = BRV_N.get(bname, "—")
        row = [model_cell, Paragraph(bname, cell_left_small), n_val, n_val, btype]
        row += [""] * (NCOL - 5)   # blank score cells
        data.append(row)
        if i % 2 == 0:
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), C_ALT))
        i += 1

_idw = [1.30*inch, 1.30*inch, 0.50*inch, 0.50*inch, 0.55*inch]
_valw = [0.66*inch] * (NCOL - 5)
tbl = Table(data, colWidths=_idw + _valw, repeatRows=2)
tbl.setStyle(TableStyle(style_cmds))
story.append(tbl)
story.append(Spacer(1, 8))

# ── Table 2: previously-blocked models, updated to current state ──
story.append(PageBreak())
story.append(Paragraph("Previously-blocked models — now resolved", h2))
story.append(Paragraph(
    "All four models that completed the 9-model enrichment-target set previously had "
    "no step-25 score CSVs. As of this update <b>every one now has on-disk eval data</b>, "
    "so their blockers are resolved. CSV counts are score/acc files under "
    "<code>results/25-merge/&lt;model&gt;/</code> at update time.",
    body))

blocked_rows = [
    ["Model", "Auto-routed math LLM", "Prior blocker", "Current status"],
    [Paragraph("LLaVA-OneVision-7B", cell_left_small),
     Paragraph("Qwen2-Math-7B (HF)", cell_left_small),
     Paragraph("VLMEvalKit_brv venv missing the <i>llava</i> Python package; eval jobs "
               "died with <code>No module named 'llava'</code> (also hit LLaVA-1.5-7B).",
               cell_left_small),
     Paragraph("<b>Resolved.</b> 24 merges, 402 score/acc CSVs on disk. "
               "(Handoff: <code>notes/vlmevalkit_handoff_llava_next_install.md</code>.)",
               cell_left_small)],
    [Paragraph("Qwen2.5-VL-7B", cell_left_small),
     Paragraph("Qwen2.5-Math-7B (HF)", cell_left_small),
     Paragraph("(1) <code>hf-xet</code> Rust panic mid-download of Qwen2.5-Math-7B; "
               "(2) eval <code>KeyError: 'Qwen2.5-VL-7B-Instruct'</code> "
               "(missing from VLMEvalKit <code>supported_VLM</code>).",
               cell_left_small),
     Paragraph("<b>Resolved.</b> 24 merges, 357 score/acc CSVs on disk. "
               "(Handoff: <code>notes/vlmevalkit_handoff_qwen25vl_7b_registry.md</code>.)",
               cell_left_small)],
    [Paragraph("InternVL2.5-8B", cell_left_small),
     Paragraph("internlm2-math-plus-7b (HF)", cell_left_small),
     Paragraph("Merge once killed by TERM_THREADLIMIT + <code>TypeError: dtype</code> "
               "(both fixed); eval failed with "
               "<code>'InternLM2ForCausalLM' object has no attribute 'generate'</code> "
               "(GenerationMixin not inherited).",
               cell_left_small),
     Paragraph("<b>Resolved.</b> 24 merges, 449 score/acc CSVs on disk. "
               "(Handoff: <code>notes/internvl_handoff_generation_mixin.md</code>.)",
               cell_left_small)],
    [Paragraph("Qwen2.5-VL-3B", cell_left_small),
     Paragraph("Qwen2.5-3B-DAPO-math-reasoning", cell_left_small),
     Paragraph("Hard-blocked (run_pipeline.sh exit 1): official Qwen2.5-Math ships only "
               "1.5B/7B (28L), shape-incompatible with the 36-layer Qwen2.5-3B backbone, "
               "so no math donor was wired in.",
               cell_left_small),
     Paragraph("<b>Resolved.</b> Donor <code>jaygala24/Qwen2.5-3B-DAPO-math-reasoning</code> "
               "(36-layer, shape-compatible) added to step 25; 4 merges, 74 score/acc CSVs "
               "on disk.",
               cell_left_small)],
]
btbl = Table(blocked_rows, colWidths=[1.4*inch, 1.7*inch, 3.6*inch, 3.4*inch])
btbl.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), C_HEADER),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("FONTSIZE", (0, 0), (-1, 0), 9),
    ("ALIGN", (0, 0), (-1, 0), "CENTER"),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
    ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ("TOPPADDING", (0, 0), (-1, -1), 4),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ("BACKGROUND", (0, 1), (-1, 1), C_ALT),
    ("BACKGROUND", (0, 3), (-1, 3), C_ALT),
]))
story.append(btbl)
story.append(Spacer(1, 8))

doc.build(story)
print(f"Wrote {OUT}")
