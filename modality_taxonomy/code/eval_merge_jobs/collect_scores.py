"""Collect score files across the 4 merged-eval arms into one comparison table.

Produces:
  results/25-merge/llava-1.5-7b/MetaMath-7B-V1.0/merged_eval_summary.csv
  results/25-merge/llava-1.5-7b/MetaMath-7B-V1.0/merged_eval_summary.md

Rows = 11 benchmarks, cols = the 4 arms. Each cell is the primary headline metric
normalized to a percentage. VLMEvalKit names/formats the headline file differently
per benchmark, so each benchmark has an explicit (glob, extractor) spec rather than a
generic "first numeric" guess:

  MathVista / MathVision : <...>_score.csv          "Overall" row, 'acc' column      (already %)
  MathVerse (5 splits)   : <...>_score_score.csv    row 1, 'accuracy' column         (already %)
  MMStar                 : <...>_acc.csv            "none" row, 'Overall' column     (fraction ->x100)
  DynaMath               : <...>_score.csv          "Average" row, 'Overall' column  (fraction ->x100)
  MM-Math                : <...>_score.json         'overall' key                    (fraction ->x100)
  POPE                   : <...>_score.csv          "Overall" row, 'Overall' (F1)    (already %)
"""
import os
import csv as csvmod
import glob
import json
from pathlib import Path

RESULTS_BASE = Path("/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy/"
                    "results/25-merge/llava-1.5-7b/MetaMath-7B-V1.0")

ARMS = [
    ("baseline",        "baseline"),                       # unmerged stock LLaVA-1.5 (no-merge ref)
    ("uniform",         "uniform_a0.9"),                    # BRV uniform linear interp, alpha=0.9
    # PMBT @ alpha=0.05 (preserved in _alpha_0.05 siblings). The original
    # hook_*_pmbt_t0.9_v1.0_m1.0/ paths are reserved for the new alpha=0.001 PMBT
    # merges once Session C's reclassified labels land.
    ("gate_pmbt_a05",   "hook_gate_pmbt_t0.9_v1.0_m1.0_alpha_0.05"),
    ("gateup_pmbt_a05", "hook_gateup_pmbt_t0.9_v1.0_m1.0_alpha_0.05"),
    ("gate_ft",         "hook_gate_ft_t0.9_v1.0_m1.0"),
]

# (benchmark, glob under <arm>/eval/, extractor kind)
BENCH_SPEC = [
    ("MathVista_MINI",                  "*MathVista_MINI*_score.csv",                       "row_acc"),
    ("MathVerse_MINI_Text_Dominant",    "*MathVerse_MINI_Text_Dominant*_score_score.csv",   "mathverse"),
    ("MathVerse_MINI_Text_Lite",        "*MathVerse_MINI_Text_Lite*_score_score.csv",       "mathverse"),
    ("MathVerse_MINI_Vision_Intensive", "*MathVerse_MINI_Vision_Intensive*_score_score.csv", "mathverse"),
    ("MathVerse_MINI_Vision_Dominant",  "*MathVerse_MINI_Vision_Dominant*_score_score.csv", "mathverse"),
    ("MathVerse_MINI_Vision_Only",      "*MathVerse_MINI_Vision_Only*_score_score.csv",     "mathverse"),
    ("MMStar",                          "*MMStar*_acc.csv",                                 "mmstar"),
    ("DynaMath",                        "*DynaMath*_score.csv",                             "dynamath"),
    ("MathVision_MINI",                 "*MathVision_MINI*_score.csv",                      "row_acc"),
    ("MM-Math",                         "*MM-Math*_score.json",                             "mmmath_json"),
    ("POPE",                            "*POPE*_score.csv",                                 "pope"),
]


def find_file(arm_dir: Path, pattern: str):
    """Newest file (by mtime) matching pattern anywhere under <arm_dir>/eval/."""
    eval_dir = arm_dir / "eval"
    if not eval_dir.exists():
        return None
    matches = glob.glob(str(eval_dir / "**" / pattern), recursive=True)
    if not matches:
        return None
    matches.sort(key=os.path.getmtime, reverse=True)
    return matches[0]


def _read_csv(path):
    with open(path) as f:
        return list(csvmod.reader(f))


def _row_by_label(rows, label):
    """First data row whose first cell == label (case-insensitive)."""
    for r in rows[1:]:
        if r and r[0].strip().lower() == label.lower():
            return r
    return None


def _col_index(header, name):
    for i, h in enumerate(header):
        if h.strip().lower() == name.lower():
            return i
    return None


def _f(x):
    try:
        return float(x)
    except (ValueError, TypeError):
        return None


def extract(kind: str, path: str):
    """Return the headline metric as a percentage (float), or None if unparseable."""
    if kind == "mmmath_json":
        with open(path) as f:
            d = json.load(f)
        v = _f(d.get("overall"))
        return v * 100 if v is not None else None

    rows = _read_csv(path)
    if len(rows) < 2:
        return None
    header = rows[0]

    if kind == "row_acc":  # MathVista / MathVision: "Overall" row, 'acc' col (already %)
        r = _row_by_label(rows, "Overall")
        ci = _col_index(header, "acc")
        return _f(r[ci]) if r and ci is not None else None

    if kind == "mathverse":  # 'accuracy' col, value in first data row (already %)
        ci = _col_index(header, "accuracy")
        return _f(rows[1][ci]) if ci is not None and ci < len(rows[1]) else None

    if kind == "mmstar":  # "none" row, 'Overall' col (fraction)
        r = _row_by_label(rows, "none") or (rows[1] if len(rows) > 1 else None)
        ci = _col_index(header, "Overall")
        v = _f(r[ci]) if r and ci is not None else None
        return v * 100 if v is not None else None

    if kind == "dynamath":  # "Average" row, 'Overall' col (fraction)
        r = _row_by_label(rows, "Average")
        ci = _col_index(header, "Overall")
        v = _f(r[ci]) if r and ci is not None else None
        return v * 100 if v is not None else None

    if kind == "pope":  # "Overall" row, 'Overall' col = F1 (already %)
        r = _row_by_label(rows, "Overall")
        ci = _col_index(header, "Overall")
        return _f(r[ci]) if r and ci is not None else None

    return None


def main():
    table = {}  # bench -> arm_name -> (value_or_None, path_or_"")
    for bench, pattern, kind in BENCH_SPEC:
        table[bench] = {}
        for arm_name, arm_suffix in ARMS:
            arm_dir = RESULTS_BASE / arm_suffix
            path = find_file(arm_dir, pattern)
            if not path:
                table[bench][arm_name] = (None, "")
                continue
            try:
                val = extract(kind, path)
            except Exception as e:  # noqa: BLE001 - report which file failed, keep going
                print(f"  [warn] {bench}/{arm_name}: failed to parse {path}: {e}")
                val = None
            table[bench][arm_name] = (val, path)

    cols = [a for a, _ in ARMS]
    benches = [b for b, _, _ in BENCH_SPEC]

    def fmt(v):
        return f"{v:.2f}" if v is not None else "MISSING"

    # CSV
    out_csv = RESULTS_BASE / "merged_eval_summary.csv"
    with open(out_csv, "w") as f:
        f.write("benchmark," + ",".join(cols) + "\n")
        for b in benches:
            f.write(b + "," + ",".join(fmt(table[b][a][0]) for a in cols) + "\n")

    # Markdown
    out_md = RESULTS_BASE / "merged_eval_summary.md"
    with open(out_md, "w") as f:
        f.write("| benchmark | " + " | ".join(cols) + " |\n")
        f.write("|---|" + "|".join(["---"] * len(cols)) + "|\n")
        for b in benches:
            f.write("| " + b + " | "
                    + " | ".join(fmt(table[b][a][0]) for a in cols) + " |\n")

    # Console
    print(f"\nwrote {out_csv}")
    print(f"wrote {out_md}\n")
    n_done = sum(1 for b in benches for a in cols if table[b][a][0] is not None)
    print(f"populated cells: {n_done}/{len(benches) * len(cols)}  (all values are %)\n")
    print(f"  {'benchmark':34s}  " + "  ".join(f"{a:>11}" for a in cols))
    for b in benches:
        print(f"  {b:34s}  " + "  ".join(f"{fmt(table[b][a][0]):>11}" for a in cols))


if __name__ == "__main__":
    main()
