#!/usr/bin/env python3
"""
Coverage-extraction walk for step-25 weight-merge results.

Walks results/25-merge/{model}/{tv}/ recursively, finds every *_score.csv,
classifies each into a (mode, benchmark) tuple based on its path, keeps the
latest by mtime per tuple, extracts the headline score, and prints a
markdown coverage matrix.

Handles the multiple historical layouts mixed in the tree:

  Newest:           baseline/eval/<model>/[Tstamp/]<bench>_score.csv
  Newest:           uniform_a<α>/eval/<model><α>/[Tstamp/]<bench>_score.csv
  Mid (Apr):        <tv>/baseline/<model>/<bench>_score.csv
  Old (Apr early):  <tv>/eval_baseline/<model>/<bench>_score.csv
  Old (Apr):        <tv>/eval_uniform_<α>/<model><α>/<bench>_score.csv
  Lambda sweep:     <tv>/eval/<model><α>/[Tstamp/]<bench>_score.csv
  PMBT:             <tv>/pmbt_t<T>_v<V>_m<M>[_...]/eval/<model><tag>/...

We do *not* classify PMBT/FT-variant results here — they go in a
"non-{baseline,uniform} other" bucket so the table stays focused on
baseline + uniform_a0.9 coverage. Run with --include-pmbt to surface them.

Usage:
    python3 code/extract_merge_coverage.py
    python3 code/extract_merge_coverage.py --root /path/to/results/25-merge
    python3 code/extract_merge_coverage.py --alpha 0.9 --csv coverage.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ── Benchmark name normalization ───────────────────────────────────────
# Map a filename fragment to a canonical short benchmark name.
BENCH_PATTERNS = [
    (re.compile(r"MathVista_MINI"),                   "MathVista"),
    (re.compile(r"MathVerse_MINI_Text_Dominant"),     "MathVerse-T-D"),
    (re.compile(r"MathVerse_MINI_Text_Lite"),         "MathVerse-T-L"),
    (re.compile(r"MathVerse_MINI_Vision_Intensive"),  "MathVerse-V-I"),
    (re.compile(r"MathVerse_MINI_Vision_Dominant"),   "MathVerse-V-D"),
    (re.compile(r"MathVerse_MINI_Vision_Only"),       "MathVerse-V-O"),
    (re.compile(r"MathVision_MINI"),                  "MathVision"),
    (re.compile(r"MMStar"),                           "MMStar"),
    (re.compile(r"DynaMath"),                         "DynaMath"),
    (re.compile(r"MM-Math"),                          "MM-Math"),
    (re.compile(r"POPE"),                             "POPE"),
    (re.compile(r"MME"),                              "MME"),
    (re.compile(r"TriviaQA"),                         "TriviaQA"),
    (re.compile(r"HallusionBench"),                   "HallusionBench"),
]


def normalize_bench(name: str) -> Optional[str]:
    for pat, canonical in BENCH_PATTERNS:
        if pat.search(name):
            return canonical
    return None


# ── Mode classification ────────────────────────────────────────────────
# Walk path components from leaf to root; the first matching segment wins.
ALPHA_RE = re.compile(r"^uniform_a(\d+(?:\.\d+)?)$")
EVAL_UNIFORM_RE = re.compile(r"^eval_uniform_(\d+(?:\.\d+)?)$")
# Model-name-with-trailing-alpha (e.g. Qwen2-VL-7B-Instruct0.85, llava_next_llama30.9)
SWEEP_DIR_RE = re.compile(r"(\d+\.\d+)$")
PMBT_RE = re.compile(r"^(ft_)?pmbt_t.*")


@dataclass
class Mode:
    bucket: str       # "baseline", "uniform", "pmbt", "other"
    alpha: str = ""   # for uniform: e.g. "0.9"; empty otherwise
    label: str = ""   # for pmbt: full mode-dir name; empty otherwise

    def key(self) -> str:
        if self.bucket == "uniform":
            return f"uniform_a{self.alpha}"
        if self.bucket == "pmbt":
            return self.label
        return self.bucket


def classify_mode(path: Path, model_root: Path) -> Optional[Mode]:
    """
    Walk path components between model_root and the score CSV; pick the
    bucket from the most-specific (deepest) matching segment.
    """
    try:
        rel = path.relative_to(model_root)
    except ValueError:
        return None
    parts = rel.parts

    # Direct mode-dir matches anywhere in the path
    for seg in parts:
        if seg == "baseline" or seg == "eval_baseline":
            return Mode("baseline")
        m = ALPHA_RE.match(seg)
        if m:
            return Mode("uniform", alpha=m.group(1))
        m = EVAL_UNIFORM_RE.match(seg)
        if m:
            return Mode("uniform", alpha=m.group(1))
        if PMBT_RE.match(seg):
            return Mode("pmbt", label=seg)

    # Lambda-sweep style: model name segment ends in a float (e.g. "Qwen2-VL-7B-Instruct0.85")
    # Walk from leaf up looking for it.
    for seg in reversed(parts):
        m = SWEEP_DIR_RE.search(seg)
        if m and not seg.startswith("T") and "." in m.group(1):
            return Mode("uniform", alpha=m.group(1))

    return None


# ── Score extraction ──────────────────────────────────────────────────
def extract_score(csv_path: Path, canonical_bench: str) -> Optional[float]:
    """
    Pull the headline % score from one *_score.csv using benchmark-specific
    column conventions. Returns None on parse failure.
    """
    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.reader(f))
    except (OSError, csv.Error):
        return None
    if len(rows) < 2:
        return None
    header = rows[0]
    r1 = rows[1]

    def col(name: str) -> Optional[int]:
        for i, h in enumerate(header):
            if h == name:
                return i
        return None

    try:
        if canonical_bench == "MathVista":
            i = col("acc")
            return float(r1[i]) if i is not None else None
        if canonical_bench.startswith("MathVerse-"):
            i = col("accuracy")
            return float(r1[i]) if i is not None else None
        if canonical_bench == "MathVision":
            i = col("acc")
            return float(r1[i]) if i is not None else None
        if canonical_bench == "DynaMath":
            i = col("Overall")
            return float(r1[i]) * 100 if i is not None else None
        if canonical_bench == "MMStar":
            i = col("Overall")
            v = float(r1[i]) if i is not None else None
            return v * 100 if v is not None and v < 1.5 else v
        if canonical_bench == "MM-Math":
            i = col("Overall") if col("Overall") is not None else col("acc")
            if i is None:
                return None
            v = float(r1[i])
            return v * 100 if v < 1.5 else v
        if canonical_bench == "POPE":
            i = col("Overall")
            return float(r1[i]) if i is not None else None
        if canonical_bench == "TriviaQA":
            i = col("Overall") if col("Overall") is not None else col("acc")
            if i is None:
                return None
            v = float(r1[i])
            return v * 100 if v < 1.5 else v
        if canonical_bench == "MME":
            ip = col("perception")
            ir = col("reasoning")
            if ip is not None and ir is not None:
                return float(r1[ip]) + float(r1[ir])
            i = col("Overall")
            return float(r1[i]) if i is not None else None
        if canonical_bench == "HallusionBench":
            i = col("aAcc") or col("Overall")
            return float(r1[i]) if i is not None else None
    except (ValueError, IndexError):
        return None
    return None


# ── Walk and aggregate ────────────────────────────────────────────────
@dataclass
class Entry:
    score: Optional[float]
    mtime: float
    path: Path


def _extract_extras(p: Path, primary_bench: str) -> list[tuple[str, Optional[float]]]:
    """
    Some benchmarks pack multiple BRV-style splits into a single _score.csv:

      MathVista_MINI → MathVista (Overall), MathVista-General, MathVista-Math
      DynaMath       → DynaMath (Average row), DynaMath-Worst (Worst Case row)

    For each split we return (canonical_sub_bench, score) so the index
    surfaces them as if they were independent benchmarks.
    """
    extras: list[tuple[str, Optional[float]]] = []
    try:
        with open(p, newline="") as f:
            rows = list(csv.reader(f))
    except (OSError, csv.Error):
        return extras
    if len(rows) < 2:
        return extras
    header = rows[0]

    if primary_bench == "MathVista":
        # CSV has per-skill rows; aggregate to General/Math via the BRV split
        # (Math = the "math-targeted-vqa" skill row; General = everything else
        # weighted by row count).
        try:
            i_tot = header.index("tot")
            i_hit = header.index("hit")
            i_acc = header.index("acc")
            i_skill = 0
        except ValueError:
            return extras
        math_hit = math_tot = 0
        gen_hit = gen_tot = 0
        for r in rows[2:]:  # skip header + Overall row
            try:
                tot = int(r[i_tot])
                hit = int(r[i_hit])
            except (ValueError, IndexError):
                continue
            if r[i_skill] == "math-targeted-vqa":
                math_hit, math_tot = hit, tot
            else:
                gen_hit += hit
                gen_tot += tot
        if math_tot:
            extras.append(("MathVista-Math", 100.0 * math_hit / math_tot))
        if gen_tot:
            extras.append(("MathVista-General", 100.0 * gen_hit / gen_tot))

    elif primary_bench == "MMStar":
        # MMStar's exact-matching acc CSV has columns including a "math" split.
        try:
            i_math = header.index("math")
            v = float(rows[1][i_math])
            extras.append(("MMStar-Math", v * 100 if v < 1.5 else v))
        except (ValueError, IndexError):
            pass

    elif primary_bench == "DynaMath":
        # Row 2 is "Average" (already extracted as DynaMath). Row 3 is
        # "Worst Case" — surface as DynaMath-Worst.
        try:
            i_set = header.index("Setting")
            i_overall = header.index("Overall")
        except ValueError:
            return extras
        for r in rows[2:]:
            if len(r) > max(i_set, i_overall) and r[i_set] == "Worst Case":
                try:
                    v = float(r[i_overall])
                except ValueError:
                    continue
                extras.append(("DynaMath-Worst", v * 100 if v < 1.5 else v))
                break
    return extras


def scan_model_tv(model_root: Path) -> dict[tuple[str, str], Entry]:
    """
    Return {(mode_key, bench): Entry} with latest-by-mtime per tuple.

    Globs both *_score.csv (canonical) and *_acc.csv (MMStar's
    exact-matching judge output uses this suffix).
    """
    out: dict[tuple[str, str], Entry] = {}
    paths = list(model_root.rglob("*_score.csv")) + list(model_root.rglob("*_acc.csv"))
    for p in paths:
        bench = normalize_bench(p.name)
        if bench is None:
            continue
        mode = classify_mode(p, model_root)
        if mode is None:
            continue
        mtime = p.stat().st_mtime
        # Primary benchmark
        key = (mode.key(), bench)
        prev = out.get(key)
        if prev is None or mtime > prev.mtime:
            out[key] = Entry(score=extract_score(p, bench), mtime=mtime, path=p)
        # Additional sub-benchmarks packed in the same CSV (MathVista
        # splits, DynaMath Worst Case, ...). Each gets its own index row.
        for sub_bench, sub_score in _extract_extras(p, bench):
            sub_key = (mode.key(), sub_bench)
            prev = out.get(sub_key)
            if prev is None or mtime > prev.mtime:
                out[sub_key] = Entry(score=sub_score, mtime=mtime, path=p)
    return out


def looks_like_mode_dir(name: str) -> bool:
    """
    True if a model-level subdir name looks like a *mode* dir
    (baseline / eval_baseline / uniform_a* / eval_uniform_* / [ft_]pmbt_*)
    rather than a task-vector dir. These stray mode dirs at the model level
    are legacy artifacts from earlier output-path conventions; treating them
    as task vectors yields empty rows in the coverage matrix.
    """
    if name in ("baseline", "eval_baseline", "eval"):
        return True
    if name.startswith("uniform_a") or name.startswith("eval_uniform_"):
        return True
    if PMBT_RE.match(name):
        return True
    return False


def discover_model_tv_pairs(root: Path) -> list[tuple[str, str, Path]]:
    """
    Return [(model, tv, tv_root_path), ...] under results/25-merge/.
    Only includes dirs that contain at least one _score.csv anywhere below.
    Skips model-level mode-like dirs (see looks_like_mode_dir).
    """
    pairs = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for tv_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            if looks_like_mode_dir(tv_dir.name):
                continue
            if any(tv_dir.rglob("*_score.csv")):
                pairs.append((model_dir.name, tv_dir.name, tv_dir))
    return pairs


# ── Output ─────────────────────────────────────────────────────────────
BENCH_ORDER = [
    "MathVista", "MathVista-General", "MathVista-Math",
    "MathVerse-T-D", "MathVerse-T-L",
    "MathVerse-V-I", "MathVerse-V-D", "MathVerse-V-O",
    "MathVision", "MMStar", "MMStar-Math",
    "DynaMath", "DynaMath-Worst", "MM-Math",
    "POPE", "MME", "TriviaQA", "HallusionBench",
]


def fmt(score: Optional[float]) -> str:
    return f"{score:.2f}" if score is not None else "—"


def print_coverage_md(
    pairs: list[tuple[str, str, Path]],
    data: dict[tuple[str, str], dict[tuple[str, str], Entry]],
    alpha: str,
    include_pmbt: bool,
) -> None:
    base_mode = "baseline"
    uni_mode = f"uniform_a{alpha}"

    print(f"# Step-25 result coverage (latest-by-mtime per tuple, α={alpha})\n")
    print(f"Scanned {len(pairs)} model × task-vector pairs under `results/25-merge/`.")
    print(f"Columns are the canonical benchmark set; `—` means no `*_score.csv` "
          f"exists for that (model, tv, mode, bench) tuple.\n")

    # ── Coverage matrix (baseline vs uniform_aα) ──
    for label, mode_key in [("Baseline", base_mode), (f"+Uniform α={alpha}", uni_mode)]:
        print(f"## {label}\n")
        header = "| model / tv |" + "".join(f" {b} |" for b in BENCH_ORDER)
        sep = "|---|" + "".join("---:|" for _ in BENCH_ORDER)
        print(header)
        print(sep)
        for model, tv, _ in pairs:
            row = [f"{model} / {tv}"]
            entries = data[(model, tv)]
            for b in BENCH_ORDER:
                e = entries.get((mode_key, b))
                row.append(fmt(e.score) if e else "—")
            print("| " + " | ".join(row) + " |")
        print()

    # ── Other modes observed (PMBT, FT, other sweep α) ──
    if include_pmbt:
        seen_modes = defaultdict(set)  # mode_key -> set of (model, tv)
        for (model, tv, _), _ in zip(pairs, pairs):
            for (mode_key, _bench), e in data[(model, tv)].items():
                if mode_key in (base_mode, uni_mode):
                    continue
                seen_modes[mode_key].add((model, tv))
        if seen_modes:
            print("## Other modes observed on disk\n")
            print("| mode | # (model, tv) pairs | example |")
            print("|---|---:|---|")
            for mk in sorted(seen_modes):
                pair_set = seen_modes[mk]
                example = sorted(pair_set)[0]
                print(f"| `{mk}` | {len(pair_set)} | {example[0]} / {example[1]} |")
            print()


def write_csv(
    pairs: list[tuple[str, str, Path]],
    data: dict[tuple[str, str], dict[tuple[str, str], Entry]],
    out_path: Path,
) -> None:
    fields = ["model", "tv", "mode", "benchmark", "score", "csv_path", "mtime"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for model, tv, _ in pairs:
            for (mode_key, bench), e in sorted(data[(model, tv)].items()):
                w.writerow(dict(
                    model=model, tv=tv, mode=mode_key, benchmark=bench,
                    score=("" if e.score is None else f"{e.score:.4f}"),
                    csv_path=str(e.path),
                    mtime=f"{e.mtime:.0f}",
                ))


# ── Main ───────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        default="/home/projects/bagon/ortalda/mh-neuron/modality_taxonomy/results/25-merge",
        help="results/25-merge root path",
    )
    ap.add_argument("--alpha", default="0.9", help="Uniform α to focus the coverage matrix on")
    ap.add_argument("--include-pmbt", action="store_true",
                    help="Also list PMBT / FT / other-α modes observed")
    ap.add_argument("--csv", default="", help="Optional flat CSV output path")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"error: --root {root} is not a directory")

    pairs = discover_model_tv_pairs(root)
    if not pairs:
        sys.exit(f"error: no model/tv pairs with score CSVs found under {root}")

    data: dict[tuple[str, str], dict[tuple[str, str], Entry]] = {}
    for model, tv, tv_root in pairs:
        data[(model, tv)] = scan_model_tv(tv_root)

    print_coverage_md(pairs, data, alpha=args.alpha, include_pmbt=args.include_pmbt)

    if args.csv:
        write_csv(pairs, data, Path(args.csv))
        print(f"\n_Wrote flat CSV to {args.csv}_", file=sys.stderr)


if __name__ == "__main__":
    main()
