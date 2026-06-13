#!/usr/bin/env python3
"""Re-run MM-Math scoring on existing generation xlsx files.

The generation (5901 forward passes) is already done, but scoring failed due to
missing antlr4 dependency. Now that antlr4 is installed, we can re-run just the
evaluate step — no GPU, ~5-10 min per config.

Usage:
    python rescore_mmmath.py <xlsx_path> [<xlsx_path> ...]

Must be run with VLMEvalKit_brv's venv:
    modern_vlms/VLMEvalKit_brv/.venv/bin/python code/rescore_mmmath.py <path>
"""
import sys
import os


def rescore(xlsx_path):
    """Re-score a single MM-Math generation xlsx."""
    from vlmeval.dataset import build_dataset
    from vlmeval.smp import load, dump

    if not os.path.isfile(xlsx_path):
        print(f"  [skip] File not found: {xlsx_path}")
        return None

    print(f"\n[rescore] {xlsx_path}")

    # Build MM-Math dataset
    dataset = build_dataset('MM-Math')

    # Run evaluate — writes _score.csv next to the xlsx
    try:
        # MM-Math's evaluate() uses AutoScoringJudge (now works with antlr4)
        score_file = dataset.evaluate(xlsx_path, nproc=4)
        print(f"  [OK] Score file: {score_file}")
        return score_file
    except Exception as e:
        print(f"  [FAIL] {type(e).__name__}: {e}")
        return None


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    for xlsx in sys.argv[1:]:
        rescore(xlsx)
