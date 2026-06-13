"""Extract POPE scores from step-25 merge folders and print per-model tables."""
import csv
from pathlib import Path

MODELS = [
    ("LLaVA-Next-Llama3-8B + dart-math-llama3-8b-prop2diff",
     "llava-next-llama3-8b", "dart-prop"),
    ("Idefics2-8B + MAmmoTH-7B-Mistral",
     "idefics2-8b", "mammoth1"),
    ("Qwen2-VL-7B + Qwen2-Math-7B",
     "qwen2-vl-7b", "qwen2-math"),
]
ROOT = Path("results/25-merge")


def find_pope_csv(config_dir):
    # Find POPE score CSV under the config folder, shortest path wins
    cands = list(config_dir.glob("**/*POPE*_score.csv"))
    if not cands:
        return None
    cands.sort(key=lambda p: (len(p.parts), str(p)))
    return str(cands[0])


def read_pope_overall(csv_path):
    # Parse POPE score CSV and return the Overall-split row as dict
    try:
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                if row.get("split") == "Overall":
                    return {
                        "f1": float(row["Overall"]),
                        "acc": float(row["acc"]),
                        "prec": float(row["precision"]),
                        "rec": float(row["recall"]),
                    }
    except Exception as e:
        print(f"  [warn] failed to read {csv_path}: {e}")
    return None


def config_sort_key(name):
    # Sort order: baseline, uniform_aN (by N), pmbt_tN (by N)
    if name == "baseline":
        return (0, 0, "")
    if name.startswith("uniform_a"):
        try:
            return (1, float(name.split("uniform_a")[1]), "")
        except Exception:
            return (1, 99, name)
    if name.startswith("pmbt_"):
        t = 0.5
        for part in name.split("_"):
            if part.startswith("t") and len(part) > 1:
                try:
                    t = float(part[1:]); break
                except Exception:
                    pass
        return (2, t, name)
    return (3, 0, name)


def extract_for_model(model_dir):
    rows = []
    if not model_dir.is_dir():
        return rows
    for sub in sorted(model_dir.iterdir()):
        if not sub.is_dir():
            continue
        csvp = find_pope_csv(sub)
        if not csvp:
            continue
        parsed = read_pope_overall(csvp)
        if parsed:
            rows.append((sub.name, parsed))
    rows.sort(key=lambda r: config_sort_key(r[0]))
    return rows


def print_table(title, rows):
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    if not rows:
        print("  (no POPE scores found)\n"); return
    print(f"{'Config':35s}  {'F1':>6s}  {'Acc':>6s}  {'Prec':>6s}  {'Recall':>6s}")
    print("-" * 80)
    baseline = next((v for n, v in rows if n == "baseline"), None)
    for name, vals in rows:
        line = (f"{name:35s}  {vals['f1']:6.2f}  {vals['acc']:6.2f}  "
                f"{vals['prec']:6.2f}  {vals['rec']:6.2f}")
        if baseline and name != "baseline":
            df1 = vals["f1"] - baseline["f1"]
            line += f"   (ΔF1={df1:+.2f})"
        print(line)
    print()


def main():
    for title, model, mathllm in MODELS:
        print_table(title, extract_for_model(ROOT / model / mathllm))


if __name__ == "__main__":
    main()
