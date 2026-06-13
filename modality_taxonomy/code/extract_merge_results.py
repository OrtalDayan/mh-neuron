#!/usr/bin/env python3
"""Extract step 25 weight merge results — handles MMStar_acc.csv and DynaMath scale.

Patched to group configs by family in the main table (Uniform / PMBT full /
Down_proj only / Other scope variants) and to add a DOWN_PROJ head-to-head
section comparing each pmbt_*_mlponly_a1.0_o1.0_downproj config against its
matching full-PMBT triplet.
"""
import csv, os, glob, sys, re


def find_csv(d, patterns):
    """Find first matching file (CSV or JSON), excluding timestamped subdirs for CSVs only."""
    if isinstance(patterns, str):
        patterns = [patterns]
    for pattern in patterns:
        hits = glob.glob(os.path.join(d, '**', pattern), recursive=True)
        if not pattern.endswith('.json'):
            hits = [h for h in hits if '/T2' not in h]
        if hits:
            return hits[0]
    return None


def get_score(d, bench):
    if bench == 'MathVista':
        f = find_csv(d, '*MathVista_MINI*_score.csv')
        if not f: return None
        with open(f) as fh:
            for r in csv.reader(fh):
                if r and r[0] == 'Overall': return float(r[-1])
    elif bench == 'MathVision':
        f = find_csv(d, '*MathVision_MINI*_score.csv')
        if not f: return None
        with open(f) as fh:
            for r in csv.reader(fh):
                if r and r[0] == 'Overall': return float(r[-1])
    elif bench == 'MMStar':
        f = find_csv(d, ['*MMStar*_acc.csv', '*MMStar*_score.csv'])
        if not f: return None
        with open(f) as fh:
            rows = list(csv.reader(fh))
            if len(rows) < 2: return None
            header = rows[0]
            try:
                col = header.index('Overall')
            except ValueError:
                try:
                    col = header.index('avg')
                except ValueError:
                    return None
            try:
                v = float(rows[1][col])
                return v * 100 if v < 1.5 else v
            except (ValueError, IndexError):
                return None
    elif bench.startswith('MVe_'):
        suffix = bench[len('MVe_'):]
        f = find_csv(d, f'*MathVerse_MINI_{suffix}*_score.csv')
        if not f: return None
        with open(f) as fh:
            rows = list(csv.reader(fh))
            if len(rows) < 2: return None
            # MathVerse CSV: header = "accuracy,correct,total,..."; row 1 col 0 is the accuracy
            # (no "Overall" row exists)
            try:
                return float(rows[1][0])
            except (ValueError, IndexError):
                return None
    elif bench == 'DynaMath':
        # Newer runs write a CSV; legacy runs wrote JSON. Try CSV first.
        f = find_csv(d, '*DynaMath*_score.csv')
        if f:
            with open(f) as fh:
                rows = list(csv.reader(fh))
                if len(rows) < 2: return None
                # Header: Setting, Overall, Subject-..., Level-...
                # Row 1: "Average", <overall>, ...
                # Row 2: "Worst Case", <overall>, ...
                try:
                    overall_col = rows[0].index('Overall')
                except ValueError:
                    return None
                # Use the "Worst Case" row if present (matches legacy 'worst_case_accuracy')
                # fallback to "Average" if Worst Case missing
                target_row = None
                for r in rows[1:]:
                    if r and r[0] == 'Worst Case':
                        target_row = r
                        break
                if target_row is None:
                    target_row = rows[1]  # Average
                try:
                    v = float(target_row[overall_col])
                    return v * 100 if v < 1.5 else v
                except (ValueError, IndexError):
                    return None
        # Legacy JSON fallback
        f = find_csv(d, '*DynaMath*.json')
        if not f: return None
        try:
            import json
            with open(f) as fh:
                data = json.load(fh)
            worst = data.get('worst_case_accuracy')
            if worst is None:
                return None
            return worst * 100 if worst < 1.5 else worst
        except Exception:
            return None
    elif bench == 'MM-Math':
        f = find_csv(d, ['*MM-Math*_score.json', '*MM_Math*_score.json', '*MM-Math*.json'])
        if not f: return None
        try:
            import json
            with open(f) as fh:
                data = json.load(fh)
            v = data.get('Overall') or data.get('overall') or data.get('accuracy')
            if v is None:
                return None
            return v * 100 if v < 1.5 else v
        except Exception:
            return None
    elif bench == 'TriviaQA':
        f = find_csv(d, '*TriviaQA*_score.csv')
        if not f: return None
        with open(f) as fh:
            rows = list(csv.reader(fh))
            if len(rows) < 2: return None
            try:
                col = rows[0].index('accuracy')
            except ValueError:
                try:
                    col = rows[0].index('Overall')
                except ValueError:
                    return None
            try:
                v = float(rows[1][col])
                return v * 100 if v < 1.5 else v
            except (ValueError, IndexError):
                return None
    elif bench.startswith('POPE_'):
        # POPE CSV has splits as rows and metrics as columns:
        #   header: split, Overall, acc, precision, recall
        #   row "Overall":     F1-equivalent overall + per-metric averages
        #   row "random":      Overall = F1 on random split
        #   row "popular":     Overall = F1 on popular split
        #   row "adversarial": Overall = F1 on adversarial split
        sub = bench[len('POPE_'):]
        # Map extract column -> (row label, csv column)
        # POPE_F1 = the "Overall" F1 (Overall row, Overall col)
        # POPE_Adv/Pop/Rnd = F1 on those splits (that row, Overall col)
        # POPE_Acc = Overall row, 'acc' column
        # POPE_Prec = Overall row, 'precision' column
        row_col = {
            'F1':   ('Overall', 'Overall'),
            'Adv':  ('adversarial', 'Overall'),
            'Pop':  ('popular', 'Overall'),
            'Rnd':  ('random', 'Overall'),
            'Acc':  ('Overall', 'acc'),
            'Prec': ('Overall', 'precision'),
        }
        target = row_col.get(sub)
        if target is None: return None
        row_name, col_name = target
        f = find_csv(d, '*POPE*_score.csv')
        if not f: return None
        with open(f) as fh:
            rows = list(csv.reader(fh))
            if len(rows) < 2: return None
            try:
                col_idx = rows[0].index(col_name)
            except ValueError:
                return None
            for r in rows[1:]:
                if r and r[0] == row_name:
                    try:
                        v = float(r[col_idx])
                        return v * 100 if v < 1.5 else v
                    except (ValueError, IndexError):
                        return None
            return None
    elif bench.startswith('MME'):
        col_map = {'MME_P': 'perception', 'MME_R': 'reasoning'}
        col_name = col_map.get(bench)
        if col_name is None: return None
        f = find_csv(d, '*MME_score.csv')
        if not f: return None
        with open(f) as fh:
            rows = list(csv.reader(fh))
            if len(rows) < 2: return None
            try:
                col_idx = rows[0].index(col_name)
                return float(rows[1][col_idx])
            except (ValueError, IndexError):
                return None
    return None


benchmarks = ['MathVista', 'MVe_Text_Dominant', 'MVe_Text_Lite', 'MVe_Vision_Intensive',
              'MVe_Vision_Dominant', 'MVe_Vision_Only', 'MMStar', 'DynaMath', 'MathVision', 'MM-Math',
              'POPE_F1', 'POPE_Adv', 'POPE_Pop', 'POPE_Rnd',
              'POPE_Acc', 'POPE_Prec', 'TriviaQA', 'MME_P', 'MME_R']
short = ['MV', 'TD', 'TL', 'VI', 'VD', 'VO', 'MMS', 'DM', 'MVis', 'MMM',
         'PPE-F1', 'PPE-Ad', 'PPE-Po', 'PPE-Rn',
         'PPE-Ac', 'PPE-Pr', 'TQA', 'MME-P', 'MME-R']

# ═══════════════════════════════════════════════════════════════════
# Config family classification
# ═══════════════════════════════════════════════════════════════════
DOWNPROJ_SUFFIX = '_mlponly_a1.0_o1.0_downproj'  # pure down_proj isolation


def _classify(subdir):
    """Return family string for grouping."""
    if subdir in ('baseline', 'eval_baseline'):
        return 'baseline'
    if subdir.startswith('uniform_') or subdir.startswith('eval_uniform_'):
        return 'uniform'
    if not subdir.startswith('pmbt_'):
        return 'other'
    if subdir.endswith(DOWNPROJ_SUFFIX):
        return 'downproj'
    # Scope filtering markers — anything with these is genuine non-vanilla scope.
    # NOTE: _a<x>_o<x> (alpha-tuning) and _p<x> (strict-p) are tuning variants
    # of full PMBT, NOT scope-filtered, so they should remain pmbt_full.
    if any(s in subdir for s in ('_mlponly', '_attnonly', '_downproj', '_gateupproj',
                                  '_gateproj', '_upproj', '_updownproj')):
        return 'other_scope'
    return 'pmbt_full'


def _triplet_base(subdir):
    """Extract pmbt_t<T>_v<V>_m<M> base, dropping any suffix."""
    m = re.match(r'(pmbt_t[\d.]+_v[\d.]+_m[\d.]+)', subdir)
    return m.group(1) if m else None


def print_table(title, base):
    _w = 32 + 8 * len(short)
    print(f"\n{'='*_w}")
    print(f"  {title}")
    print('=' * _w)
    header = f'{"Config":<32s}' + ''.join(f'{s:>8s}' for s in short)
    print(header)
    print('-' * len(header))
    if not os.path.isdir(base):
        print("  (no results)")
        return
    configs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

    baseline_dirs = [d for d in configs if d in ('baseline', 'eval_baseline')]
    uniform09_dirs = [d for d in configs if d in ('uniform_a0.9', 'eval_uniform_0.9')]

    def _score_from_dirs(dirs, b):
        for subdir in dirs:
            d_eval = os.path.join(base, subdir, 'eval')
            d = d_eval if os.path.isdir(d_eval) else os.path.join(base, subdir)
            v = get_score(d, b)
            if v is not None:
                return v
        return None

    # Baseline row (combined)
    if baseline_dirs:
        row = f'{"Baseline":<32s}'
        has_any = False
        for b in benchmarks:
            v = _score_from_dirs(baseline_dirs, b)
            if v is not None: has_any = True
            row += f'{v:>8.1f}' if v is not None else f'{"—":>8s}'
        if has_any:
            print(row)

    # Group remaining configs by family for readable output
    groups = {'uniform': [], 'pmbt_full': [], 'downproj': [], 'other_scope': [], 'other': []}
    for subdir in configs:
        if subdir in ('baseline', 'eval_baseline', 'eval_uniform_0.9'):
            continue
        fam = _classify(subdir)
        groups.setdefault(fam, []).append(subdir)

    def _print_group(label, subdirs):
        if not subdirs:
            return
        printed_header = False
        for subdir in subdirs:
            if subdir == 'uniform_a0.9' and uniform09_dirs:
                dirs_to_check = uniform09_dirs
                disp_name = 'uniform_a0.9'
            else:
                dirs_to_check = [subdir]
                disp_name = subdir
            row = f'{disp_name:<32s}'
            has_any = False
            for b in benchmarks:
                v = _score_from_dirs(dirs_to_check, b)
                if v is not None: has_any = True
                row += f'{v:>8.1f}' if v is not None else f'{"—":>8s}'
            if has_any:
                if not printed_header:
                    print(f'  --- {label} ---')
                    printed_header = True
                print(row)

    _print_group('Uniform', groups['uniform'])
    _print_group('PMBT (full merge)', groups['pmbt_full'])
    _print_group('Down_proj only (scope=mlp, projs=down, α=1.0, α_other=1.0)', groups['downproj'])
    _print_group('Other scope variants', groups['other_scope'])
    if groups.get('other'):
        _print_group('Other', groups['other'])


def _math_mean(row_dict):
    math_benches = ['MathVista', 'MVe_Text_Dominant', 'MVe_Text_Lite', 'MVe_Vision_Intensive',
                    'MVe_Vision_Dominant', 'MVe_Vision_Only', 'MMStar', 'DynaMath',
                    'MathVision', 'MM-Math']
    vals = [row_dict.get(b) for b in math_benches]
    if any(v is None for v in vals):
        return None
    return sum(vals) / len(vals)


def _math_mean_partial(row_dict):
    """Math mean over whichever math benches ARE present. Useful for early preview."""
    math_benches = ['MathVista', 'MVe_Text_Dominant', 'MVe_Text_Lite', 'MVe_Vision_Intensive',
                    'MVe_Vision_Dominant', 'MVe_Vision_Only', 'MMStar', 'DynaMath',
                    'MathVision', 'MM-Math']
    vals = [row_dict.get(b) for b in math_benches if row_dict.get(b) is not None]
    if len(vals) < 5:
        return None
    return sum(vals) / len(vals)


def _collect_row(base, subdirs):
    out = {}
    for b in benchmarks:
        v = None
        for sd in subdirs:
            d_eval = os.path.join(base, sd, 'eval')
            d = d_eval if os.path.isdir(d_eval) else os.path.join(base, sd)
            v = get_score(d, b)
            if v is not None:
                break
        out[b] = v
    return out


def print_comparison(title, base):
    """Summary: baseline vs best PMBT vs best uniform, with delta rows."""
    if not os.path.isdir(base):
        return
    configs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    baseline_dirs = [d for d in configs if d in ('baseline', 'eval_baseline')]
    uniform09_dirs = [d for d in configs if d in ('uniform_a0.9', 'eval_uniform_0.9')]

    scored = {}
    if baseline_dirs:
        r = _collect_row(base, baseline_dirs)
        r['__mm__'] = _math_mean(r)
        scored['baseline'] = r
    for subdir in configs:
        if subdir in ('baseline', 'eval_baseline', 'eval_uniform_0.9'):
            continue
        if subdir == 'uniform_a0.9' and uniform09_dirs:
            r = _collect_row(base, uniform09_dirs)
        else:
            r = _collect_row(base, [subdir])
        r['__mm__'] = _math_mean(r)
        scored[subdir] = r

    # Best PMBT — restricted to "full" family so that down_proj variants don't
    # hijack the headline comparison
    def _best_full_pmbt():
        cands = []
        for n, r in scored.items():
            if r['__mm__'] is None:
                continue
            if _classify(n) == 'pmbt_full':
                cands.append((n, r))
        if not cands:
            return None, None
        name, r = max(cands, key=lambda x: x[1]['__mm__'])
        return name, r

    def _best_uniform():
        cands = [(n, r) for n, r in scored.items()
                 if n.startswith('uniform_') and r['__mm__'] is not None]
        if not cands:
            return None, None
        name, r = max(cands, key=lambda x: x[1]['__mm__'])
        return name, r

    bl = scored.get('baseline')
    pmbt_name, pmbt = _best_full_pmbt()
    uni_name, uni = _best_uniform()

    if bl is None or pmbt is None or uni is None:
        print(f"\n  (comparison skipped — missing baseline/PMBT/uniform rows for {title})")
        return

    _w = 32 + 8 * (len(short) + 1)
    print()
    print("=" * _w)
    print(f"  {title} — HEAD-TO-HEAD: best PMBT vs best uniform (by math_mean)")
    print("=" * _w)
    print(f"    Best PMBT:    {pmbt_name}")
    print(f"    Best Uniform: {uni_name}")
    print("-" * _w)

    header = f'{"Row":<32s}' + ''.join(f'{s:>8s}' for s in short) + f'{"MathMn":>8s}'
    print(header)
    print('-' * _w)

    def _fmt(v, signed=False):
        if v is None:
            return f'{"—":>8s}'
        return f'{v:>+8.2f}' if signed else f'{v:>8.1f}'

    def _row(label, r, signed=False):
        line = f'{label:<32s}'
        for b in benchmarks:
            line += _fmt(r.get(b), signed)
        line += _fmt(r.get('__mm__'), signed)
        print(line)

    _row("Baseline",         bl)
    _row("Best PMBT",        pmbt)
    _row("Best Uniform",     uni)
    print('-' * _w)

    def _sub(a, b):
        out = {}
        for k in list(benchmarks) + ['__mm__']:
            va, vb = a.get(k), b.get(k)
            out[k] = (va - vb) if (va is not None and vb is not None) else None
        return out

    _row("Δ PMBT − Baseline",     _sub(pmbt, bl), signed=True)
    _row("Δ Uniform − Baseline",  _sub(uni, bl), signed=True)
    _row("Δ PMBT − Uniform (KEY)", _sub(pmbt, uni), signed=True)

    pmbt_vs_bl = _sub(pmbt, bl)
    uni_vs_bl = _sub(uni, bl)

    def _ratio_fmt(num, den):
        if num is None or den is None:
            return f'{"—":>8s}'
        if abs(den) < 0.05:
            if abs(num) < 0.05:
                return f'{"~tied":>8s}'
            return f'{"∞":>8s}'
        r = num / den
        if abs(r) >= 100:
            return f'{"∞":>8s}'
        return f'{r:>+7.1f}x'

    ratio_line = f'{"Gain ratio (PMBT/Uniform)":<32s}'
    for b in benchmarks:
        ratio_line += _ratio_fmt(pmbt_vs_bl.get(b), uni_vs_bl.get(b))
    ratio_line += _ratio_fmt(pmbt_vs_bl.get('__mm__'), uni_vs_bl.get('__mm__'))
    print(ratio_line)


# ═══════════════════════════════════════════════════════════════════
# NEW: down_proj-only head-to-head
# ═══════════════════════════════════════════════════════════════════

def print_downproj_comparison(title, base):
    """For each pmbt_*_mlponly_a1.0_o1.0_downproj config, find the matching full-PMBT
    config at the same (t, v, m) triplet and print side-by-side deltas.
    """
    if not os.path.isdir(base):
        return
    configs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

    pairs = []  # (triplet_str, downproj_name, fullpmbt_name_or_None)
    for subdir in configs:
        if not subdir.endswith(DOWNPROJ_SUFFIX):
            continue
        dp_base = subdir[:-len(DOWNPROJ_SUFFIX)]  # e.g. "pmbt_t0.7_v1.0_m1.0"
        match = None
        # Prefer exact vanilla triplet; then try with _o1.0 (text-only sweep variant)
        if dp_base in configs:
            match = dp_base
        elif f'{dp_base}_o1.0' in configs:
            match = f'{dp_base}_o1.0'
        triplet = dp_base[len('pmbt_'):]  # "t0.7_v1.0_m1.0"
        pairs.append((triplet, subdir, match))

    if not pairs:
        return

    _w = 34 + 8 * (len(short) + 1)
    print()
    print("=" * _w)
    print(f"  {title} — DOWN_PROJ HEAD-TO-HEAD (down_proj only vs full PMBT at same triplet)")
    print("=" * _w)
    header = f'{"Triplet / Row":<34s}' + ''.join(f'{s:>8s}' for s in short) + f'{"MathMn":>8s}'
    print(header)
    print('-' * _w)

    def _fmt(v, signed=False):
        if v is None:
            return f'{"—":>8s}'
        return f'{v:>+8.2f}' if signed else f'{v:>8.1f}'

    score_cache = {}
    def _scores_for(cfg):
        if cfg not in score_cache:
            r = _collect_row(base, [cfg])
            r['__mm__'] = _math_mean(r)
            r['__mm_partial__'] = _math_mean_partial(r)
            score_cache[cfg] = r
        return score_cache[cfg]

    deltas_summary = []

    for triplet, dp_name, full_name in sorted(pairs):
        dp_r = _scores_for(dp_name)
        print(f'  {triplet}')
        # Down_proj row
        line = f'{"    down_proj only":<34s}'
        for b in benchmarks:
            line += _fmt(dp_r.get(b))
        mm = dp_r.get('__mm__') or dp_r.get('__mm_partial__')
        line += _fmt(mm)
        print(line)

        if full_name:
            full_r = _scores_for(full_name)
            line = f'{"    full PMBT":<34s}'
            for b in benchmarks:
                line += _fmt(full_r.get(b))
            mm_f = full_r.get('__mm__') or full_r.get('__mm_partial__')
            line += _fmt(mm_f)
            print(line)

            # Delta row (down_proj − full)
            delta = {}
            for k in list(benchmarks) + ['__mm__']:
                va = dp_r.get(k) if k != '__mm__' else (dp_r.get('__mm__') or dp_r.get('__mm_partial__'))
                vb = full_r.get(k) if k != '__mm__' else (full_r.get('__mm__') or full_r.get('__mm_partial__'))
                delta[k] = (va - vb) if (va is not None and vb is not None) else None

            line = f'{"    Δ (down_proj − full)":<34s}'
            for b in benchmarks:
                line += _fmt(delta.get(b), signed=True)
            line += _fmt(delta.get('__mm__'), signed=True)
            print(line)

            if delta.get('__mm__') is not None:
                deltas_summary.append((triplet, delta['__mm__']))
        else:
            print(f'    (no matching full-PMBT config for {triplet})')
        print()

    if deltas_summary:
        print('-' * _w)
        pos = sum(1 for _, d in deltas_summary if d > 0.01)
        neg = sum(1 for _, d in deltas_summary if d < -0.01)
        ties = len(deltas_summary) - pos - neg
        mean = sum(d for _, d in deltas_summary) / len(deltas_summary)
        print(f'  SUMMARY ({len(deltas_summary)} matched pairs on MathMn):')
        print(f'    down_proj > full PMBT:  {pos} triplets')
        print(f'    tied (|Δ| ≤ 0.01):      {ties} triplets')
        print(f'    down_proj < full PMBT:  {neg} triplets')
        print(f'    mean Δ MathMn:          {mean:+.3f}')


if __name__ == '__main__':
    MODELS = [
        ("LLaVA-Next-LLaMA3-8B + Dart-Math-Prop",
         'results/25-merge/llava-next-llama3-8b/dart-prop'),
        ("Idefics2-8B + MAmmoTH-7B-Mistral",
         'results/25-merge/idefics2-8b/mammoth1'),
        ("Qwen2-VL-7B + Qwen2-Math-7B",
         'results/25-merge/qwen2-vl-7b/qwen2-math'),
    ]
    for title, base in MODELS:
        print_table(title, base)
        print_comparison(title, base)
        print_downproj_comparison(title, base)