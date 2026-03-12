#!/usr/bin/env python3
"""
patch_pipeline_vtp.py — Patches run_pipeline.sh to add:
  1. llava-ov-si model type (Stage 2 checkpoint for P3)
  2. --step vtp / --step 9 (Visual Token Pressure analysis)

Usage:
    python code/patch_pipeline_vtp.py                    # dry-run (shows diff)
    python code/patch_pipeline_vtp.py --apply             # apply patches in-place

The patches are surgical: each inserts at a specific anchor line so they
won't break if the file has been edited elsewhere.
"""

import argparse
import re
import sys
import os


# ═══════════════════════════════════════════════════════════════════
# Patch definitions — each is (anchor_pattern, insert_position, new_lines)
# anchor_pattern: regex to find the insertion point
# insert_position: 'after' or 'before' the anchor
# new_lines: lines to insert (list of strings)
# ═══════════════════════════════════════════════════════════════════

PATCHES = []

# ────────────────────────────────────────────────────────────────
# PATCH 1: Add VTP_SCRIPT variable in the Defaults section
#   Anchor: the ATTN_SCRIPT= line
#   Insert: after it
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'VTP_SCRIPT variable',
    'anchor': r'^ATTN_SCRIPT="code/attention_analysis\.py"',
    'position': 'after',
    'lines': [
        'VTP_SCRIPT="code/test_visual_token_pressure.py"              # VTP hypothesis analysis for step 9 (vtp)',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 2: Add llava-ov-si to the _VALID_MODELS list
#   Anchor: the _VALID_MODELS= line
#   Replace: add llava-ov-si to the string
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'llava-ov-si in _VALID_MODELS',
    'anchor': r'^_VALID_MODELS="llava-liuhaotian llava-hf llava-ov internvl qwen2vl all"',
    'position': 'replace',
    'lines': [
        '_VALID_MODELS="llava-liuhaotian llava-hf llava-ov llava-ov-si internvl qwen2vl all"',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 3: Add llava-ov-si to ALL_MODELS array
#   Anchor: ALL_MODELS=(llava-liuhaotian llava-hf llava-ov internvl qwen2vl)
#   Replace: add llava-ov-si
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'llava-ov-si in ALL_MODELS',
    'anchor': r'^ALL_MODELS=\(llava-liuhaotian llava-hf llava-ov internvl qwen2vl\)',
    'position': 'replace',
    'lines': [
        'ALL_MODELS=(llava-liuhaotian llava-hf llava-ov llava-ov-si internvl qwen2vl)',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 4: Add llava-ov-si MODEL_PATH default
#   Anchor: the elif for llava-ov MODEL_PATH
#   Insert: after the llava-ov block
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'llava-ov-si MODEL_PATH default',
    'anchor': r'^\s+MODEL_PATH="llava-hf/llava-onevision-qwen2-7b-ov-hf"\s+# LLaVA-OneVision-7B',
    'position': 'after',
    'lines': [
        '    elif [[ "$MODEL_TYPE" == "llava-ov-si" ]]; then',
        '        MODEL_PATH="llava-hf/llava-onevision-qwen2-7b-si-hf"        # LLaVA-OV-7B Stage 2 (single-image)',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 5: Add llava-ov-si MODEL_NAME default
#   Anchor: the elif for llava-ov MODEL_NAME
#   Insert: after it
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'llava-ov-si MODEL_NAME default',
    'anchor': r'^\s+MODEL_NAME="llava-onevision-7b"\s+# LLaVA-OneVision-7B output dir name',
    'position': 'after',
    'lines': [
        '    elif [[ "$MODEL_TYPE" == "llava-ov-si" ]]; then',
        '        MODEL_NAME="llava-onevision-7b-si"                                # LLaVA-OV-7B Stage 2 output dir name',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 6: Add llava-ov-si to N_LAYERS override
#   Anchor: the line that sets N_LAYERS=28 for qwen2vl/llava-ov
#   Replace: add llava-ov-si
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'llava-ov-si N_LAYERS=28',
    'anchor': r'^if \[\[ "\$MODEL_TYPE" == "qwen2vl" \|\| "\$MODEL_TYPE" == "llava-ov" \]\]; then',
    'position': 'replace',
    'lines': [
        'if [[ "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" ]]; then',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 7: Add llava-ov-si SHORT_MODEL alias
#   Anchor: the llava-ov SHORT_MODEL case
#   Insert: after it
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'llava-ov-si SHORT_MODEL',
    'anchor': r"^\s+llava-ov\)\s+SHORT_MODEL=\"lo\"",
    'position': 'after',
    'lines': [
        '    llava-ov-si)      SHORT_MODEL="losi" ;;',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 8: Add llava-ov-si to Python interpreter selection
#   Anchor: the elif for qwen2vl || llava-ov Python
#   Replace: add llava-ov-si
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'llava-ov-si Python interpreter',
    'anchor': r'^elif \[\[ "\$MODEL_TYPE" == "qwen2vl" \|\| "\$MODEL_TYPE" == "llava-ov" \]\]; then',
    'position': 'replace',
    'lines': [
        'elif [[ "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" ]]; then',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 9: Add vtp to --step normalization
#   Anchor: the check_collisions case in step normalization
#   Insert: after it
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': '--step vtp normalization',
    'anchor': r'^\s+check_collisions\|collisions\)\s+STEP="check_collisions"',
    'position': 'after',
    'lines': [
        '    9|vtp)                    STEP="vtp" ;;',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 10: Add vtp to --clean step resolution
#   Anchor: the plot CLEAN_FROM case
#   Insert: after it
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'vtp CLEAN_FROM',
    'anchor': r'^\s+plot\)\s+CLEAN_FROM=8',
    'position': 'after',
    'lines': [
        '        vtp)       CLEAN_FROM=9 ;;',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 11: Add JN9 job name base
#   Anchor: JN8= line
#   Insert: after it
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'JN9 job name',
    'anchor': r'^JN8="8_\$\{SHORT_MODEL\}"',
    'position': 'after',
    'lines': [
        'JN9="9_${SHORT_MODEL}"     # vtp_analysis',
    ],
})

# ────────────────────────────────────────────────────────────────
# PATCH 12: Add llava-ov-si to attention heatmap layer override
#   Anchor: the llava-ov || qwen2vl || internvl heatmap layers check
#   Replace: add llava-ov-si
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'llava-ov-si attention heatmap layers',
    'anchor': r'^\s+if \[\[ "\$MODEL_TYPE" == "llava-ov" \|\| "\$MODEL_TYPE" == "qwen2vl" \|\| "\$MODEL_TYPE" == "internvl" \]\]; then',
    'position': 'replace',
    'lines': [
        '    if [[ "$MODEL_TYPE" == "llava-ov" || "$MODEL_TYPE" == "llava-ov-si" || "$MODEL_TYPE" == "qwen2vl" || "$MODEL_TYPE" == "internvl" ]]; then',
    ],
    'match_first': True,  # only first occurrence
})

# ────────────────────────────────────────────────────────────────
# PATCH 13: Add llava-ov-si to attention --hf_id selection
#   Anchor: the llava-ov --hf_id elif
#   Insert: after it
# ────────────────────────────────────────────────────────────────
PATCHES.append({
    'name': 'llava-ov-si attention --hf_id',
    'anchor': r'^\s+ATTN_COMMON_ARGS="\$ATTN_COMMON_ARGS --hf_id llava-hf/llava-onevision-qwen2-7b-ov-hf"',
    'position': 'after',
    'lines': [
        'elif [[ "$MODEL_TYPE" == "llava-ov-si" ]]; then',
        '    ATTN_COMMON_ARGS="$ATTN_COMMON_ARGS --hf_id llava-hf/llava-onevision-qwen2-7b-si-hf"',
    ],
    'match_first': True,
})


# ────────────────────────────────────────────────────────────────
# PATCH 14: The main VTP step block (step 9)
#   Anchor: just before the Summary section
#   Insert: before the summary
# ────────────────────────────────────────────────────────────────
VTP_STEP_BLOCK = r'''
# ═══════════════════════════════════════════════════════════════
# STEP 9 (vtp): Visual Token Pressure Hypothesis Analysis
# ═══════════════════════════════════════════════════════════════
if [[ "$STEP" == "vtp" || $STEP_ALL == true ]]; then

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  STEP 9: Visual Token Pressure Hypothesis Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
STEP_LOG_DIR="${LOG_DIR}/9-vtp"
mkdir -p "$STEP_LOG_DIR" "$WORK_DIR/$STEP_LOG_DIR"

VTP_OUT="results/9-vtp/${MODE_DIR}"
VTP_MARKER="$VTP_OUT/done.marker"

JOB_NAME="${JN9}"
if [[ -f "$VTP_MARKER" ]] || is_job_active "$JOB_NAME"; then
    echo "  [skip] $JOB_NAME — already done or active"
    SKIPPED=$((SKIPPED + 1))
else

# Build model list from what's available in OUTPUT_DIR
VTP_ARGS="--output_dir $OUTPUT_DIR --out $VTP_OUT --taxonomy pmbt"

# Include Stage 2 checkpoint if llava-ov-si results exist
SI_STATS="$OUTPUT_DIR/llava-onevision-7b-si/llm_permutation/permutation_stats_all.json"
if [[ -f "$SI_STATS" ]]; then
    VTP_ARGS="$VTP_ARGS --include_si"
    echo "  Including llava-ov-si (Stage 2) for P3 comparison"
fi

if $LOCAL; then
    (cd "$WORK_DIR" && python3 $VTP_SCRIPT $VTP_ARGS) \
        2>&1 | tee "${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log"
    touch "$VTP_MARKER"
else
    BSUB_ARGS=(-q "$QUEUE" -J "$JOB_NAME" \
        -oo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
        -eo "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err" \
        -R "rusage[mem=8192]" -M 8192)
    # Wait for merge_nc if running in all mode
    if [[ $STEP_ALL == true ]]; then
        if [[ -n "${MERGE_NC_SUBMITTED:-}" ]]; then
            BSUB_ARGS+=(-w "done($MERGE_NC_SUBMITTED)")
        elif is_job_active "${JN4}"; then
            BSUB_ARGS+=(-w "done(${JN4})")
        fi
    fi
    rm -f "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.log" \
          "$WORK_DIR/${STEP_LOG_DIR}/${JOB_NAME}${LOG_SUFFIX}.err"
    bsub "${BSUB_ARGS[@]}" \
        "cd $WORK_DIR && python3 $VTP_SCRIPT $VTP_ARGS && touch $VTP_MARKER"
    echo "  → Job: $JOB_NAME (CPU only, no GPU needed)"
    SUBMITTED=$((SUBMITTED + 1))
fi

fi  # end skip check

fi  # end step 9 (vtp)
'''

PATCHES.append({
    'name': 'Step 9 VTP block',
    'anchor': r'^# ═+\n# Summary\n# ═+',
    'position': 'before',
    'lines': VTP_STEP_BLOCK.strip().split('\n'),
    'multiline_anchor': True,
})


# ═══════════════════════════════════════════════════════════════════
# Apply patches
# ═══════════════════════════════════════════════════════════════════

def apply_patches(filepath, patches, dry_run=True):
    """Apply all patches to a file.

    Line 1: read original file lines
    Line 2: for each patch, find anchor and insert/replace
    Line 3: write modified file (or print diff in dry-run)
    """
    with open(filepath) as f:
        lines = f.readlines()

    applied = 0
    skipped = 0

    for patch in patches:
        name = patch['name']
        anchor = patch['anchor']
        position = patch['position']
        new_lines = patch['lines']
        match_first = patch.get('match_first', False)
        multiline = patch.get('multiline_anchor', False)

        # Check if patch already applied (look for first new line in file)
        first_new = new_lines[0].rstrip()
        already_applied = any(first_new in line for line in lines)
        if already_applied:
            print(f'  [skip] {name} — already applied')
            skipped += 1
            continue

        if multiline:
            # For multiline anchors, search for the Summary section
            found = False
            for i, line in enumerate(lines):
                if '# Summary' in line and i > 0 and '═' in lines[i-1]:
                    # Insert before the separator line
                    insert_idx = i - 1
                    insert_lines = [l + '\n' for l in new_lines] + ['\n']
                    lines = lines[:insert_idx] + insert_lines + lines[insert_idx:]
                    print(f'  [apply] {name} — inserted at line {insert_idx+1}')
                    applied += 1
                    found = True
                    break
            if not found:
                print(f'  [FAIL] {name} — anchor not found')
            continue

        # Single-line anchor
        found = False
        for i, line in enumerate(lines):
            if re.match(anchor, line.rstrip()):
                if position == 'after':
                    insert_idx = i + 1
                    insert_lines = [l + '\n' for l in new_lines]
                    lines = lines[:insert_idx] + insert_lines + lines[insert_idx:]
                elif position == 'before':
                    insert_idx = i
                    insert_lines = [l + '\n' for l in new_lines]
                    lines = lines[:insert_idx] + insert_lines + lines[insert_idx:]
                elif position == 'replace':
                    lines[i] = new_lines[0] + '\n'
                    if len(new_lines) > 1:
                        extra = [l + '\n' for l in new_lines[1:]]
                        lines = lines[:i+1] + extra + lines[i+1:]
                print(f'  [apply] {name} — line {i+1}')
                applied += 1
                found = True
                if match_first:
                    break
                break
        if not found:
            print(f'  [FAIL] {name} — anchor not found: {anchor[:60]}...')

    if dry_run:
        print(f'\n  DRY RUN: {applied} patches would be applied, {skipped} already present')
        print(f'  Run with --apply to modify {filepath}')
    else:
        with open(filepath, 'w') as f:
            f.writelines(lines)
        print(f'\n  APPLIED: {applied} patches to {filepath} ({skipped} already present)')

    return applied


def main():
    p = argparse.ArgumentParser(
        description='Patch run_pipeline.sh for VTP hypothesis (step 9 + llava-ov-si)')
    p.add_argument('--apply', action='store_true',
                   help='Apply patches (default: dry-run showing what would change)')
    p.add_argument('--pipeline', default='code/run_pipeline.sh',
                   help='Path to run_pipeline.sh')
    args = p.parse_args()

    if not os.path.isfile(args.pipeline):
        print(f'ERROR: {args.pipeline} not found')
        sys.exit(1)

    print(f'Patching {args.pipeline} for VTP hypothesis support...\n')
    n = apply_patches(args.pipeline, PATCHES, dry_run=not args.apply)
    if n == 0 and not args.apply:
        print('\n  All patches already applied — nothing to do.')


if __name__ == '__main__':
    main()
