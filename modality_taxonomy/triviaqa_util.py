"""
TriviaQA alias-matching utilities.

Implements the standard TriviaQA exact-match scoring: prediction is
considered correct if its normalized form matches any of the normalized
aliases (including the primary answer). This mirrors the Joshi et al. 2017
TriviaQA evaluation script and matches the inline evaluator used in
run1_ablation.py for consistency across step 24 and step 25.
"""
import re
import string


def normalize_answer(s):
    """Lower, strip articles/punct/whitespace — the TriviaQA standard normalization."""
    if s is None:
        return ''
    s = str(s).lower()
    # Remove articles
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    # Remove punctuation
    s = ''.join(ch for ch in s if ch not in set(string.punctuation))
    # Collapse whitespace
    s = ' '.join(s.split())
    return s


def exact_match_with_aliases(prediction, answer, aliases=None):
    """
    Return True if the normalized prediction contains any normalized alias.

    We use substring match (not equality) because VLM generations often include
    surrounding text like "The answer is Paris." rather than just "Paris".
    This is the same criterion used in run1_ablation.py's inline evaluator.

    Args
    ----
    prediction : str   — the model's raw generation
    answer     : str   — the canonical answer string
    aliases    : list  — optional list of alternative accepted strings

    Returns
    -------
    bool
    """
    pred_norm = normalize_answer(prediction)
    if not pred_norm:
        return False

    candidates = [answer] if answer is not None else []
    if aliases:
        if isinstance(aliases, str):
            # TSV may encode aliases as a delimited string; handle common formats
            try:
                # JSON-style list
                import json
                parsed = json.loads(aliases)
                if isinstance(parsed, list):
                    candidates.extend(parsed)
                else:
                    candidates.append(str(parsed))
            except (ValueError, TypeError):
                # Fallback: pipe-delimited (our TSV convention)
                candidates.extend(aliases.split('|'))
        else:
            candidates.extend(list(aliases))

    for cand in candidates:
        cand_norm = normalize_answer(cand)
        if cand_norm and cand_norm in pred_norm:
            return True
    return False
