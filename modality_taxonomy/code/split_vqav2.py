#!/usr/bin/env python3
"""Split VQAv2 questions into visual-perception vs knowledge-reasoning.

Visual-Perception: answer is directly observable in pixels.
  - "What color is the car?"  → look at the car, see red
  - "How many people are there?" → count visible people
  - "Is the door open?" → look at the door

Knowledge-Reasoning: answer requires world knowledge or inference.
  - "What sport is being played?" → see field + ball, know it's soccer
  - "What breed is this dog?" → see dog, know it's a labrador
  - "What room is this?" → see furniture, infer kitchen

Usage:
    python split_vqav2.py \
        --questions data/VQAv2/v2_OpenEnded_mscoco_val2014_questions.json \
        --annotations data/VQAv2/v2_mscoco_val2014_annotations.json \
        --output_dir data/VQAv2/split \
        --num_per_split 500
"""

import argparse                                                             # Line 1: CLI parsing
import json                                                                 # Line 2: read/write JSON
import os                                                                   # Line 3: directory ops
import re                                                                   # Line 4: regex for pattern matching
import numpy as np                                                          # Line 5: random sampling


# ═══════════════════════════════════════════════════════════
# Classification rules
# ═══════════════════════════════════════════════════════════

# Line 6: Patterns that strongly indicate VISUAL-PERCEPTION questions.
# These ask about directly observable attributes — color, count, position,
# presence, shape, size, visible state.
PERCEPTION_PATTERNS = [
    r'^what colou?r ',                                                      # Line 7: "what color is..."
    r'^how many ',                                                          # Line 8: counting visible objects
    r'^is there (a|an) ',                                                   # Line 9: object presence
    r'^are there (any )?',                                                  # Line 10: plural presence
    r'^is (the |this |that )',                                              # Line 11: binary attribute ("is the sky blue")
    r'^are (the |these |those )',                                           # Line 12: plural binary attribute
    r'^do you see ',                                                        # Line 13: visual presence
    r'^can you see ',                                                       # Line 14: visual presence
    r'^what is (the |this )?(person|man|woman|boy|girl|child) (doing|wearing|holding|eating|riding|carrying)', # Line 15: visible actions
    r'^what is on (the|top of) ',                                           # Line 16: spatial/visible
    r'^what is in (the |front|back)',                                       # Line 17: spatial
    r'^what is (under|below|above|behind|next to|near) ',                   # Line 18: spatial relationships
    r'^where (is|are) (the|this|that) ',                                    # Line 19: spatial location
    r'^which (direction|way|side) ',                                        # Line 20: spatial direction
    r'^what (shape|pattern) ',                                              # Line 21: visual attribute
    r'^(does|do) (the |this |that )?(person|man|woman|boy|girl).*(wear|have|hold|carry)', # Line 22: visible attributes
    r'^what number ',                                                       # Line 23: reading visible numbers
    r'^what letter',                                                        # Line 24: reading visible letters
    r'^what time ',                                                         # Line 25: reading a clock (perception)
    r'^what does the sign say',                                             # Line 26: reading visible text
]

# Line 27: Patterns that strongly indicate KNOWLEDGE-REASONING questions.
# These require world knowledge, categorization, or inference beyond
# what is directly visible.
KNOWLEDGE_PATTERNS = [
    r'^what (type|kind|breed|brand|make|model|species|flavor|genre) ',       # Line 28: categorization
    r'^what sport ',                                                         # Line 29: sport identification
    r'^what (sport|game) is ',                                              # Line 30: sport identification
    r'^what animal ',                                                        # Line 31: species identification
    r'^what food ',                                                          # Line 32: food identification
    r'^what fruit ',                                                         # Line 33: fruit identification
    r'^what vegetable ',                                                     # Line 34: vegetable identification
    r'^what vehicle ',                                                       # Line 35: vehicle identification
    r'^what instrument ',                                                    # Line 36: instrument identification
    r'^what (room|place|location|country|city|state) ',                     # Line 37: location inference
    r'^what season ',                                                        # Line 38: temporal inference
    r'^what holiday ',                                                       # Line 39: cultural knowledge
    r'^what (is|are) (the |this |that |these )?(name|title) ',              # Line 40: naming/identification
    r'^why ',                                                                # Line 41: causal reasoning
    r'^what (is|are) .* (used for|made of|called)',                         # Line 42: function/material knowledge
    r'^what (is|are) .* (doing|about to|going to) ',                        # Line 43: intent inference (ambiguous)
    r'^(who|whose) ',                                                        # Line 44: identity (needs knowledge)
    r'^what (year|decade|era|century) ',                                    # Line 45: temporal knowledge
    r'^what (material|fabric) ',                                            # Line 46: material identification
    r'^what (company|team|organization) ',                                  # Line 47: entity identification
    r'^is this (a |an )',                                                    # Line 48: categorization ("is this a kitchen?")
]

# Line 49: Compile patterns for efficiency
PERCEPTION_RE = [re.compile(p, re.IGNORECASE) for p in PERCEPTION_PATTERNS]
KNOWLEDGE_RE = [re.compile(p, re.IGNORECASE) for p in KNOWLEDGE_PATTERNS]


def classify_question(question):
    """Classify a VQAv2 question as 'perception' or 'knowledge'.

    Line 50: First checks perception patterns, then knowledge patterns.
    Line 51: Questions matching neither are classified by answer type heuristic.

    Args:
        question: question string

    Returns:
        'perception', 'knowledge', or 'ambiguous'
    """
    q = question.strip()

    # Line 52: Check perception patterns first
    for pat in PERCEPTION_RE:
        if pat.search(q):
            return 'perception'

    # Line 53: Check knowledge patterns
    for pat in KNOWLEDGE_RE:
        if pat.search(q):
            return 'knowledge'

    # Line 54: Fallback heuristics for unmatched questions
    q_lower = q.lower()

    # Line 55: Yes/no questions about visual attributes tend to be perception
    if q_lower.startswith(('is ', 'are ', 'does ', 'do ', 'has ', 'have ', 'can ')):
        # Line 56: But "is this a kitchen?" is knowledge, "is the light on?" is perception
        # Default yes/no to perception (majority in VQAv2 are about visible state)
        return 'perception'

    # Line 57: "What is..." is ambiguous — could be "what is this?" (knowledge)
    # or "what is the color?" (perception). Default to knowledge since
    # "what is this/that" requires identification.
    if q_lower.startswith('what is') or q_lower.startswith('what are'):
        return 'knowledge'

    # Line 58: Everything else is ambiguous
    return 'ambiguous'


def classify_by_answer_type(item, annotations_lookup):
    """Secondary classifier: use answer characteristics.

    Line 59: Color words as answers → perception.
    Line 60: Numbers as answers → likely perception (counting).
    Line 61: Common nouns / proper nouns → likely knowledge.

    Args:
        item: dict with question_id, question, answers
        annotations_lookup: dict mapping question_id to annotation dict

    Returns:
        'perception' or 'knowledge'
    """
    answers = item.get('answers', [])
    if not answers:
        return 'ambiguous'

    # Line 62: Get most common answer
    from collections import Counter
    answer_counts = Counter(a.strip().lower() for a in answers)
    top_answer = answer_counts.most_common(1)[0][0]

    # Line 63: Color answers → perception
    colors = {'red', 'blue', 'green', 'yellow', 'white', 'black', 'brown',
              'orange', 'pink', 'purple', 'gray', 'grey', 'tan', 'beige',
              'gold', 'silver', 'maroon', 'navy', 'teal', 'cream'}
    if top_answer in colors:
        return 'perception'

    # Line 64: Pure number answers → perception (counting)
    if top_answer.isdigit():
        return 'perception'

    # Line 65: Yes/no → likely perception
    if top_answer in ('yes', 'no'):
        return 'perception'

    return 'knowledge'


def main():
    parser = argparse.ArgumentParser(description='Split VQAv2 into perception vs knowledge')
    parser.add_argument('--questions', required=True,                       # Line 66: VQAv2 questions file
                        help='Path to v2_OpenEnded_mscoco_val2014_questions.json')
    parser.add_argument('--annotations', required=True,                     # Line 67: VQAv2 annotations file
                        help='Path to v2_mscoco_val2014_annotations.json')
    parser.add_argument('--output_dir', required=True,                      # Line 68: output directory
                        help='Directory to save split question files')
    parser.add_argument('--num_per_split', type=int, default=500,           # Line 69: questions per split
                        help='Number of questions per split (default: 500)')
    parser.add_argument('--seed', type=int, default=42,                     # Line 70: random seed
                        help='Random seed for sampling')
    args = parser.parse_args()

    # ── Load data ──
    print(f'Loading questions from {args.questions} ...')
    with open(args.questions) as f:                                         # Line 71: load questions
        q_data = json.load(f)
    with open(args.annotations) as f:                                       # Line 72: load annotations
        a_data = json.load(f)

    # Line 73: Build annotation lookup
    ann_lookup = {}
    for ann in a_data['annotations']:
        answers = [a['answer'] for a in ann['answers']]
        ann_lookup[ann['question_id']] = {
            'answers': answers,
            'answer_type': ann.get('answer_type', ''),
            'question_type': ann.get('question_type', ''),
        }

    # ── Classify each question ──
    perception_items = []                                                   # Line 74: accumulate perception Qs
    knowledge_items = []                                                    # Line 75: accumulate knowledge Qs
    ambiguous_items = []                                                    # Line 76: accumulate ambiguous Qs

    for q in q_data['questions']:                                           # Line 77: iterate all questions
        qid = q['question_id']
        if qid not in ann_lookup:
            continue

        item = {
            'question_id': qid,
            'image_id': q['image_id'],
            'question': q['question'],
            'answers': ann_lookup[qid]['answers'],
            'answer_type': ann_lookup[qid]['answer_type'],
            'question_type': ann_lookup[qid]['question_type'],
        }

        label = classify_question(q['question'])                            # Line 78: primary classifier

        if label == 'ambiguous':                                            # Line 79: try answer-based fallback
            label = classify_by_answer_type(item, ann_lookup)

        if label == 'perception':
            perception_items.append(item)
        elif label == 'knowledge':
            knowledge_items.append(item)
        else:
            ambiguous_items.append(item)

    # ── Print statistics ──
    total = len(perception_items) + len(knowledge_items) + len(ambiguous_items)
    print(f'\nClassification results ({total} total):')
    print(f'  Perception:  {len(perception_items):>6d} ({100*len(perception_items)/total:.1f}%)')
    print(f'  Knowledge:   {len(knowledge_items):>6d} ({100*len(knowledge_items)/total:.1f}%)')
    print(f'  Ambiguous:   {len(ambiguous_items):>6d} ({100*len(ambiguous_items)/total:.1f}%)')

    # Line 80: Print example questions from each category
    print(f'\n  Example PERCEPTION questions:')
    rng = np.random.RandomState(args.seed)
    for item in rng.choice(perception_items, min(5, len(perception_items)), replace=False):
        top_ans = max(set(item['answers']), key=item['answers'].count)
        print(f'    Q: {item["question"]:<50s}  A: {top_ans}')

    print(f'\n  Example KNOWLEDGE questions:')
    for item in rng.choice(knowledge_items, min(5, len(knowledge_items)), replace=False):
        top_ans = max(set(item['answers']), key=item['answers'].count)
        print(f'    Q: {item["question"]:<50s}  A: {top_ans}')

    # ── Sample balanced subsets ──
    rng = np.random.RandomState(args.seed)                                  # Line 81: reset RNG for reproducibility
    n = args.num_per_split

    if len(perception_items) < n:
        print(f'\n  [warn] Only {len(perception_items)} perception questions available (requested {n})')
        n = min(len(perception_items), len(knowledge_items))
    if len(knowledge_items) < n:
        print(f'\n  [warn] Only {len(knowledge_items)} knowledge questions available (requested {n})')
        n = min(len(perception_items), len(knowledge_items))

    perc_idx = rng.choice(len(perception_items), size=n, replace=False)     # Line 82: sample perception
    know_idx = rng.choice(len(knowledge_items), size=n, replace=False)      # Line 83: sample knowledge

    perc_sample = [perception_items[i] for i in perc_idx]
    know_sample = [knowledge_items[i] for i in know_idx]

    # ── Save as VQAv2-format JSON (compatible with existing load_vqav2) ──
    os.makedirs(args.output_dir, exist_ok=True)                             # Line 84: create output dir

    def save_split(items, name):
        """Save a split as separate questions and annotations files.

        Line 85: Outputs match VQAv2 format so load_vqav2() works unchanged.
        """
        questions_out = {                                                   # Line 86: questions file format
            'info': {'description': f'VQAv2 {name} split ({len(items)} questions)'},
            'questions': [
                {'question_id': it['question_id'],
                 'image_id': it['image_id'],
                 'question': it['question']}
                for it in items
            ]
        }
        annotations_out = {                                                 # Line 87: annotations file format
            'info': {'description': f'VQAv2 {name} split ({len(items)} annotations)'},
            'annotations': [
                {'question_id': it['question_id'],
                 'image_id': it['image_id'],
                 'question_type': it.get('question_type', ''),
                 'answer_type': it.get('answer_type', ''),
                 'answers': [{'answer': a, 'answer_confidence': 'yes',
                              'answer_id': i}
                             for i, a in enumerate(it['answers'])]}
                for it in items
            ]
        }

        q_path = os.path.join(args.output_dir, f'vqav2_{name}_questions.json')
        a_path = os.path.join(args.output_dir, f'vqav2_{name}_annotations.json')

        with open(q_path, 'w') as f:                                       # Line 88: write questions
            json.dump(questions_out, f)
        with open(a_path, 'w') as f:                                        # Line 89: write annotations
            json.dump(annotations_out, f)

        print(f'\n  Saved {name}: {q_path}')
        print(f'              {a_path}')
        return q_path, a_path

    save_split(perc_sample, 'perception')
    save_split(know_sample, 'knowledge')

    # ── Save classification stats ──
    stats = {                                                               # Line 90: save metadata
        'total_questions': total,
        'perception_total': len(perception_items),
        'knowledge_total': len(knowledge_items),
        'ambiguous_total': len(ambiguous_items),
        'sampled_per_split': n,
        'seed': args.seed,
    }
    stats_path = os.path.join(args.output_dir, 'split_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f'\n  Stats → {stats_path}')
    print(f'\nDone. Use these files with --vqav2_perception / --vqav2_knowledge flags.')


if __name__ == '__main__':
    main()
