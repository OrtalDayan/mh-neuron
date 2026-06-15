"""
TriviaQA dataset for VLMEvalKit.

Text-only free-form QA evaluation. Inherits from TextBaseDataset to skip
image loading entirely. Uses alias-matching (TriviaQA standard) rather than
MCQ logic.

Expected TSV columns at ~/LMUData/TriviaQA.tsv:
    index    — unique integer id
    question — the question string
    answer   — canonical answer string
    aliases  — pipe-delimited (|) list of accepted alternative answers
    split    — e.g. 'verified-web-dev' (for future filtering)
    category — optional category tag (for future filtering)
"""
import os
import os.path as osp
import pandas as pd

from .text_base import TextBaseDataset
from .utils.triviaqa import exact_match_with_aliases
from ..smp import load, dump, get_logger, d2df


class TriviaQA(TextBaseDataset):
    TYPE = 'VQA'  # free-form generation; VQA type is closest match for VLMEvalKit's result handling
    MODALITY = 'TEXT'

    # No DATASET_URL — the TSV is prepared locally by json_to_tsv.py
    # from data/triviaqa/qa/verified-web-dev.json. Loading goes through the
    # CustomTextMCQDataset-style local path below.
    DATASET_URL = {'TriviaQA': ''}
    DATASET_MD5 = {'TriviaQA': ''}

    def load_data(self, dataset):
        """Load TSV from LMUData/ — local file, no remote download."""
        from ..smp import LMUDataRoot
        data_path = osp.join(LMUDataRoot(), f'{dataset}.tsv')
        assert osp.exists(data_path), (
            f'TriviaQA TSV not found at {data_path}. '
            f'Run code/json_to_tsv_triviaqa.py to generate it.'
        )
        return load(data_path)

    def build_prompt(self, line):
        """Text-only prompt — no image, just the question."""
        if isinstance(line, int):
            line = self.data.iloc[line]
        question = line['question']
        return [dict(type='text', value=question)]

    def evaluate(self, eval_file, **judge_kwargs):
        """
        Alias-matched exact-match scoring.

        Produces <eval_file>_score.csv with overall accuracy and, if 'category'
        or 'split' columns are present, per-category/split breakdowns.
        """
        logger = get_logger('Evaluation')
        data = load(eval_file)

        # Align eval_file predictions with TSV aliases/splits by index
        meta = self.data
        meta_by_idx = {str(row['index']): row for _, row in meta.iterrows()}

        correct_list = []
        for i in range(len(data)):
            item = data.iloc[i]
            idx = str(item['index'])
            pred = item.get('prediction', '')
            meta_row = meta_by_idx.get(idx)
            if meta_row is None:
                correct_list.append(False)
                continue
            answer = meta_row.get('answer', '')
            aliases = meta_row.get('aliases', None)
            is_correct = exact_match_with_aliases(pred, answer, aliases)
            correct_list.append(is_correct)

        data['correct'] = correct_list
        suffix = eval_file.split('.')[-1]
        result_file = eval_file.replace(f'.{suffix}', f'_result.{suffix}')
        dump(data, result_file)

        # Score CSV — overall + optional per-split/per-category
        scores = {'Overall': float(sum(correct_list)) / max(len(correct_list), 1)}
        # Merge meta split/category into data for groupby
        if 'split' in meta.columns:
            data['_split'] = [meta_by_idx.get(str(i), {}).get('split', '') for i in data['index']]
            for sp in set(data['_split']):
                if sp:
                    mask = data['_split'] == sp
                    scores[f'Split-{sp}'] = float(data.loc[mask, 'correct'].sum()) / max(mask.sum(), 1)
        if 'category' in meta.columns:
            data['_cat'] = [meta_by_idx.get(str(i), {}).get('category', '') for i in data['index']]
            for ct in set(data['_cat']):
                if ct:
                    mask = data['_cat'] == ct
                    scores[f'Category-{ct}'] = float(data.loc[mask, 'correct'].sum()) / max(mask.sum(), 1)

        score_df = d2df(scores)
        score_file = eval_file.replace(f'.{suffix}', '_score.csv')
        dump(score_df, score_file)
        logger.info(f'TriviaQA eval complete. Score file: {score_file}')
        logger.info(f'Accuracy: {scores["Overall"]:.4f} ({sum(correct_list)}/{len(correct_list)})')
        return score_df
