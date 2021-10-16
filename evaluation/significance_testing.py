import json
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import mannwhitneyu, chisquare, ttest_ind

import argparse
import textstat
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring

from evaluation.eval_readability import get_dataset

textstat.set_lang('en_US')


def get_explanation_scores(justification: str, gold: str):
    res = {}
    res['flesch'] = textstat.flesch_reading_ease(justification)
    res['dale'] = textstat.dale_chall_readability_score(justification)

    score = scorer.score(prediction='\n'.join(sent_tokenize(justification)),
                         target='\n'.join(gold))
    res['rouge1'] = score['rouge1'].fmeasure * 100
    res['rouge2'] = score['rouge2'].fmeasure * 100
    res['rougeLsum'] = score['rougeLsum'].fmeasure * 100

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", help="Top sentences to select from file",
                        type=int, default=6)
    parser.add_argument("--dataset_path", help="Path to original split",
                        type=str)
    parser.add_argument("--sentences_path", help="Path to original split",
                        type=str)
    parser.add_argument("--file_path1", help="Path to final explanations",
                        type=str, default=None)
    parser.add_argument("--file_path2", help="Path to final explanations",
                        type=str, default=None)
    parser.add_argument("--dataset_name", help="Name of dataset",
                        choices=['liar', 'pubhealth'])
    parser.add_argument("--mode", help="Mode for firs dataset",
                        choices=['justification', 'lead_3', 'lead_4', 'lead_5',
                                 'lead_6', 'from_file', 'from_top'])

    args = parser.parse_args()

    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    dataset1 = get_dataset(args, args.file_path1)
    dataset2 = get_dataset(args, args.file_path2)

    scores1, scores2 = [], []

    for i, row in tqdm(enumerate(dataset1)):
        if args.mode == 'from_file':
            text = row['input_file']
        elif args.mode == 'from_top':
            text = row['scored_sentences']

        scores = get_explanation_scores(text, row['justification_sentences'])
        scores1.append(scores)

    for i, row in tqdm(enumerate(dataset2)):
        text = row['input_file']
        scores = get_explanation_scores(text, row['justification_sentences'])
        scores2.append(scores)

    for k in scores1[0].keys():
        print(k, end=': ')

        prop_scores1 = [t[k] for t in scores1]
        prop_scores2 = [t[k] for t in scores2]

        print(ttest_ind(prop_scores1, prop_scores2))