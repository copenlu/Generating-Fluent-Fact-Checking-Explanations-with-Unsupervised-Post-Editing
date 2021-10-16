import argparse
import textstat
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring

from data_loader import get_dataset_df
from old.eval_coherence import get_dataset
from baselines import aggregate_print_rouges

textstat.set_lang('en_US')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", help="Top sentences to select from file",
                        type=int, default=6)
    parser.add_argument("--dataset_path", help="Path to original split",
                        type=str)
    parser.add_argument("--sentences_path", help="Path to original split",
                        type=str)
    parser.add_argument("--file_path", help="Path to final explanations",
                        type=str, default=None)
    parser.add_argument("--dataset_name", help="Name of dataset",
                        choices=['liar', 'pubhealth'])
    parser.add_argument("--mode", help="Name of dataset",
                        choices=['justification', 'lead_3', 'lead_4', 'lead_5',
                                 'lead_6', 'from_file', 'from_top'])

    args = parser.parse_args()
    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    if args.mode in ['from_file', 'from_top']:
        dataset = get_dataset(args)
    else:
        df = get_dataset_df(args.dataset_name, args.dataset_path)
        dataset = [row.to_dict() for i, row in df.iterrows()]

    scores = []
    for i, row in tqdm(enumerate(dataset)):
        gold = row['justification_sentences']
        text = sent_tokenize(row['input_file'])
        score = scorer.score(prediction='\n'.join(text),
                             target='\n'.join(gold))
        scores.append(score)


    aggregate_print_rouges(score_names, scores)
