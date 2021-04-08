import argparse
import random
from multiprocessing import Pool
from typing import List
import pprint
import numpy as np
from rouge_score import rouge_scorer, scoring
from tqdm import tqdm

from data_loader import get_datasets


def greedy_oracle(sentences: List[str],
                  query: str,
                  scorer: rouge_scorer.RougeScorer,
                  criterion: str = 'rouge2'):
    sentence_scores = []

    for sentence in sentences:
        sentence_scores.append(scorer.score(prediction=sentence,
                                            target=query)[criterion].fmeasure)

    best_sentences = np.argsort(sentence_scores)[::-1][:4]

    return [sentences[i] for i in best_sentences]


def aggregate_print_rouges(score_names, scores):
    aggregator = scoring.BootstrapAggregator()

    for score in scores:
        aggregator.add_scores(score)
    aggregated_scores = aggregator.aggregate()
    pprint.pprint(aggregated_scores)

    for score_name in score_names:
        mid = aggregated_scores[score_name].mid
        low = aggregated_scores[score_name].low
        high = aggregated_scores[score_name].high

        print(f'{score_name} P: {mid.precision * 100:.2f} \t CI({low.precision* 100:.2f}-{high.precision* 100:.2f}) \t '
              f' R: {mid.recall * 100:.2f} \t CI({low.recall* 100:.2f}-{high.recall* 100:.2f}) \t '
              f' F1: {mid.fmeasure * 100:.2f} \t CI({low.fmeasure* 100:.2f}-{high.fmeasure* 100:.2f})')


def greedy_beam_search(sentences: List[str],
                       query: str,
                       scorer: rouge_scorer.RougeScorer,
                       beam_size: int = -1,
                       criterion: str = 'rouge2'):
    individual_scores = p.starmap(scorer.score, [(query, s) for s in sentences])
    individual_scores = [s[criterion].fmeasure for s in individual_scores]

    sentences_idx = np.argsort(individual_scores)[::-1][:beam_size]
    selected_combinations = [[sentences[i]]
                             for i in sentences_idx
                             if individual_scores[i] > 0]
    if len(selected_combinations) <= 4:
        return selected_combinations[0]

    sentences = [sentences[i] for i in sentences_idx]
    for num_sent in range(0, 3):
        all_combinations = [comb+[s]
                            for comb in selected_combinations
                            for i, s in enumerate(sentences)
                            if s not in comb and individual_scores[i] > 0]

        scores = p.starmap(scorer.score, [(query, '\n'.join(comb)) for comb in all_combinations])
        scores = [s[criterion].fmeasure for s in scores]

        comb_idx = np.argsort(scores)[::-1][:beam_size]
        selected_combinations = [all_combinations[i] for i in comb_idx]

    oracle = selected_combinations[0]
    return oracle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", help="Random seed", type=int, default=73)
    parser.add_argument("--labels", help="Number of labels", type=int,
                        default=6)
    parser.add_argument("--leadn", help="Number of sentences for the "
                                        "Lead-N baseline", type=int,
                        default=4)
    parser.add_argument("--dataset", help="Type of the dataset",
                        choices=['liar', 'pubhealth'],
                        default='liar')
    parser.add_argument("--dataset_dir", help="Path to the datasets",
                        default='data/liar/', type=str)

    parser.add_argument("--baseline", help="Type of baseline",
                        choices=['lead', 'oracle_claim_rouge', 'claim'])
    parser.add_argument("--beam_size", help="Size of beam in oracle baseline",
                        type=int, default=-1)

    args = parser.parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)

    train, val, test = get_datasets(args.dataset_dir,
                                    args.dataset,
                                    args.labels,
                                    add_logits=False)
    print(f'Train size {len(train)}', flush=True)
    print(f'Dev size {len(val)}', flush=True)

    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    for dataset_tuple in [(val, 'validation'), (test, 'test')]:
        dataset, dataset_name = dataset_tuple
        print(dataset_name)

        scores = []
        if args.baseline == 'lead':
            for instance in dataset:
                prediction = instance['text'][:args.leadn]
                score = scorer.score(prediction='\n'.join(prediction),
                                     target='\n'.join(instance['explanation_text']))
                scores.append(score)

        if args.baseline == 'claim':

            for instance in dataset:
                prediction = instance['text_query']
                score = scorer.score(prediction=prediction,
                                     target='\n'.join(instance['explanation_text']))
                scores.append(score)

        elif args.baseline == 'oracle_claim_rouge':
            p = Pool()
            for instance in tqdm(dataset):
                prediction = greedy_beam_search(instance['text'],
                                        instance['text_query'], scorer)
                score = scorer.score(prediction='\n'.join(prediction),
                                     target='\n'.join(instance['explanation_text']))
                scores.append(score)

        print(f'Baseline {args.baseline} score:')
        aggregate_print_rouges(score_names, scores)