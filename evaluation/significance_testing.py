import json
from statsmodels.stats.contingency_tables import mcnemar
from scipy.stats import mannwhitneyu, chisquare, ttest_ind

import argparse
import textstat
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from rouge_score import rouge_scorer, scoring

from data_loader import get_dataset_df
from old.eval_coherence import clean_str

textstat.set_lang('en_US')


def get_dataset(args, fp):
    df = get_dataset_df(args.dataset_name, args.dataset_path)
    df['claim_id'] = df['claim_id'].astype('str')

    scored_sentences = [json.loads(line) for line in open(args.sentences_path)]
    scored_sentences = {
        str(item["id"]): sorted(item['sentence_scores'], key=lambda x: x[1],
                                reverse=True)[:args.top_n] for item in
        scored_sentences}

    inp_scored_sentences = {}
    for k, v in scored_sentences.items():
        temp = []
        for sent in v:
            temp.append(sent[0])
        inp_scored_sentences[k] = clean_str(" ".join(temp))

    scored_sentences = inp_scored_sentences

    df['scored_sentences'] = df.apply(
        lambda x: scored_sentences.get(x['claim_id'], None), axis=1)
    df = df[df['scored_sentences'].notna()]

    df["scored_sentences"] = df["scored_sentences"].apply(
        lambda x: x.replace("\n", ""))
    df['justification_sentences'] = df.apply(
        lambda x: sent_tokenize(x['justification']), axis=1)

    dataset = [row.to_dict() for i, row in df.iterrows()]

    print(f'Size of dataset: {len(dataset)}')

    with open(fp) as out:
        total_expl = 0
        for i, line in enumerate(out):
            total_expl += 1
            dataset[i]['input_file'] = line

    print(total_expl)
    return dataset


def is_readable(justification: str, gold: str):
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

        scores = is_readable(text, row['justification_sentences'])
        scores1.append(scores)

    for i, row in tqdm(enumerate(dataset2)):
        text = row['input_file']
        scores = is_readable(text, row['justification_sentences'])
        scores2.append(scores)

    for k in scores1[0].keys():
        print(k, end=': ')

        prop_scores1 = [t[k] for t in scores1]
        prop_scores2 = [t[k] for t in scores2]

        print(ttest_ind(prop_scores1, prop_scores2))