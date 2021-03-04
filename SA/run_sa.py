import json
import argparse
import pandas as pd
import numpy as np
import random
from nltk.tokenize import word_tokenize

from SA.NLI_objective import NLIScorer
from SA.editor import RobertaEditor
from SA.generator_gpt import GPT2FluencyScorer
from SA.scoring_algos import SimulatedAnnealing
from SA.args import get_model_args

from rouge_score import rouge_scorer
import os

def get_dataset(scored_sentences_path, dataset_path, top_n):

    df = pd.read_csv(dataset_path, sep='\t', index_col=0)
    df = df.dropna()
    columns = ['dummy', 'id', 'statement', 'justification',
               'ruling_without_summary', 'label', 'just_tokenized',
               'ruling_tokenized', 'statement_tokenized', 'oracle_ids']
    df.columns = columns

    scored_sentences = [json.loads(line) for line in open(scored_sentences_path)] #, encoding='utf-8'
    scored_sentences = {item['id']: sorted(item['sentence_scores'], key=lambda x: x[1], reverse=True)[:top_n] for item in scored_sentences}
    scored_sentences = {k: [word_tokenize(sentence[0]) for sentence in v] for k, v in scored_sentences.items()}

    df['scored_sentences'] = df.apply(lambda x: scored_sentences.get(x['id'], None), axis=1)
    df = df[df['scored_sentences'] != None]
    df = df[['id', 'statement', 'justification', 'label', 'scored_sentences']]
    dataset = [row.to_dict() for i, row in df.iterrows()]

    print(f'Size of dataset: {len(dataset)}')
    print('Sample: ', dataset[0])

    return dataset

if __name__ == "__main__":

    sa_args = get_model_args()

    random.seed(sa_args.seed)
    np.random.seed(sa_args.seed)

    dataset = get_dataset(sa_args.sentences_path, sa_args.dataset_path, sa_args.top_n)

    editor = RobertaEditor(sa_args.editor_model_id, sa_args.device_type)
    fluency_scorer  =  GPT2FluencyScorer(sa_args.fluencyscorer_model_id, sa_args.device_type)
    editor.cuda()
    fluency_scorer.cuda()

    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    simulated_annealing = SimulatedAnnealing(editor,
                                             fluency_scorer,
                                             NLIScorer(sa_args.device_type),
                                             sa_args)

    sa_outputs = []
    os.remove('sa_inp.txt')
    os.remove('sa_out.txt')

    sa_inp = open('sa_inp.txt', 'a+')
    sa_out = open('sa_out.txt', 'a+')


    for i in range(0, len(dataset), sa_args.batch_size):
        batch_data = dataset[i: i + sa_args.batch_size]
        sa_outputs_batch = simulated_annealing.run(batch_data)

        scores_sa_justs = []
        scores_scored_justs = []

        for inp_batch, sa_just in zip(batch_data, sa_outputs_batch):


            temp_inp = []
            temp_out = []
            for sent in inp_batch['scored_sentences']:
                temp_inp.append(" ".join(sent))

            for out in sa_just:
                temp_out.append(" ".join(out))

            sa_inp.write(" ".join(temp_inp) + "\n")
            sa_out.write(" ".join(temp_out) + "\n")

            score1 = scorer.score(prediction='\n'.join(temp_out), target=inp_batch['justification'])
            scores_sa_justs.append(score1)

            score2 = scorer.score(prediction='\n'.join(temp_inp), target=inp_batch['justification'])
            scores_scored_justs.append(score2)

        sa_outputs += sa_outputs_batch

        print("Scores for justifications obtained by saliency scores")
        for score_name in score_names:
            print(f'{score_name} P: {np.mean([s[score_name].precision for s in scores_scored_justs]) * 100:.3f} '
                  f'R: {np.mean([s[score_name].recall for s in scores_scored_justs]) * 100:.3f} '
                  f'F1: {np.mean([s[score_name].fmeasure for s in scores_scored_justs]) * 100:.3f}')

        print("Scores for justifications given by SA")
        for score_name in score_names:
            print(f'{score_name} P: {np.mean([s[score_name].precision for s in scores_sa_justs]) * 100:.3f} '
                  f'R: {np.mean([s[score_name].recall for s in scores_sa_justs]) * 100:.3f} '
                  f'F1: {np.mean([s[score_name].fmeasure for s in scores_sa_justs]) * 100:.3f}')

        break
