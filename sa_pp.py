import torch
import json
import language_tool_python
import time
import numpy as np
import os
from tqdm import tqdm

from nltk.tokenize import sent_tokenize
from SA.args import get_model_args
from run_sa import get_string_scores
from run_sa import get_dataset
from rouge_score import rouge_scorer
from baselines import aggregate_print_rouges


import spacy

tool = language_tool_python.LanguageTool('en-US')
nlp = spacy.load("en_core_web_sm")


def gramatical_tool(self, sent):
    matches = tool.check(sent)
    return language_tool_python.utils.correct(sent, matches)


def post_process(text):
    valid_sents = []
    for i in sent_tokenize(text):
        i = gramatical_tool(i)
        doc = nlp(i)
        verbs = [token.text for token in doc if token.pos_ in ['VERB', 'AUX']]
        if len(verbs) > 0:
            valid_sents.append(i)

    return " ".join(valid_sents)



if __name__== "__main__":

    '''
    Arguments: 
    sa_agrs.sentence_path, sa_agrs.dataset_path, sa_args.outfile, sa_args.outdir, sa_agrs.outfile_filtered, sa_args.dataset_name 
    '''

    sa_args = get_model_args()
    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    file_path1 = os.path.join(sa_args.outdir, sa_args.outfile) #sa-outputs
    file_path2 = os.path.join(sa_args.outdir, sa_args.outfile_filtered)

    if os.path.exists(file_path2):
        print("Removing already present output file: ",file_path2)
        os.remove(file_path2)

    sainps_saouts = [line for line in open(file_path1, 'r')]
    saouts_pp = open(file_path2, 'a+')

    scores_sa_pp_justs = []
    sa_inp_tokens = []
    sa_out_tokens = []
    sa_out_pp_tokens = []

    dataset = get_dataset(sa_args)
    processed_samples = 0

    for sainp_saout, org_data in tqdm(zip(sainps_saouts, dataset)):

        processed_samples+=1

        saout = sainp_saout.split("\t")[1]
        filter_saout = post_process(saout)

        saouts_pp.write(filter_saout + "\n")

        sa_inp_tokens.append(len(org_data['scored_sentences'].split(" ")))
        sa_out_tokens.append(len(saout.split(" ")))
        sa_out_pp_tokens.append(len(filter_saout.split(" ")))

        score1 = scorer.score(prediction='\n'.join(sent_tokenize(filter_saout)),
                              target='\n'.join(org_data['justification_sentences']))
        scores_sa_pp_justs.append(score1)

    saouts_pp.close()

    print("Scores for filtered justifications (SA+Post-Process)")
    aggregate_print_rouges(score_names, scores_sa_pp_justs)

    print("Average tokens in SA inputs: ", np.mean(sa_inp_tokens))
    print("Average tokens in SA outputs: ", np.mean(sa_out_tokens))
    print("Average tokens in SA outputs + Pegasus: ", np.mean(sa_out_pp_tokens))

    print("Processed: ", processed_samples)



