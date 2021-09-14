import torch
import json
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


from transformers import PegasusForConditionalGeneration, PegasusTokenizer
from sentence_transformers import SentenceTransformer, util


class PegasusTool():

    def __init__(self, pegasus_model_name, model_sbert, device):
        print(device)

        self.model_name = pegasus_model_name
        self.device = device
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_name)
        self.model = PegasusForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        self.model_sbert = SentenceTransformer(model_sbert)
        self.num_beams = 10
        self.num_return_sequences = 10

    def get_response(self, input_text, num_return_sequences, num_beams):

        batch = self.tokenizer([input_text],truncation=True,padding='longest',max_length=60, return_tensors="pt").to(self.device)
        translated = self.model.generate(**batch, max_length=60,num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
        tgt_text = self.tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def sentence_level_semantic_scorer_sbert(self, org, rep):
        org_embeds = self.model_sbert.encode(org)
        rep_embeds = self.model_sbert.encode(rep)
        return torch.FloatTensor([util.pytorch_cos_sim(e1, e2) for e1, e2 in zip(org_embeds, rep_embeds)])

    def filter_justs(self, sa_out):
        temp = []
        
        for i in sent_tokenize(sa_out):

            if len(i.split(" ")) == 1:
                temp.append(i)
                continue
            else:
                all_responses = self.get_response(i, self.num_return_sequences, self.num_beams)
                temp_str = ''
                sim = self.sentence_level_semantic_scorer_sbert(all_responses, [i] * 10)
                max_sim_rep = all_responses[torch.argmax(sim)]
                temp.append(max_sim_rep)

        return (" ".join(temp))


if __name__== "__main__":

    '''
    Arguments: 
    sa_agrs.sentence_path, sa_agrs.dataset_path, sa_agrs.outfile_filtered, sa_args.outdir, device, sa_agrs.dataset_name, sa_agrs.outfile_pegasus
    
    '''

    sa_args = get_model_args()
    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    pegasus_mod = PegasusTool(sa_args.pegasus_modelname, sa_args.sbertname_pegasus, 'cuda')

    file_path1 = os.path.join(sa_args.outdir, sa_args.outfile_filtered)
    file_path2 = os.path.join(sa_args.outdir, sa_args.outfile_pegasus)


    if os.path.exists(file_path2):
        print("Removing already present output file: ", file_path2)
        os.remove(file_path2)

    saouts_pp = [line for line in open(file_path1, 'r')] #sainps_saouts, when we want to run Pegasus on SA inputs
    saouts_pegasus = open(file_path2, 'a+')

    scores_justs_pegasus = []
    sa_inp_tokens = []
    sa_out_tokens = []
    sa_out_pegasus_tokens = []
    gold_tokens = []

    dataset = get_dataset(sa_args)
    processed_samples = 0

    time1 = time.time()
    for saout, org_data in tqdm(zip(saouts_pp, dataset)):

        processed_samples+=1

        saout = saout.split("\t")[0]#Uncomment when running Pegasus on SA input

        filter_saout_pegasus = pegasus_mod.filter_justs(saout)
        saouts_pegasus.write(filter_saout_pegasus + "\n")

        sa_inp_tokens.append(len(org_data['scored_sentences'].split(" ")))
        sa_out_tokens.append(len(saout.split(" ")))
        gold_tokens.append(len(org_data['justification'].split(" ")))
        sa_out_pegasus_tokens.append(len(filter_saout_pegasus.split(" ")))

        score1 = scorer.score(prediction='\n'.join(sent_tokenize(filter_saout_pegasus)),
                              target='\n'.join(org_data['justification_sentences']))
        scores_justs_pegasus.append(score1)

    saouts_pegasus.close()

    print("Scores for filtered justifications (SA+Pegasus)")
    aggregate_print_rouges(score_names, scores_justs_pegasus)

    print("Average tokens in SA inputs: ", np.mean(sa_inp_tokens))
    print("Average tokens in SA outputs: ", np.mean(sa_out_tokens))
    print("Average tokens in SA outputs + Pegasus: ", np.mean(sa_out_pegasus_tokens))
    print("Average tokens in gold justifications: ", np.mean(gold_tokens))

    print("Processed: ", processed_samples)



