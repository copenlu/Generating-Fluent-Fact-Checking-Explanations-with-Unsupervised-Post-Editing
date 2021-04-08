import json
import pandas as pd
import numpy as np
import random
from nltk.tokenize import sent_tokenize

from SA.editor import RobertaEditor
from SA.generator_gpt import GPT2FluencyScorer
from SA.scoring_algos import SimulatedAnnealing
from SA.args import get_model_args
from baselines import aggregate_print_rouges
from SA.extract_phrases import parser

from rouge_score import rouge_scorer
import os.path
import torch


def clean_str(sent):
    sent = sent.replace("’", "'")
    sent = sent.replace("‘", "`")
    sent = sent.replace('"', "''")
    sent = sent.replace("—", "--")
    sent = sent.replace("…", "...")
    sent = sent.replace("–", "--")

    return sent.strip()

def get_dataset(scored_sentences_path, dataset_path, top_n, parser):

    df = pd.read_csv(dataset_path, sep='\t', index_col=0)
    df = df.dropna()
    columns = ['dummy', 'id', 'statement', 'justification',
               'ruling_without_summary', 'label', 'just_tokenized',
               'ruling_tokenized', 'statement_tokenized', 'oracle_ids']
    df.columns = columns

    scored_sentences = [json.loads(line) for line in open(scored_sentences_path)]
    scored_sentences = {item['id']: sorted(item['sentence_scores'], key=lambda x: x[1], reverse=True)[:top_n] for item in scored_sentences}
    #scored_sentences = {k: [word_tokenize(sentence[0]) for sentence in v] for k, v in scored_sentences.items()}
    #Modification to take input as a complete justification instead of separate sentences.
    inp_scored_sentences = {}
    for k, v in scored_sentences.items():
        temp = []
        for sent in v:
            temp.append(sent[0])
        #inp_scored_sentences[k] = clean_str(" ".join(parser.tokenize(" ".join(temp))))
        inp_scored_sentences[k] = clean_str(" ".join(temp))
        if len(inp_scored_sentences[k]) == 0:
            inp_scored_sentences[k] = None

    scored_sentences = inp_scored_sentences

    df['scored_sentences'] = df.apply(lambda x: scored_sentences.get(x['id'], None), axis=1)
    df = df[df['scored_sentences'] != None]

    df['justification_sentences'] = df.apply(lambda x: sent_tokenize(x['justification']), axis=1)

    df = df[['id', 'statement', 'justification', 'label', 'scored_sentences',
             'justification_sentences']]
    dataset = [row.to_dict() for i, row in df.iterrows()]
    new_dataset = []
    for i in dataset:
        if i["scored_sentences"] is None or i["id"] == '2001.json': #Sentence in Liarplus is too long:
            continue
        else:
            new_dataset.append(i)

    print(f'Size of dataset: {len(dataset)}')
    print(f'Size of new dataset: {len(new_dataset)}')
    print('Sample: ', new_dataset[0])

    return new_dataset


if __name__ == "__main__":

    sa_args = get_model_args()

    random.seed(sa_args.seed)
    np.random.seed(sa_args.seed)

    dataset = get_dataset(sa_args.sentences_path, sa_args.dataset_path, sa_args.top_n, parser)

    if sa_args.sample:
        print(f"Sampling {sa_args.sample} instances from the dataset")
        dataset = np.random.choice(dataset, sa_args.sample)

    if sa_args.device_type == "gpu":
        device = "cuda"
    else:
        device = sa_args.device_type

    num_gpus = torch.cuda.device_count()

    fluency_scorer = GPT2FluencyScorer(sa_args.fluencyscorer_model_id, device)
    editor = RobertaEditor(sa_args.editor_model_id, device, sa_args.min_length_of_edited_sent, fluency_scorer)

    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    simulated_annealing = SimulatedAnnealing(editor,
                                             fluency_scorer,
                                             sa_args, device)

    if num_gpus > 1:
        print(f'Using {num_gpus} GPUs')
        editor.model = torch.nn.DataParallel(editor.model, device_ids=[0, 1])
        fluency_scorer.model = torch.nn.DataParallel(fluency_scorer.model, device_ids=[0, 1])
        # TODO add parallel computation for the sentence sim model
        # simulated_annealing.sbert = torch.nn.DataParallel(simulated_annealing.sbert, device_ids=[0, 1])

    # TODO write is needed once for gold and separately for each step
    if os.path.exists('sa_inp.txt'):
        print("Removing already present output file")
        os.remove('sa_inp.txt')

    if os.path.exists('sa_out.txt'):
        os.remove('sa_out.txt')

    sa_inp = open('sa_inp.txt', 'a+')
    sa_out = open('sa_out.txt', 'a+')

    processed_samples = 0
    scores_sa_justs = []
    sa_outputs = []
    sa_inp_tokens = []
    sa_out_tokens = []
    gold_tokens = []

    for i in range(0, len(dataset), sa_args.batch_size):
       
        batch_data = dataset[i: i + sa_args.batch_size]
        sa_outputs_batch = simulated_annealing.run(batch_data)
        processed_samples += len(batch_data)
        print("Processing: ", processed_samples)
        print("------------")

        for instance, instance_edit in zip(batch_data, sa_outputs_batch):

            # TODO write new text and what was the edit operation
            sa_inp.write(instance['scored_sentences'] + "\n")
            sa_out.write(instance_edit + "\n")

            sa_inp_tokens.append(len(instance['scored_sentences'].split(" ")))
            sa_out_tokens.append(len(instance_edit.split(" ")))
            gold_tokens.append(len(instance['justification'].split(" ")))

            print("SA_input: ", instance['scored_sentences'])
            print("\n")
            print("SA_output: ", instance_edit)
            print("\n")
            print("Golden_just: ", instance['justification'])
            print("\n")
            print("----------------------------------------------------------------------\n")

            score1 = scorer.score(prediction='\n'.join(sent_tokenize(instance_edit)),
                                  target='\n'.join(instance['justification_sentences']))
            scores_sa_justs.append(score1)

        sa_outputs += sa_outputs_batch

    scores_original_sentences = [scorer.score(prediction='\n'.join(sent_tokenize(instance['scored_sentences'])),
                                              target='\n'.join(instance['justification_sentences']))
                                 for instance in dataset]

    print(f"Scores for originally selected {sa_args.top_n} sentences")
    aggregate_print_rouges(score_names, scores_original_sentences)

    print("Scores for justifications given by SA")
    aggregate_print_rouges(score_names, scores_sa_justs)

    print("Average tokens in SA inputs: ", np.mean(sa_inp_tokens))
    print("Average tokens in SA outputs: ", np.mean(sa_out_tokens))
    print("Average tokens in gold justifications: ", np.mean(gold_tokens))

    print("Processed: ", processed_samples)
