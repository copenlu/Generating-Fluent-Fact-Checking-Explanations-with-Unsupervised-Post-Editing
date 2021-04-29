import json
import numpy as np
import random
from nltk.tokenize import sent_tokenize

from SA.editor import RobertaEditor
from SA.generator_gpt import GPT2FluencyScorer
from SA.scoring_algos import SimulatedAnnealing
from SA.args import get_model_args
from baselines import aggregate_print_rouges
from SA.extract_phrases import parser
from data_loader import get_dataset_df

from rouge_score import rouge_scorer
import os.path
import torch
import os


def clean_str(sent):
    sent = sent.replace("’", "'")
    sent = sent.replace("‘", "`")
    sent = sent.replace('"', "''")
    sent = sent.replace("—", "--")
    sent = sent.replace("…", "...")
    sent = sent.replace("–", "--")

    return sent.strip()


def get_dataset(args):
    df = get_dataset_df(args.dataset_name, args.dataset_path)
    df['claim_id'] = df['claim_id'].astype('str')

    scored_sentences = [json.loads(line) for line in open(args.sentences_path)]
    scored_sentences = {str(item["id"]): sorted(item['sentence_scores'], key=lambda x: x[1], reverse=True)[:args.top_n] for item in scored_sentences}

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

    # if dataset_name == 'liar':
    #     df['scored_sentences'] = df.apply(lambda x: scored_sentences.get(x['id'], None), axis=1)
    #     # df = df[df['scored_sentences'] != None]
    #     df["scored_sentences"] = df["scored_sentences"].apply(lambda x: x.replace("\n", ""))
    #     df['justification_sentences'] = df.apply(lambda x: sent_tokenize(x['justification']), axis=1)
    #     df = df[['id', 'statement', 'justification', 'label', 'scored_sentences',
    #          'justification_sentences']]
    #
    # elif dataset_name == 'pub_health':
    #     # df['claim_id'] = df['claim_id'].astype('str')
    #     df['scored_sentences'] = df.apply(lambda x: scored_sentences.get(x['claim_id'], None), axis=1)
    #     # df = df[df['scored_sentences'] != None]
    #     df['justification_sentences'] = df.apply(lambda x: sent_tokenize(x['explanation']), axis=1)
    #     df = df[['claim_id', 'claim', 'explanation', 'label', 'scored_sentences',
    #          'justification_sentences']]
        
    dataset = [row.to_dict() for i, row in df.iterrows()]
    # new_dataset = []

    # if dataset_name == 'liar':
    #     for i in dataset:
    #         # remove_ids = ['2161.json', '2001.json', '1777.json']  # in validation, '1777.json-supTest'
    #         if i["scored_sentences"] is None or i["id"] in remove_ids: #Sentence in Liarplus is too long:
    #             continue
    #         else:
    #         	i["scored_sentences"] = i["scored_sentences"].replace("\n", "")
    #         	new_dataset.append(i)
    # elif dataset_name == 'pub_health':
    #     for i in dataset:
    #         # remove_ids = ['41862', '36094', '30819', '36167', '37958']
    #         if i["scored_sentences"] is None or i["claim_id"] in remove_ids or i["scored_sentences"] == None:
    #             continue
    #         else:
    #             new_dataset.append(i)
    
    print(f'Size of dataset: {len(dataset)}')
    print('Sample: ', dataset[0])
    return dataset
 
def get_string_scores(scores, score_names):
    for score_name in score_names:
        print(f'{score_name} P: {np.mean([s[score_name].precision for s in scores]) * 100:.3f} '
              f'R: {np.mean([s[score_name].recall for s in scores]) * 100:.3f} '
              f'F1: {np.mean([s[score_name].fmeasure for s in scores]) * 100:.3f}')


if __name__ == "__main__":

    sa_args = get_model_args()

    random.seed(sa_args.seed)
    np.random.seed(sa_args.seed)

    dataset = get_dataset(sa_args)
    dataset = dataset

    if sa_args.sample:
        print(f"Sampling {sa_args.sample} instances from the dataset")
        dataset = np.random.choice(dataset, sa_args.sample)

    if sa_args.device_type == "gpu":
        device = "cuda"
    else:
        device = sa_args.device_type

    num_gpus = torch.cuda.device_count()

    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    fluency_scorer = GPT2FluencyScorer(sa_args.fluencyscorer_model_id, device)
    editor = RobertaEditor(sa_args.editor_model_id, device, sa_args.min_length_of_edited_sent, fluency_scorer)
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

    file_path = os.path.join(sa_args.outdir, sa_args.outfile)
    if os.path.exists(file_path):
        print("Removing already present output file")
        os.remove(file_path)

    if os.path.isdir(sa_args.outdir) is not True:
        os.mkdir(sa_args.outdir)
    sa_inp_out = open(file_path, 'a+')

    processed_samples = 0
    scores_sa_justs = []
    sa_outputs = []
    sa_inp_tokens = []
    sa_out_tokens = []
    gold_tokens = []

    for i in range(0, len(dataset), sa_args.batch_size):
       
        batch_data = dataset[i: i + sa_args.batch_size]
        '''
        for i in batch_data:
            print(i["claim_id"])
            print(i["scored_sentences"])
        '''
        sa_outputs_batch = simulated_annealing.run(batch_data)
        processed_samples += len(batch_data)
        print("Processing: ", processed_samples)
        print("------------")

        for instance, instance_edit in zip(batch_data, sa_outputs_batch):

            # TODO write new text and what was the edit operation
            
            sa_inp_out.write(instance['scored_sentences'] + '\t' + instance_edit + "\n")
            
            sa_inp_tokens.append(len(instance['scored_sentences'].split(" ")))
            sa_out_tokens.append(len(instance_edit.split(" ")))
            gold_tokens.append(len(instance['justification'].split(" ")))

            score1 = scorer.score(prediction='\n'.join(sent_tokenize(instance_edit)),
                                  target='\n'.join(instance['justification_sentences']))
            scores_sa_justs.append(score1)

        sa_outputs += sa_outputs_batch

    scores_original_sentences = [scorer.score(prediction='\n'.join(sent_tokenize(instance['scored_sentences'])),
                                              target='\n'.join(instance['justification_sentences']))
                                 for instance in dataset]

    sa_inp_out.close()
    print(f"Scores for originally selected {sa_args.top_n} sentences")
    aggregate_print_rouges(score_names, scores_original_sentences)

    print("Scores for justifications given by SA")
    aggregate_print_rouges(score_names, scores_sa_justs)

    print("Average tokens in SA inputs: ", np.mean(sa_inp_tokens))
    print("Average tokens in SA outputs: ", np.mean(sa_out_tokens))
    print("Average tokens in gold justifications: ", np.mean(gold_tokens))

    print("Processed: ", processed_samples)
