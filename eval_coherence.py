import argparse
import itertools

import torch
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from transformers import PreTrainedTokenizer, \
    RobertaForSequenceClassification, \
    RobertaTokenizerFast

from data_loader import get_dataset_df


def is_coherent_locally(args, justification: str,
                        tokeniser: PreTrainedTokenizer, model):
    """there is no pairwise disagreement between sentences which make
        up the explanation"""

    sentences = sent_tokenize(justification)

    if len(sentences) < 2:
        return True

    sentence_tuples = list(itertools.combinations(sentences, 2))
    predictions = []
    for i in range(0, len(sentence_tuples), args.batch_size):
        tuples_batch = sentence_tuples[i: i + args.batch_size]
        encoded_output = tokeniser(tuples_batch, padding=True,
                                   truncation=True, return_tensors='pt').to(
            model.device)

        output = model(**encoded_output)
        prediction = torch.argmax(output[0],
                                  dim=1).detach().cpu().numpy().tolist()
        predictions += prediction

    """"{ "0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"},"""
    if all(label in [1, 2] for label in predictions):
        return True

    return False


def is_coherent_globally(args, claim: str, justification: str,
                        tokeniser: PreTrainedTokenizer, model):
    """all explanatory sentences should entail or have a neutral relation
    with respect to the claim"""

    sentences = sent_tokenize(justification)

    if len(sentences) == 0:
        return True

    sentence_tuples = [(claim, sent) for sent in sentences]
    predictions = []

    for i in range(0, len(sentence_tuples), args.batch_size):
        tuples_batch = sentence_tuples[i: i + args.batch_size]
        encoded_output = tokeniser(tuples_batch, padding=True,
                                   truncation=True, return_tensors='pt').to(
            model.device)

        output = model(**encoded_output)
        prediction = torch.argmax(output[0],
                                  dim=1).detach().cpu().numpy().tolist()
        predictions += prediction

    """"{ "0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"},"""

    if all(label in [1, 2] for label in predictions):
        return True

    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=8)
    parser.add_argument("--df_path", help="Path to original split", type=str)
    parser.add_argument("--dataset", help="Name of dataset",
                        choices=['liar', 'pubhealth'])
    parser.add_argument("--coherence_type", help="Name of dataset",
                        choices=['local', 'global_weak'])

    args = parser.parse_args()

    device = torch.device("cuda") if args.gpu else torch.device("cpu")

    tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large-mnli')
    model = RobertaForSequenceClassification.from_pretrained(
        'roberta-large-mnli').to(device)

    df = get_dataset_df(args.dataset, args.df_path)

    coherent_local = []

    for just in tqdm(df['justification'].values):
        is_coherent = is_coherent_locally(args, just, tokenizer, model)
        coherent_local.append(is_coherent)

    print(f"Local coherence: {sum(coherent_local) / len(coherent_local)}")

    coherent_global_weak = []

    for i, row in tqdm(df.iterrows()):
        is_coherent = is_coherent_globally(args, row['statement'], row['justification'], tokenizer, model)
        coherent_global_weak.append(is_coherent)

    print(f"Global weak coherence: {sum(coherent_global_weak) / len(coherent_global_weak)}")


"""
python3.9 eval_coherence.py --df_path ../just_summ/oracles/ruling_oracles_test.tsv --dataset liar --gpu
Local coherence: 0.6685303514376997
Global weak coherence: 0.6445686900958466

python3.9 eval_coherence.py --df_path ../just_summ/oracles/ruling_oracles_val.tsv --dataset liar --gpu
Local coherence: 0.647887323943662
Global weak coherence: 0.634585289514867
"""