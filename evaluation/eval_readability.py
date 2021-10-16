import argparse
import textstat
import numpy as np
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from data_loader import get_dataset_df


def get_dataset(args):
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

    with open(args.file_path) as out:
        total_expl = 0
        for i, line in enumerate(out):
            total_expl += 1
            dataset[i]['input_file'] = line

    print(total_expl)
    return dataset


textstat.set_lang('en_US')


def is_readable(justification: str):
    res = {}
    res['flesch'] = textstat.flesch_reading_ease(justification)
    res['flesch_grade'] = textstat.flesch_kincaid_grade(justification)
    res['dale'] = textstat.dale_chall_readability_score(justification)

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", help="Flag for training on gpu.",
                        action='store_true', default=False)
    parser.add_argument("--top_n", help="Top sentences to select from file.",
                        type=int, default=6)
    parser.add_argument("--dataset_path", help="Path to original split.",
                        type=str)
    parser.add_argument("--sentences_path", help="Path to selected sentences.",
                        type=str)
    parser.add_argument("--file_path", help="Path to final explanations.",
                        type=str, default=None)
    parser.add_argument("--dataset_name", help="Name of dataset",
                        choices=['liar', 'pubhealth'])
    parser.add_argument("--mode", help="Name of dataset",
                        choices=['justification', 'lead_3', 'lead_4', 'lead_5',
                                 'lead_6', 'from_file', 'from_top'])

    args = parser.parse_args()

    if args.mode in ['from_file', 'from_top']:
        dataset = get_dataset(args)
    else:
        df = get_dataset_df(args.dataset_name, args.dataset_path)
        dataset = [row.to_dict() for i, row in df.iterrows()]

    readability = []

    for i, row in tqdm(enumerate(dataset)):
        if args.mode == 'justification':
            text = row['justification']
        elif args.mode == 'from_file':
            text = row['input_file']
        elif args.mode == 'from_top':
            text = row['scored_sentences']
        elif args.mode.startswith('lead'):
            top_n = int(args.mode.split('_')[1])
            text = sent_tokenize(row['ruling_without_summary'])[:top_n]
            text = ' '.join(text)

        readable = is_readable(text)
        readability.append(readable)

    print(args)
    print(f"Readability:")
    print({k: (
        np.mean([r[k] for r in readability]),
        np.std([r[k] for r in readability]),
    )
        for k in readability[0].keys()})