import argparse
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", help="Random seed", type=str)
    parser.add_argument("--output_file", help="Random seed", type=str)
    parser.add_argument("--df_path", help="Random seed", type=str)
    parser.add_argument("--top_n",
                        help="Top n sentences to consider for clean-up",
                        type=int)
    parser.add_argument("--remove_forbidden", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--remove_questions", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--remove_short", help="Flag for training on gpu",
                        action='store_true', default=False)
    parser.add_argument("--remove_long", help="Flag for training on gpu",
                        action='store_true', default=False)
    args = parser.parse_args()

    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    forbidden_words = set(open('data/forbidden_words.txt').read().split('\n'))

    columns = ['dummy', 'id', 'statement', 'justification',
               'ruling_without_summary', 'label', 'just_tokenized',
               'ruling_tokenized', 'statement_tokenized', 'oracle_ids']

    df = pd.read_csv(args.df_path, sep='\t', index_col=0)
    df = df.dropna()
    df.columns = columns

    len_sent_pre, len_sent_post, scores = [], [], []
    with open(args.output_file, 'w') as out_file:
        with open(args.file_path) as out:
            for line in tqdm(out):
                json_line = json.loads(line)

                post_sentences = []

                len_sent_pre.append(len(json_line['sentence_scores']))
                for sentence in json_line['sentence_scores']:
                    sentence_text = sentence[0]
                    sentence_text = sentence_text.lower()
                    tokens = sentence_text.split(' ')

                    if args.remove_forbidden and any(token in forbidden_words for token in tokens):
                        if 'rate' in sentence:
                            pass
                            # print(json_line['sentence_scores'])
                    elif args.remove_short and len(tokens) <= 1:
                        # print(json_line['sentence_scores'])
                        pass
                    elif args.remove_long and len(tokens) > 150:
                        pass
                        # print(json_line['sentence_scores'])
                    elif args.remove_questions and sentence_text.endswith('?'):
                        pass
                        # print(json_line['sentence_scores'])
                    else:
                        post_sentences.append(sentence)

                ordered_sentences = sorted(post_sentences,
                                           key=lambda x: x[1],
                                           reverse=True)[:args.top_n]

                len_sent_post.append(len(post_sentences))
                gold = list(df[df['id'] == json_line['id']].to_dict()['justification'].values())[0]
                instance = sent_tokenize(gold)
                score = scorer.score(prediction='\n'.join([s[0] for s in ordered_sentences]),
                                     target='\n'.join(instance))
                scores.append(score)

                json_line['sentence_scores'] = post_sentences
                out_file.write(json.dumps(json_line)+'\n')

    print(np.mean(len_sent_pre))
    print(np.mean(len_sent_post))

    print('sentence ROUGE:')
    for score_name in score_names:
        print(f'{score_name} P: {np.mean([s[score_name].precision for s in scores]) * 100:.3f} '
              f'R: {np.mean([s[score_name].recall for s in scores]) * 100:.3f} '
              f'F1: {np.mean([s[score_name].fmeasure for s in scores]) * 100:.3f}')

"python clean_sentences.py --file_path /image/image-copenlu/unsupervised_fc/sup_sccores/results_serialized_val.jsonl --df_path ../just_summ/oracles/ruling_oracles_val.tsv --top_n 6 --output_file /image/image-copenlu/unsupervised_fc/sup_sccores/results_serialized_val_filtered.jsonl --remove_long --remove_short --remove_questions"