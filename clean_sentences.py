import argparse
import json
import numpy as np
from tqdm import tqdm
from rouge_score import rouge_scorer
from nltk.tokenize import sent_tokenize
from data_loader import get_dataset_df

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
    parser.add_argument("--dataset", help="Flag for training on gpu",
                        choices=['liar', 'pubhealth'])
    args = parser.parse_args()

    score_names = ['rouge1', 'rouge2', 'rougeLsum']
    scorer = rouge_scorer.RougeScorer(score_names, use_stemmer=True)

    forbidden_words = set(open('data/forbidden_words.txt').read().split('\n')+['rate'])
    df = get_dataset_df(args.dataset, args.df_path)
    df['claim_id'] = df['claim_id'].astype('str')

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
                        pass
                    elif args.remove_short and len(tokens) <= 1:
                        pass
                    elif args.remove_long and len(tokens) > 70:
                        pass
                    elif args.remove_questions and sentence_text.endswith('?'):
                        pass
                    else:
                        post_sentences.append(sentence)

                if len (post_sentences) == 0:
                    print("No sentences!")

                ordered_sentences = sorted(post_sentences,
                                           key=lambda x: x[1],
                                           reverse=True)[:args.top_n]

                len_sent_post.append(len(ordered_sentences))

                gold = list(df[df['claim_id'] == str(json_line['id'])].to_dict()['justification'].values())
                if len(gold) == 0:
                    print(json_line['id'])
                    continue
                gold = gold[0]

                instance = sent_tokenize(gold)
                score = scorer.score(prediction='\n'.join([s[0] for s in ordered_sentences]),
                                     target='\n'.join(instance))
                scores.append(score)

                json_line['sentence_scores'] = ordered_sentences
                out_file.write(json.dumps(json_line)+'\n')

    print(min(len_sent_post))
    print(np.mean(len_sent_pre))
    print(np.mean(len_sent_post))

    print('sentence ROUGE:')
    for score_name in score_names:
        print(f'{score_name} P: {np.mean([s[score_name].precision for s in scores]) * 100:.3f} '
              f'R: {np.mean([s[score_name].recall for s in scores]) * 100:.3f} '
              f'F1: {np.mean([s[score_name].fmeasure for s in scores]) * 100:.3f}')
