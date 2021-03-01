import json
import argparse
import pandas as pd
from nltk.tokenize import word_tokenize

from NLI_objective import NLIScorer
from editor import RobertaEditor
from generator_gpt import scorer_batch as gpt_scorer
from scoring_algos import SimulatedAnnealing


def get_dataset(scored_sentences_path, dataset_path, top_n):

    df = pd.read_csv(dataset_path, sep='\t', index_col=0)
    df = df.dropna()
    columns = ['dummy', 'id', 'statement', 'justification',
               'ruling_without_summary', 'label', 'just_tokenized',
               'ruling_tokenized', 'statement_tokenized', 'oracle_ids']
    df.columns = columns

    scored_sentences = [json.loads(line) for line in open(scored_sentences_path)]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences_path",
                        help="Path to pre-selected sentences.",
                        type=str,
                        default='../../results_serialized_val_filtered.jsonl')

    parser.add_argument("--dataset_path", help="Path to dataset", type=str,
                        default='../../liar_data/ruling_oracles_val.tsv')

    parser.add_argument("--seed", help="Random seed", type=int, default=33)
    parser.add_argument("--t_init",
                        help="Temperature initial value.",
                        type=float, default=3e-2)
    # TODO (Shailza): add help
    parser.add_argument("--C",
                        help="",
                        type=float, default=3e-4)
    parser.add_argument("--fluency_weight",
                        help="Weight for fluency score.",
                        type=int, default=3)
    parser.add_argument("--semantic_weight",
                        help="Weight for semantic similarity score.",
                        type=int, default=5)
    parser.add_argument("--length_weight",
                        help="Weight for length score.",
                        type=int, default=20)
    parser.add_argument("--nli_weight",
                        help="Weight for nli score.", type=int, default=8)
    parser.add_argument("--max_steps",
                        help="Max steps for running SA.", type=int, default=30)
    parser.add_argument("--top_n",
                        help="Number of top sentences to start SA with",
                        type=int, default=6)

    parser.add_argument("--batch_size",
                        help="Batch size.", type=int, default=5)

    sa_args = parser.parse_args()

    print(sa_args)

    dataset = get_dataset(sa_args.sentences_path, sa_args.dataset_path, sa_args.top_n)

    editor = RobertaEditor()
    editor.cuda()
    device = 'cpu'

    simulated_annealing = SimulatedAnnealing(editor,
                                             gpt_scorer,
                                             NLIScorer(device),
                                             sa_args)

    sa_outputs = []

    for i in range(0, len(dataset), sa_args.batch_size):
        batch_data = dataset[i: i + sa_args.batch_size]
        sa_outputs_batch = simulated_annealing.run(batch_data)

        for inp_just, out_just in zip(batch_data, sa_outputs_batch):

            temp_inp = []
            temp_out = []
            for sent in inp_just['scored_sentences']:
                temp_inp.append(" ".join(sent))

            for out in out_just:
                temp_out.append(" ".join(out))

            print("Input to SA:", " ".join(temp_inp))
            print("Output from SA:", " ".join(temp_out))
            print("*********************")

        sa_outputs += sa_outputs_batch
        break
