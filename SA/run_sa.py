import json
import math
import argparse

from NLI_objective import nli_scorer
from editor import RobertaEditor
from generator_gpt import scorer_batch as gpt_scorer
from scoring_algos import SimulatedAnnealing

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences_path",
                        help="Path to pre-selected sentences.",
                        type=str,
                        default='data/top6_val_justifications.json')
    parser.add_argument("--dataset_path", help="Path to dataset", type=str,
                        default='data/top6_val_justifications.json')

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
                        help="Weight for lenght score.",
                        type=int, default=8)
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
    simulated_annealing = SimulatedAnnealing(editor,
                                             gpt_scorer,
                                             nli_scorer,
                                             sa_args)

    sa_outputs = []

    for i in range(0, len(dataset), sa_args.batch_size):
        batch_data = dataset[i: i + sa_args.batch_size]
        sa_outputs_batch = simulated_annealing.run(batch_data)
        for inp_just, out_just in zip(batch_data, sa_outputs_batch):
            print(inp_just)
            print(out_just)
            print("------------------------")

        sa_outputs += sa_outputs_batch
        break
