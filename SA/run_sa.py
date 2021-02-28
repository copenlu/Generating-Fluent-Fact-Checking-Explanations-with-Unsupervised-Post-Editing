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

    parser.add_argument("--dataset_path", help="Path to dataset", type=str,
                        default='data/top6_val_justifications.json')
    parser.add_argument("--seed", help="Random seed", type=int, default=33)
    parser.add_argument("--t_init",
                        help="Temperature initial value.",
                        type=float, default=3e-2)
    # TODO: add help
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

    sa_args = parser.parse_args()
    print(sa_args)

    editor = RobertaEditor()
    editor.cuda()

    simulated_annealing = SimulatedAnnealing(editor, gpt_scorer, nli_scorer,
                                             sa_args.t_init, sa_args.C,
                                             sa_args.fluency_weight,
                                             sa_args.semantic_weight,
                                             sa_args.length_weight,
                                             sa_args.nli_weight,
                                             sa_args.max_steps)

    data = json.load(open("../../top6_val_justifications.json", "r"))

    print("Length of data:", len(data))

    batch_size = 5
    num_batches = math.ceil(len(data) / float(batch_size))

    sa_outputs = []

    for i in range(num_batches):

        batch_data = data[batch_size * i:batch_size * (i + 1)]
        input_batch = list(
            zip(*[[i["id"], i["justifications"]] for i in batch_data]))
        sa_outputs_batch = simulated_annealing.run(input_batch)
        for inp_just, out_just in zip(input_batch[1], sa_outputs_batch):
            print(inp_just)
            print(out_just)
            print("------------------------")

        break
        sa_outputs += sa_outputs_batch
