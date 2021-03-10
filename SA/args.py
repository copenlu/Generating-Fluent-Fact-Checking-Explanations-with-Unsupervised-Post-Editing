import argparse

def get_model_args():

    parser = argparse.ArgumentParser(description="model parameters")

    parser.add_argument("--sentences_path",
                        help="Path to pre-selected sentences.",
                        type=str,
                        default='/Users/jolly/PycharmProjects/COPENLU/results_serialized_val_filtered.jsonl')

    parser.add_argument("--dataset_path", help="Path to dataset", type=str,
                        default='/Users/jolly/PycharmProjects/COPENLU/liar_data/ruling_oracles_val.tsv')

    parser.add_argument("--seed", help="Random seed", type=int, default=33)

    parser.add_argument("--device_type", help="Type of device, CPU or GPU", type=str,
                        default='cpu')#gpu

    parser.add_argument("--sample",
                        help="Number of instances to sample from "
                             "original dataset.",
                        type=int, default=None)

    #SA Args
    parser.add_argument("--t_init",
                        help="Temperature initial value.",
                        type=float, default=3e-2)

    parser.add_argument("--C",
                        help="",
                        type=float, default=3e-4)

    parser.add_argument("--fluency_weight",
                        help="Weight for fluency score.",
                        type=int, default=8)

    parser.add_argument("--semantic_weight",
                        help="Weight for semantic similarity score.",
                        type=int, default=10)

    parser.add_argument("--length_weight",
                        help="Weight for length score.",
                        type=int, default=20)

    parser.add_argument("--nli_weight",
                        help="Weight for nli score.", type=int, default=12)

    parser.add_argument("--max_steps",
                        help="Max steps for running SA.", type=int, default=30)

    parser.add_argument("--top_n",
                        help="Number of top sentences to start SA with",
                        type=int, default=6)

    parser.add_argument("--batch_size",
                        help="Batch size.", type=int, default=3)

    parser.add_argument("--editor_model_id", help="Model-id for editor.",
                        type=str, default='roberta-base')

    parser.add_argument("--fluencyscorer_model_id",
                        help="Model-id for fluency scorer.",
                        type=str,
                        default='gpt2')

    args, unparsed = parser.parse_known_args()
    print(args)
    return args