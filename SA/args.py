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
                        default='cpu')

    parser.add_argument("--sample",
                        help="Number of instances to sample from "
                             "original dataset.",
                        type=int, default=None)

    #SA Args
    parser.add_argument("--t_init",
                        help="Temperature initial value.",
                        type=float, default=60000)

    parser.add_argument("--C",
                        help="scale of temp",
                        type=float, default=300)

    parser.add_argument("--fluency_weight",
                        help="Weight for fluency score.",
                        type=float, default=1.4)

    parser.add_argument("--length_weight",
                        help="Weight for length score.",
                        type=float, default=1.25)

    parser.add_argument("--semantic_weight_keywords",
                        help="Weight for semantic similarity score.",
                        type=float, default=1.0)

    parser.add_argument("--semantic_weight_sentences",
                        help="Weight for sentence semantic similarity score.",
                        type=float, default=1.1)

    parser.add_argument("--named_entity_score_weight",
                        help="Weight for entity scorer.",
                        type=float, default=0.95)
   
    parser.add_argument("--insert_th",
                        help="accept threshold for insert",
                        type=float, default=0.99)

    parser.add_argument("--reorder_th",
                        help="accept threshold for reorder",
                        type=float, default=0.95)

    parser.add_argument("--delete_th",
                        help="accept threshold for delete",
                        type=float, default=0.97)

    parser.add_argument("--algo_type",
                        help="Type of iterative algorithm.", type=str, default="noSA")
    
    parser.add_argument("--dataset_name",
                        help="Name of the dataset.", type=str, default="pub_health")
 
    parser.add_argument("--outfile",
                                help="Name of the file in which output is stored.", type=str, default="")

    parser.add_argument("--max_steps",
                        help="Max steps for running SA.", type=int, default=200)

    parser.add_argument("--top_n",
                        help="Number of top sentences to start SA with",
                        type=int, default=6)

    parser.add_argument("--batch_size",
                        help="Batch size.", type=int, default=1)

    parser.add_argument("--min_length_of_edited_sent",
                        help="Minimum size of a justification. Stopping Criterion", type=int, default=40)

    parser.add_argument("--editor_model_id", help="Model-id for editor.",
                        type=str, default='roberta-base')

    parser.add_argument("--fluencyscorer_model_id",
                        help="Model-id for fluency scorer.",
                        type=str,
                        default='gpt2')

    args, unparsed = parser.parse_known_args()
    print(args)
    return args
