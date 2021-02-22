import argparse

def get_sa_args():

    args_dict = {
        "seed": 33,
        "t_init": 3e-2,
        "C": 3e-4,
        "fluency_weight": 3,
        "semantic_weight": 5,
        "length_weight":8,
        "nli_weight": 8,
        "max_steps": 30,
    }

    args = argparse.Namespace(**args_dict)
    return args