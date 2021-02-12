import argparse

def get_sa_args():

    args_dict = {
        "t_init": 3e2,
        "C": 3e-4,
        "fluency_weight": 3,
        "semantic_weight": 8,
        "max_steps": 20,
    }

    args = argparse.Namespace(**args_dict)
    return args