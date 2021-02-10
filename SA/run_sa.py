import json
import math

from scoring_algos import SimulatedAnnealing
from editor import RobertaEditor
from generator_gpt import scorer_batch as gpt_scorer
from args import get_sa_args

if __name__=="__main__":

    sa_args = get_sa_args()

    editor  = RobertaEditor()
    editor.cuda()

    simulated_annealing = SimulatedAnnealing(editor, gpt_scorer, sa_args.t_init, sa_args.C, sa_args.fluency_weight,
                                             sa_args.semantic_weight, sa_args.max_steps)

    data = json.load(open("../../org_justs_val.json", "r"))

    batch_size = 6
    num_batches = math.ceil(len(data)/float(batch_size))

    sa_outputs = []

    for i in range(num_batches):

        batch_data = data[batch_size*i:batch_size*(i+1)]
        input_batch = list(zip(*[[i["id"], i["justs"][:512]] for i in batch_data]))
        sa_outputs_batch = simulated_annealing.run(input_batch)
        print([(i, j) for i, j in zip(input_batch[1], sa_outputs_batch)])
        break
        sa_outputs += sa_outputs_batch
