import json
import os
import random
import re

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def create_just_seq(entry):
    just_keys = ["gold", "sa", "sa_pm"]
    entry["just_seq"] = random.sample(just_keys, len(just_keys))

    return entry

def print_data(idx, entry):
    cls()
    print(f"Your progress: {idx + 1}/{len(input_data)}\n")

    print("Claim:", entry["claim"])
    print("\n")
    print("Label:", entry["label"])
    print("\n")
    print("Justification 1:", entry[entry["just_seq"][0]])
    print("\n")
    print("Justification 2:", entry[entry["just_seq"][1]])
    print("\n")
    print("Justification 3:", entry[entry["just_seq"][2]])
    print("\n")

def write_output(filepath, data):
    with open(filepath, "a") as outfile:
        outfile.write(json.dumps(data)+"\n")


def get_ranks(entry, op):

    just_rank_seq = []
    for i in range(len(entry["just_seq"])):
        flag = 1
        while flag:
            just_rank = input(f"Rank of Justification {i+1}: ").strip()
            just_rank = float(just_rank)
            if just_rank not in [1.0, 2.0, 3.0]:
                print(f"FATAL: Rank should be a value among 1, 2 or 3.\n")
            else:
                flag = 0
                just_rank_seq.append(just_rank)

    assert len(just_rank_seq) == len(entry["just_seq"])
    entry[op] = just_rank_seq

    return entry

def main(input_data, output_datapath, output_data_len):

    for idx, entry in enumerate(input_data[output_data_len:]):

        idx+=output_data_len

        entry = create_just_seq(entry)

        # Coverage
        print_data(idx, entry)
        print("Evaluation 1: Coverage\n")
        get_ranks(entry, "coverage")

        # Non-redundancy
        print_data(idx, entry)
        print("Evaluation 2: Non-redundancy\n")
        get_ranks(entry, "non-redundancy")

        # Non-contradictory
        print_data(idx, entry)
        print("Evaluation 3: Non-contradictory\n")
        get_ranks(entry, "non-contradictory")

        # Overall
        print_data(idx, entry)
        print("Evaluation 4: Overall\n")
        get_ranks(entry, "overall")

        # Fluency
        print_data(idx, entry)
        print("Evaluation 5: Fluency\n")
        get_ranks(entry, "fluency")

        write_output(output_datapath, entry)

if __name__=="__main__":

    input_datapath = "data/he_data_liar_task1.json"
    output_datapath = "evaluation_results_task1.json"

    output_data_len = 0
    if os.path.exists(output_datapath):
        resume = input("Do you want to resume? (y/n): ")
        if resume.lower()=="y":
            output_data_len = len([json.loads(line) for line in open(output_datapath, 'r').readlines()])
            print(f"Found {output_data_len} entries.\n")
        else:
            os.remove(output_datapath)


    input_data = json.load(open(input_datapath))

    main(input_data, output_datapath, output_data_len)