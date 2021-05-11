import json
import os
import random
import re


def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def create_just_seq(entry):
    just_keys = ["pipe_inp", "pipe_out"]
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


def write_output(filepath, data):
    with open(filepath, "a") as outfile:
        outfile.write(json.dumps(data)+"\n")


def get_ranks(entry, op):

    flag = 1
    best_just = 0
    while flag:
        best_just = input(f"Which justification is better?: ").strip()
        if best_just not in ['1', '2', '3']:
            print(f"FATAL: Valid value is either 1, 2 or 3.\n")
        else:
            best_just = int(best_just)
            flag = 0

    assert best_just > 0
    entry[op] = best_just

    return entry


def main(input_data, output_datapath, output_data_len):

    for idx, entry in enumerate(input_data[output_data_len:]):

        idx+=output_data_len

        entry = create_just_seq(entry)

        # Coverage
        print_data(idx, entry)
        print("Evaluation 1: Coverage - Contains important and salient information. Doesn't miss any important points that contribute to the fact-check. \n")
        get_ranks(entry, "coverage")

        # Non-redundancy
        print_data(idx, entry)
        print("Evaluation 2: Non-redundancy - Doesn't contain any information that is redundant/repeated/not relevant to the claim and the fact-check. \n")
        get_ranks(entry, "non-redundancy")

        # Non-contradictory
        print_data(idx, entry)
        print("Evaluation 3: Non-contradictory - Doesn't contain any information that is contradictory to the claim and the fact-check. \n")
        get_ranks(entry, "non-contradictory")

        # Fluency
        print_data(idx, entry)
        print("Evaluation 4: Fluency - The justification is more fluent, readable and create a coherent story.\n")
        get_ranks(entry, "fluency")

        # Overall
        print_data(idx, entry)
        print("Evaluation 5: Overall - A better overall justification based on above metrics. \n")
        get_ranks(entry, "overall")

        write_output(output_datapath, entry)

    return


if __name__=="__main__":

    input_datapath = "data/he_pub_task1.json"
    output_datapath = "eval_res_pub_task1.json"

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