import json
import os
import random
import re

def cls():
    os.system('cls' if os.name=='nt' else 'clear')

def print_data(idx, entry):
    cls()
    print(f"Your progress: {idx + 1}/{len(input_data)}\n")
    print("Claim:", entry["claim"])
    print("\n")
    print("Justification 1:", entry["just"])
    print("\n")

def write_output(filepath, data):
    with open(filepath, "a") as outfile:
        outfile.write(json.dumps(data)+"\n")

def get_label_binary(entry):

    flag = 1
    while flag:
        just_label = input(f"Label for justification: ").strip().lower()

        if just_label not in ["true", "false", "insufficient"]:
            print(f"FATAL: Rank should be a value among true, false, insufficient.\n")
        else:
            flag = 0
            entry["binary_label"] = just_label

    return entry

def get_label_sixway(entry):
    flag = 1
    while flag:
        just_label = input(f"Label for justification: ").strip().lower()

        if just_label not in ["pants-fire","barely-true", "half-true", "mostly-true", "false", "true", "insufficient"]:
            print(f"FATAL: Rank should be a value among pants-fire, barely-true, half-true, mostly-true, false, true, insufficient .\n")
        else:
            flag = 0
            entry["sixway_label"] = just_label

    return entry

def main(input_data, output_datapath, output_data_len):

    for idx, entry in enumerate(input_data[output_data_len:]):

        idx+=output_data_len

        # binary label
        print_data(idx, entry)
        get_label_binary(entry)
        get_label_sixway(entry)

        write_output(output_datapath, entry)



if __name__=="__main__":

    input_datapath = "data/he_data_liar_task2.json"
    output_datapath = "evaluation_results_task2.json"

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