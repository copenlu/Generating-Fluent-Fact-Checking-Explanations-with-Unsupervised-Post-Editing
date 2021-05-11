import json
import os


def cls():
    os.system('cls' if os.name=='nt' else 'clear')


def print_data(idx, entry):
    cls()
    print(f"Your progress: {idx + 1}/{len(input_data)}\n")
    print("Claim:", entry["claim"])
    print("\n")
    print("Justification:", entry["just"])
    print("\n")


def write_output(filepath, data):
    with open(filepath, "a") as outfile:
        outfile.write(json.dumps(data)+"\n")


def get_label_binary(entry):

    flag = 1
    while flag:
        print("Valid labels are 1 (true), 2 (false) and 3 (insufficient) \n")
        just_label = input(f"Label for justification: ").strip()

        if just_label not in ['1', '2', '3']:
            print(f"FATAL: Label should be a value among 1 (yes), 2 (no) or 3 (insufficient).\n")
        else:
            flag = 0
            just_label = int(just_label)
            entry["binary_label"] = just_label

    return entry


def main(input_data, output_datapath, output_data_len):

    for idx, entry in enumerate(input_data[output_data_len:]):

        idx+=output_data_len

        # binary label
        print_data(idx, entry)
        get_label_binary(entry)

        write_output(output_datapath, entry)


if __name__=="__main__":

    input_datapath = "data/he_pub_task2.json"
    output_datapath = "eval_res_pub_task2.json"

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