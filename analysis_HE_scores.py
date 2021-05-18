import json
import click

@click.command()
@click.option("--eval_file1")
def main1(eval_file1):

    with open(eval_file1) as inp:

        eval_results = [json.loads(line.strip()) for line in inp]
        out_results = {
            "coverage": {"pipe_inp": 0, "pipe_out": 0, "both": 0},
            "non-redundancy": {"pipe_inp": 0, "pipe_out": 0, "both": 0},
            "non-contradictory": {"pipe_inp": 0, "pipe_out": 0, "both": 0},
            "fluency": {"pipe_inp": 0, "pipe_out": 0, "both": 0},
            "overall": {"pipe_inp": 0, "pipe_out": 0, "both": 0},
        }
        num_samples = float(len(eval_results))
        for line in eval_results:
            just_seq = line["just_seq"] + ["both"]
            for metric, data in out_results.items():
                out_results[metric][just_seq[line[metric] - 1]] += 1


        for metric, data in out_results.items():
            for k, v in data.items():
                print(f"{metric} of {k}: {round((v/num_samples)*100,2)}%")
            print("\n")

        # for metric, data in out_results.items():
        #     for k in ["pipe_inp", "pipe_out"]:
        #         num_obs = data[k] + data["both"]
        #         v = round((num_obs/num_samples)*100,2)
        #         print(f"{metric} of {k}: {v}%")
        #     print("\n")

@click.command()
@click.option("--eval_file2")
def main2(eval_file2):

    with open(eval_file2) as inp:

        eval_results = [json.loads(line.strip()) for line in inp]
        num_samples = float(len(eval_results))



        label_map_liar = {
            'true': '1',
            'false': '2',
            'mostly-true': '1',
            'pants-fire': '2',
            'half-true': '1',
            'barely-true': '2'
        }

        label_map_pub = {
            'true': ['1'],
            'unproven': ['3'],
            'false': ['2'],
            'mixture': ['1', '2'],
        }


        out_results = {
            "pipe_inp": {"match": 0, "not-match": 0, "insufficient": 0},
            "pipe_out": {"match": 0, "not-match": 0, "insufficient": 0},
        }
        for line in eval_results:
            if "liar" in eval_file2:
                true_label = int(label_map_liar[line["label"]])
            if "pub" in eval_file2:
                true_label = [int(i) for i in label_map_pub[line["label"]]]
            pred_label = int(line["binary_label"])
            just_type = line["just_type"]

            if "liar" in eval_file2:
                if pred_label<3:
                    if true_label==pred_label:
                        out_results[just_type]["match"] += 1
                    else:
                        out_results[just_type]["not-match"] += 1
                else:
                    out_results[just_type]["insufficient"] += 1
            else:
                if pred_label in true_label:
                    out_results[just_type]["match"] += 1
                else:
                    out_results[just_type]["not-match"] += 1


        print(out_results)

if __name__ == '__main__':

    main2()
