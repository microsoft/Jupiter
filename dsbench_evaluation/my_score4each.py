# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import json
from tqdm import tqdm

data = []
with open("./data.json", "r") as f:
    for line in f:
        data.append(eval(line))

model = 'Qwen2.5-7B-Instruct'
gt_path = "./data/answers/"
python_path = "./evaluation/"
pred_path = f"../../Jupiter/mcts_search/inference/mcts_output_ds/{model}/"
save_path = f'../../dsbench_result_tmp/save_performance/{model}'

max_iteration = 50

for line in data:
    name = line['name']
    answer_file = os.path.join(gt_path, name, 'test_answer.csv')
    sub_dir = os.path.join(pred_path, name)
    if not os.path.isdir(sub_dir):
        print(f"{name}: {sub_dir} not found, skip")
        continue

    # check if only submission.csv exists
    single_csv = os.path.join(sub_dir, "submission.csv")
    if os.path.exists(single_csv):
        csv_files = ["submission.csv"]
    else:
        # else find all submission_{iteration_idx}_{sub_idx}.csv files
        csv_files = []
        for f in os.listdir(sub_dir):
            m = re.match(r"submission_(\d+)_(\d+)\.csv", f)
            if m:
                iteration_idx = int(m.group(1))
                if iteration_idx <= max_iteration - 1:
                    csv_files.append(f)
        if len(csv_files) == 0:
            print(f"{name}: no valid submission*.csv, skip")
            continue

    if not os.path.exists(os.path.join(save_path, name)):
        os.makedirs(os.path.join(save_path, name))

    scores = []
    for csv_file in sorted(csv_files):  # sort to ensure order
        pred_file = os.path.join(sub_dir, csv_file)
        if os.path.getsize(pred_file) == 0:
            print(f"{name}: {csv_file} is empty, skip")
            continue
        print(f"{name}: evaluate {csv_file}")

        tmp_result = os.path.join(save_path, name, "result_tmp.txt")
        cmd = (
            f"python {python_path}{name}_eval.py "
            f"--answer_file {answer_file} "
            f"--predict_file {pred_file} "
            f"--path {save_path} "
            f"--name {name} "
            f"> {tmp_result} 2>&1"
        )
        os.system(cmd)

        result_file = os.path.join(save_path, name, "result.txt")
        score = None
        if os.path.exists(result_file) and os.path.getsize(result_file) > 0:
            with open(result_file, "r") as rf:
                s = rf.read().strip()
                try:
                    float(s)
                    score = s
                except:
                    print(f"{name}: result.txt not a number, skip")
            os.remove(result_file)
        else:
            print(f"{name}: {csv_file} eval failed")
        if score is not None:
            scores.append(score)
    # aggregate scores
    if len(scores) > 0:
        result_path = os.path.join(save_path, name, ".result.txt")
        with open(result_path, "w") as f:
            f.write("||".join(scores))
        print(f"{name}: result collected, scores={scores}")
    else:
        print(f"{name}: no valid score, no result file generated.")
