# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

data = []
with open("./data.json", "r") as f:
    for line in f:
        data.append(eval(line))

model = 'Qwen2.5-7B-Instruct'
base_dir = "../../dsbench_result_tmp/save_performance"

# baseline_path = os.path.join(base_dir, "baseline")
# gt_path = os.path.join(base_dir, "GT")
baseline_path = "./save_performance/baseline"
gt_path = "./save_performance/GT"
save_path = os.path.join(base_dir, model)

task_complete = 0
scores = []

for line in data:
    name = line['name']
    gt_file = os.path.join(gt_path, name, "result.txt")
    bl_file = os.path.join(baseline_path, name, "result.txt")
    # modified to check for .result.txt
    pred_file = os.path.join(save_path, name, ".result.txt")

    # check if GT and baseline files exist
    if not (os.path.exists(gt_file) and os.path.exists(bl_file)):
        print(f"Missing GT or baseline for {name}, skip")
        continue

    with open(gt_file, "r") as f:
        gt = eval(f.read().strip())
    with open(bl_file, "r") as f:
        bl = eval(f.read().strip())

    if not os.path.exists(pred_file):
        scores.append(0)
        continue

    with open(pred_file, "r") as f:
        pre_content = f.read().strip()
    # result.txt maybe empty or all invalid, skip
    if not pre_content:
        scores.append(0)
        continue

    # parse the scores
    score_list = []
    for s in pre_content.split("||"):
        s = s.strip()
        if s == "" or s == "nan":
            continue
        try:
            val = float(s)
            score_list.append(val)
        except:
            continue

    if not score_list:
        scores.append(0)
        continue

    #pre = max(score_list)  # get max score

    # avoid division by zero
    if gt == bl:
        sc = 0
    else:
        sc = 0
        for score in score_list:
            sc = max(sc, (score - bl) / (gt - bl))  # calculate score using max score
    scores.append(sc)
    task_complete += 1

total_tasks = len(data)
print(f"Task completion rate is {task_complete}/{total_tasks} = {task_complete/total_tasks:.2%}")
print(f"Average performance score is {sum(scores)/total_tasks:.4f}")
