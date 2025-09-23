# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import argparse
import re
import os

def read_jsonl(file_name):
    data = []
    with open(file_name, "r", encoding="utf-8") as f:
        for ln in f:
            data.append(json.loads(ln.rstrip("\n")))
    return data


def write_jsonl(dict_list, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        for d in dict_list:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

def extract_format(text):
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, text)
    return [m[0] for m in matches], [m[1] for m in matches]

# add by myself
def normalize_list_string(s):
    if not isinstance(s, str):
        return [str(s)]
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return [x.strip().strip('"').strip("'") for x in s.split(",") if x.strip()]

# add by myself
def is_empty_value(val):
    if val is None:
        return True
    if isinstance(val, str):
        v = val.strip()
        return v in ['', '[]', '{}', 'null', 'None', 'nan', 'N/A', 'NaN']
    if isinstance(val, (list, dict)) and len(val) == 0:
        return True
    return False

def is_equal(response, label):
    if response == label:
        return True
    if is_empty_value(response) and is_empty_value(label):
        return True
    try:
        return abs(float(response) - float(label)) < 1e-6
    except Exception:
        pass
    try:
        resp_list = normalize_list_string(response)
        label_list = normalize_list_string(label)
        if len(resp_list) != len(label_list):
            return False
        for r, l in zip(resp_list, label_list):
            try:
                if abs(float(r) - float(l)) > 1e-6:
                    return False
            except Exception:
                if r != l:
                    return False
        return True
    except Exception:
        return False

def evaluate_responses(labels, responses):
    results = []
    for lab in labels:
        qid = lab["id"]
        lab_ans  = {k.strip(): v.strip() for k, v in lab["common_answers"]}
        resp_item = next((r for r in responses if str(r["id"]) == str(qid)), None)

        if resp_item and resp_item["response"]:
            names, answers = extract_format(resp_item["response"])
            pred_ans = {k.strip(): v.strip() for k, v in zip(names, answers)}
            print(f"qid: {qid}\n  lab_ans: {lab_ans}\n  pred_ans: {pred_ans}")
            correctness = {k: is_equal(pred_ans.get(k), lab_ans[k]) for k in lab_ans}
            print(f"  correctness: {correctness}\n")
        else:
            pred_ans = {}
            correctness = {k: False for k in lab_ans}

        results.append(
            {
                "id": qid,
                "label_answers": lab_ans,
                "predicted_answers": pred_ans,
                "correctness": correctness,
            }
        )
    return results

def read_concepts_from_file(path):
    concepts = {}
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            d = json.loads(ln.rstrip("\n"))
            concepts[d["id"]] = d["concepts"]
    return concepts


def analyze_concepts_accuracy(results, concepts_data):
    acc = {}
    for r in results:
        qid = r["id"]
        if qid not in concepts_data:
            continue
        for c in concepts_data[qid]:
            acc.setdefault(c, {"total": 0, "correct": 0})
            acc[c]["total"] += 1
            if all(r["correctness"].values()):
                acc[c]["correct"] += 1
    return {k: round(v["correct"] / v["total"], 4) for k, v in acc.items()}


def analyze_concepts_count_accuracy(results, concepts_data):
    count_acc = {}
    multi_acc = {"total": 0, "correct": 0}

    for r in results:
        qid = r["id"]
        if qid not in concepts_data:
            continue
        cnt = len(concepts_data[qid])
        count_acc.setdefault(cnt, {"total": 0, "correct": 0})
        count_acc[cnt]["total"] += 1
        if all(r["correctness"].values()):
            count_acc[cnt]["correct"] += 1

        if cnt >= 2:
            multi_acc["total"] += 1
            if all(r["correctness"].values()):
                multi_acc["correct"] += 1

    count_acc = {k: round(v["correct"] / v["total"], 4) for k, v in count_acc.items()}
    multi_rate = (
        round(multi_acc["correct"] / multi_acc["total"], 4)
        if multi_acc["total"] > 0
        else None
    )
    return count_acc, multi_rate

def acc_by_question(results):
    return round(
        sum(all(r["correctness"].values()) for r in results) / len(results), 4
    )


def acc_by_sub_question(results):
    total = sum(len(r["correctness"]) for r in results)
    correct = sum(sum(r["correctness"].values()) for r in results)
    return round(correct / total, 4) if total else 0


def acc_prop_sub_question(results):
    score = 0
    for r in results:
        n = len(r["correctness"])
        if n:
            score += sum(r["correctness"].values()) / n
    return round(score / len(results), 4) if results else 0


def main():
    parser = argparse.ArgumentParser("Evaluate Data-Agent answers (support majority vote)")
    parser.add_argument(
        "--questions_file_path",
        default="data/da-dev-questions.jsonl",
        type=str,
    )
    parser.add_argument(
        "--labels_file_path",
        default="data/da-dev-labels.jsonl",
        type=str,
    )
    parser.add_argument(
        "--response_files_path",
        default="./inference_output_vote/Llama-3.1-8B-Instruct_5",
        type=str,
    )
    args = parser.parse_args()

    os.makedirs("eval_outputs", exist_ok=True)

    labels = read_jsonl(args.labels_file_path)

    responses = []
    maj_path = os.path.join(args.response_files_path, "majority_results.json")

    if os.path.exists(maj_path):
        majority_data = json.load(open(maj_path, "r", encoding="utf-8"))
        for qid, item in majority_data.items():
            try:
                qid_val = int(qid)
            except ValueError:
                qid_val = qid
            responses.append({"id": qid_val, "response": item["majority_answer"]})
        print(f"[Info] Loaded majority_results.json → {len(responses)} responses.")
    else:

        for lab in labels:
            qid = lab["id"]
            fpath = os.path.join(args.response_files_path, f"{qid}.json")
            if os.path.exists(fpath):
                data = json.load(open(fpath, "r", encoding="utf-8"))
                responses.append({"id": qid, "response": data.get("formatted_answer", "")})
            else:
                responses.append({"id": qid, "response": ""})
        print(f"[Info] Loaded {len(responses)} responses from individual files.")


    results = evaluate_responses(labels, responses)

    q_acc   = acc_by_question(results)
    sq_acc  = acc_by_sub_question(results)
    prop_acc = acc_prop_sub_question(results)

    print(f"\n=== Accuracy ===")
    print(f"By Question                       : {q_acc:.2%}")
    print(f"Proportional by Sub-Question      : {prop_acc:.2%}")
    print(f"By Sub-Question (micro-average)   : {sq_acc:.2%}")


    concepts_data = read_concepts_from_file(args.questions_file_path)
    c_acc = analyze_concepts_accuracy(results, concepts_data)
    cnt_acc, multi_acc = analyze_concepts_count_accuracy(results, concepts_data)


    out_name = os.path.basename(args.response_files_path.rstrip("/"))
    out_path = f"eval_outputs/{out_name}_evaluation_analysis.json"

    json.dump(
        {
            "Accuracy by Question": q_acc,
            "Accuracy Proportional by Sub-Question": prop_acc,
            "Accuracy by Sub-Question": sq_acc,
            "Concept Accuracy Analysis": c_acc,
            "Concept Count Accuracy Analysis": cnt_acc,
            "Accuracy for Questions with >=2 Concepts": multi_acc,
        },
        open(out_path, "w", encoding="utf-8"),
        indent=4,
        ensure_ascii=False,
    )

    print(f"\nDetail results saved → {out_path}")


if __name__ == "__main__":
    main()
