# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import re
import argparse
from glob import glob

import pandas as pd
from tqdm import tqdm


# ========= 统一的答案提取函数（与之前一致） =========
def extract_final_answer(text: str) -> str | None:
    text = str(text).replace(",", "")
    m = re.search(r"####\s*([\-+]?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    m = re.search(r"Formatted answer:\s*([\-+]?\d+(?:\.\d+)?)", text)
    if m:
        return m.group(1)
    return None


def main(args):
    # 1. 读取 parquet
    df = pd.read_parquet(args.parquet_path, columns=["reward_model", "extra_info"])
    df = df.reset_index(drop=True)   # 确保索引就是 0,1,2...

    # 2. 收集所有候选文件
    pattern = os.path.join(args.candidates_dir, "*.json")
    cand_files = sorted(glob(pattern))

    results = []
    correct_cnt = 0

    for cand_path in tqdm(cand_files, desc="Evaluating"):
        # 从文件名提取 index
        basename = os.path.basename(cand_path)          # aime2025_train_123_candidates.json
        idx_str = basename.split('_')[2]                # 123
        idx = int(idx_str)

        if idx >= len(df):
            print(f"Skip {basename}: parquet only has {len(df)} rows")
            continue

        # 标准答案
        ground_truth = str(df.at[idx, "reward_model"]["ground_truth"]).strip()

        # 读取候选
        with open(cand_path, encoding="utf-8") as f:
            data = json.load(f)

        # 提取所有候选答案
        pred_answers = [
            extract_final_answer(c["final_answer"])
            for c in data.get("best_candidates")
        ]
        pred_answers = [p for p in pred_answers if p is not None]

        print(f"pred_answers are: {pred_answers}")

        # 判断是否命中
        hit = any(p == ground_truth for p in pred_answers)
        correct_cnt += hit

        results.append({
            "index": idx,
            "ground_truth": ground_truth,
            "pred_answers": pred_answers,
            "hit": hit,
            "file": basename,
        })

    # 3. 打印 & 保存
    acc = correct_cnt / len(results) if results else 0
    print(f"\nTotal: {len(results)},  Correct: {correct_cnt},  Acc: {acc:.4f}")

    if args.output_csv:
        pd.DataFrame(results).to_csv(args.output_csv, index=False, encoding="utf-8")
        print("Detailed results saved to", args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", default="./aime2025.parquet",
                        help="Path to aime2025.parquet")
    parser.add_argument("--candidates_dir", default="../../mcts/mcts_output",
                        help="Directory containing *train_*.json candidates")
    parser.add_argument("--output_csv", default="./aime2025_result_value.csv",
                        help="Optional CSV path for detailed results")
    main(parser.parse_args())
