import os
import json
import re
import argparse
from glob import glob
from collections import Counter

import pandas as pd
from tqdm import tqdm


# ========= 统一的答案提取函数 =========
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
    # 1. 读取 parquet 文件
    df = pd.read_parquet(args.parquet_path, columns=["reward_model", "extra_info"])
    df = df.reset_index(drop=True)

    # 2. 收集所有候选 JSON 文件
    pattern = os.path.join(args.candidates_dir, "*.json")
    cand_files = sorted(glob(pattern))

    results = []
    correct_cnt = 0

    for cand_path in tqdm(cand_files, desc="Evaluating"):
        # 从文件名提取索引
        basename = os.path.basename(cand_path)          # aime2025_train_123_candidates.json
        idx_str = basename.split('_')[2]                # 123
        idx = int(idx_str)

        if idx >= len(df):
            print(f"Skip {basename}: parquet only has {len(df)} rows")
            continue

        # 标准答案
        ground_truth = str(df.at[idx, "reward_model"]["ground_truth"]).strip()

        # 读取候选内容
        with open(cand_path, encoding="utf-8") as f:
            data = json.load(f)

        # 提取所有候选答案
        pred_answers = [
            c["final_answer"] if c["final_answer"] is not None else extract_final_answer(c["final_answer"])
            for c in data.get("best_candidates", [])
        ]
        pred_answers = [str(p).strip() for p in pred_answers if p is not None]

        print(f"[{basename}] pred_answers: {pred_answers}")

        # ====== 多数投票 ======
        final_answer = None
        if pred_answers:
            counter = Counter(pred_answers)
            final_answer, _ = counter.most_common(1)[0]  # 出现最多的答案

        hit = final_answer == ground_truth
        correct_cnt += hit

        results.append({
            "index": idx,
            "ground_truth": ground_truth,
            "pred_answers": pred_answers,
            "final_answer_voted": final_answer,
            "hit": hit,
            "file": basename,
        })

    # 3. 打印 & 保存结果
    acc = correct_cnt / len(results) if results else 0
    print(f"\nTotal: {len(results)},  Correct: {correct_cnt},  Acc: {acc:.4f}")

    if args.output_csv:
        pd.DataFrame(results).to_csv(args.output_csv, index=False, encoding="utf-8")
        print("Detailed results saved to", args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", default="/storage/v-yihaoliu/mcts/dataset/aime2025/aime2025.parquet",
                        help="Path to aime2025.parquet")
    parser.add_argument("--candidates_dir", default="/storage/v-yihaoliu/mcts/mcts_definition/aime/mcts/mcts_output",
                        help="Directory containing *train_*.json candidates")
    parser.add_argument("--output_csv", default="/storage/v-yihaoliu/mcts/dataset/aime2025/result_vote.csv",
                        help="Optional CSV path for detailed results")
    main(parser.parse_args())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet_path", default="./aime2025.parquet",
                        help="Path to aime2025.parquet")
    parser.add_argument("--candidates_dir", default="../../mcts/mcts_output",
                        help="Directory containing *train_*.json candidates")
    parser.add_argument("--output_csv", default="./aime2025_result_vote.csv",
                        help="Optional CSV path for detailed results")
    main(parser.parse_args())