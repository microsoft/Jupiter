# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import re
import argparse
from glob import glob
from collections import Counter

import pandas as pd
from tqdm import tqdm


# ========= 统一的答案提取函数（与 AIME parallel 一致） =========
def extract_final_answer(text: str) -> str | None:
    """
    从模型输出中提取最终答案，兼容 AIME 的 #### 格式。
    返回 None 表示未找到有效答案。
    """
    # 优先匹配 #### 格式
    match = re.search(r"####\s*([\-+]?\d+(?:\.\d+)?)", text.replace(",", ""))
    if match:
        return match.group(1)
    
    # 备选：匹配最后一个数字
    numbers = re.findall(r"[\-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        return numbers[-1]
    
    return None


def main(args):
    # 1. 读取 parquet 文件
    df = pd.read_parquet(args.parquet_path, columns=["reward_model", "extra_info"])
    df = df.reset_index(drop=True)

    # 2. 收集所有候选 JSON 文件 (AIME Parallel candidates)
    pattern = os.path.join(args.candidates_dir, "*_candidates.json")
    cand_files = sorted(glob(pattern))

    results = []
    correct_cnt = 0

    for cand_path in tqdm(cand_files, desc="Evaluating AIME Parallel with Vote"):
        # 从文件名提取索引 (aime_parallel_train_123_candidates.json)
        basename = os.path.basename(cand_path)
        parts = basename.split('_')
        
        # 查找数字部分作为索引
        idx_str = None
        for part in reversed(parts):
            if part.replace('.json', '').replace('candidates', '').isdigit():
                idx_str = part.replace('.json', '').replace('candidates', '')
                break
        
        if idx_str is None:
            print(f"Skip {basename}: cannot parse index")
            continue

        try:
            idx = int(idx_str)
        except ValueError:
            print(f"Skip {basename}: cannot parse index from {idx_str}")
            continue

        if idx >= len(df):
            print(f"Skip {basename}: parquet only has {len(df)} rows")
            continue

        # 标准答案
        ground_truth = str(df.at[idx, "reward_model"]["ground_truth"]).strip()

        # 读取候选内容
        with open(cand_path, encoding="utf-8") as f:
            data = json.load(f)

        # 提取所有候选答案
        pred_answers = []
        for candidate in data.get("all_results", []):
            # 跳过错误的结果
            if "error" in candidate:
                continue
                
            predicted_answer = candidate.get("predicted_answer")
            if predicted_answer is not None:
                pred_answers.append(predicted_answer)
            else:
                # 尝试从生成文本中提取
                generated_text = candidate.get("generated_text", "")
                extracted = extract_final_answer(generated_text)
                if extracted is not None:
                    pred_answers.append(extracted)

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
    parser.add_argument("--parquet_path", default="../../dataset/aime2025/aime2025_vanilla.parquet",
                        help="Path to aime2025 parquet file")
    parser.add_argument("--candidates_dir", default="../../mcts/direct/aime_parallel_output",
                        help="Directory containing AIME Parallel *_candidates.json files")
    parser.add_argument("--output_csv", default="../../dataset/aime2025/result_aime_parallel_vote.csv",
                        help="Optional CSV path for detailed results")
    main(parser.parse_args())
