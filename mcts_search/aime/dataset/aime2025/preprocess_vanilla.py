# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import re
from datasets import load_dataset

def extract_solution(answer):
    """Extract solution from AIME answer format"""
    # AIME answers are typically integers, so we just convert to string
    return str(answer)

# Instruction generation function for inference.py format
def instruction_generate(question_dict: dict) -> str:
    raw_question = question_dict['question']
    instruction_following = 'Let\'s think step by step and output the final answer after "####".'
    question = raw_question + " " + instruction_following
    return question

def make_map_fn(data_source):
    def process_fn(example, idx):
        problem_raw = example.pop("problem")
        answer_raw = example.pop("answer")
        solution = extract_solution(answer_raw)

        # Compose question dict for instruction
        question_dict = {
            "question": problem_raw,
            "constraints": "none",
            "format": "none",
            "file_name": "none"
        }

        data = {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": instruction_generate(question_dict)}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": solution},
            "extra_info": {
                "split": "train",  # AIME 2025 only has train split
                "index": idx,
                "answer": answer_raw,
                "question": problem_raw,
                "expected_answer": example.get("expected_answer", answer_raw),
                "url": example.get("url", ""),
                "year": example.get("year", 2025),
                "problem_id": example.get("problem_id", idx),
            },
        }
        return data

    return process_fn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="aime_dataset", help="Directory to save parquet files")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    data_source = "yentinglin/aime_2025"
    dataset = load_dataset(data_source)

    train_dataset = dataset["train"].map(function=make_map_fn(data_source), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.local_dir, "aime2025_vanilla.parquet"))
    
    print(f"Processed {len(train_dataset)} AIME 2025 problems")
    print(f"Saved to {os.path.join(args.local_dir, 'aime2025_vanilla.parquet')}")
