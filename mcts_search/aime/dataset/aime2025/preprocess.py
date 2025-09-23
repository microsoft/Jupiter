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
    constraints = "output the final answer after \"####\" after the \"Formatted answer:\" without other anything"

    return f"""Please complete the following data analysis task as best as you can. On each turn, you can use the python code sandbox to execute the python code snippets you generate and get the output.

### Format to follow

On each turn, you will generate a Thought and an Action based on the current context. Here is the format to follow on each turn: 

Thought: your reasoning about what to do next based on the current context.
Action: the code snippet you want to execute in the python code sandbox. The code should be wrapped with triple backticks like ```python ... your code ... ```.

Then you will get the Observation based on the execution results from user and generate a new Thought and Action based on the new context. Once you have enough information to answer the question, finish with:

Thought: ... your final thought about the question... 
Formatted answer: the formatted answer to the original input question, should follow the constraints and format provided in the input question. 

### Important Notes
- Do **NOT** include Observation or execution results. NEVER complete or predict the Observation part in your response. Cease generating further after generating the Action part on each turn.
- If you have any files outputted, write them to "./"
- For all outputs in code, THE print() function MUST be called. For example, when you need to call df.head(), please use print(df.head()).

Let's begin the task.

### Task
Question: {raw_question}  
Constraints: {constraints}  
"""

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
    parser.add_argument("--local_dir", default="aime2025_parquet", help="Directory to save parquet files")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    data_source = "yentinglin/aime_2025"
    dataset = load_dataset(data_source)

    # AIME 2025 only has train split, no need to split into train/test
    train_dataset = dataset["train"].map(function=make_map_fn(data_source), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.local_dir, "aime2025.parquet"))
    
    print(f"Processed {len(train_dataset)} AIME 2025 problems")
    print(f"Saved to {os.path.join(args.local_dir, 'aime2025.parquet')}")
