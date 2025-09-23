# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import json
import re
import pandas as pd
from typing import Dict, List, Any
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm


def extract_final_answer(text: str) -> str | None:
    """
    从模型输出中提取最终答案，兼容 GSM8K 的 #### 格式。
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


class DirectInference:
    def __init__(self, model_path: str, max_tokens: int = 4096):
        """
        初始化Direct Inference类
        
        Args:
            model_path: 模型路径
            max_tokens: 最大生成token数
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(model=model_path)
        self.max_tokens = max_tokens
        
        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.2,  # 确定性生成
            top_p=1.0,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )
    
    def process_single_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理单个样本
        
        Args:
            example: 单个数据样本，包含prompt和其他信息
            
        Returns:
            处理后的结果字典
        """
        # 提取prompt信息
        prompt = example["prompt"]
        
        # 处理不同类型的prompt数据
        if hasattr(prompt, 'tolist'):  # numpy.ndarray
            prompt = prompt.tolist()
        elif not isinstance(prompt, list):
            raise ValueError(f"Expected prompt to be a list or numpy.ndarray, got {type(prompt)}")
        
        if len(prompt) == 0:
            raise ValueError("Prompt list is empty")
        
        # 构建完整的输入
        messages = [{"role": "system", "content": "You are a helpful assistant."}] + prompt
        
        # 应用chat template
        input_text = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 生成回答
        outputs = self.llm.generate([input_text], self.sampling_params)
        generated_text = outputs[0].outputs[0].text
        
        # 提取最终答案
        predicted_answer = extract_final_answer(generated_text)
        
        # 获取ground truth
        ground_truth = example.get("reward_model", {}).get("ground_truth", None)
        
        # 判断是否正确
        is_correct = False
        if predicted_answer is not None and ground_truth is not None:
            try:
                # 将答案转换为浮点数进行比较
                pred_float = float(predicted_answer)
                gt_float = float(ground_truth)
                is_correct = abs(pred_float - gt_float) < 1e-6
            except ValueError:
                # 如果无法转换为数字，则进行字符串比较
                is_correct = str(predicted_answer).strip() == str(ground_truth).strip()
        
        return {
            "data_source": example.get("data_source", ""),
            "ability": example.get("ability", ""),
            "split": example.get("extra_info", {}).get("split", ""),
            "index": example.get("extra_info", {}).get("index", ""),
            "question": example.get("extra_info", {}).get("question", ""),
            "answer": example.get("extra_info", {}).get("answer", ""),
            "ground_truth": ground_truth,
            "input_text": input_text,
            "generated_text": generated_text,
            "predicted_answer": predicted_answer,
            "is_correct": is_correct
        }
    
    def process_dataset(self, parquet_path: str, output_path: str, limit: int = None):
        """
        处理整个数据集
        
        Args:
            parquet_path: parquet文件路径
            output_path: 输出文件路径
            limit: 限制处理的样本数量
        """
        # 读取数据
        df = pd.read_parquet(parquet_path)
        if limit:
            df = df.head(limit)
        
        print(f"Processing {len(df)} examples...")
        
        results = []
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            try:
                # 将 pandas Series 转换为字典，确保数据类型正确
                example_dict = row.to_dict()
                
                # 特别处理 prompt 字段
                if "prompt" in example_dict and hasattr(example_dict["prompt"], 'tolist'):
                    example_dict["prompt"] = example_dict["prompt"].tolist()
                
                result = self.process_single_example(example_dict)
                result["original_index"] = idx
                results.append(result)
            except Exception as e:
                print(f"Error processing example {idx}: {e}")
                # 添加错误记录
                results.append({
                    "original_index": idx,
                    "error": str(e),
                    "data_source": row.get("data_source", ""),
                    "split": row.get("extra_info", {}).get("split", "") if isinstance(row.get("extra_info"), dict) else "",
                    "index": row.get("extra_info", {}).get("index", "") if isinstance(row.get("extra_info"), dict) else "",
                })
        
        # 保存结果
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to {output_path}")
        
        # 计算准确率
        correct_count = sum(1 for r in results if r.get("is_correct", False))
        total_count = len([r for r in results if "error" not in r])
        
        if total_count > 0:
            accuracy = correct_count / total_count
            print(f"Accuracy: {correct_count}/{total_count} = {accuracy:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Direct Inference for AIME 2025')
    parser.add_argument('--parquet_path', type=str, 
                        default='../../dataset/aime2025/aime2025_vanilla.parquet',
                        help='Path to parquet file')
    parser.add_argument('--output_path', type=str, 
                        default='../../mcts/aime_direct_results.json',
                        help='Output path for results')
    parser.add_argument('--model_path', type=str,
                        default='Qwen/Qwen2.5-7B-Instruct',
                        help='Path to the model')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples to process')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum tokens to generate')
    
    args = parser.parse_args()
    
    # 创建推理器
    inferencer = DirectInference(args.model_path, args.max_tokens)
    
    # 处理数据集
    results = inferencer.process_dataset(args.parquet_path, args.output_path, args.limit)
    
    print("Direct inference completed!")


if __name__ == "__main__":
    main()
