# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import json
import re
import pandas as pd
import copy
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


class ParallelDirectInferenceProcessor:
    def __init__(self, model_path: str, max_tokens: int = 4096):
        """
        初始化并行直接推理处理器
        
        Args:
            model_path: 模型路径
            max_tokens: 最大生成token数
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.llm = LLM(model=model_path)
        self.max_tokens = max_tokens
        
        # 设置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.7,  # 稍微增加一些随机性以获得多样化的结果
            top_p=1.0,
            max_tokens=max_tokens,
            skip_special_tokens=True,
        )

    def process_single_simulation(self, example: Dict[str, Any], simulation_id: int) -> Dict[str, Any]:
        """
        处理单个模拟
        
        Args:
            example: 单个数据样本，包含prompt和其他信息
            simulation_id: 模拟ID
            
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
            "simulation_id": simulation_id,
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


class BatchDirectInferenceManager:
    def __init__(self, example: Dict[str, Any], save_path: str, num_simulations: int = 5):
        """
        初始化批处理直接推理管理器
        
        Args:
            example: 单个数据样本
            save_path: 保存路径
            num_simulations: 模拟次数
        """
        self.example = example
        self.save_path = save_path
        self.num_simulations = num_simulations
        self.all_results = []
        
    def run_batch_simulations(self, processor: ParallelDirectInferenceProcessor):
        """
        批处理运行多个直接推理模拟
        """
        print(f"Starting {self.num_simulations} direct inference simulations...")
        
        # 顺序运行所有模拟
        for sim_id in range(self.num_simulations):
            print(f"Processing simulation {sim_id + 1}/{self.num_simulations}...")
            try:
                # 为每个模拟创建独立的样本副本
                example_copy = copy.deepcopy(self.example)
                result = processor.process_single_simulation(example_copy, sim_id)
                self.all_results.append(result)
                print(f"[Sim {sim_id}] Completed with answer: {result['predicted_answer']}")
            except Exception as e:
                print(f"Error in simulation {sim_id}: {e}")
                # 添加错误记录
                error_result = {
                    "simulation_id": sim_id,
                    "error": str(e),
                    "data_source": self.example.get("data_source", ""),
                    "split": self.example.get("extra_info", {}).get("split", "") if isinstance(self.example.get("extra_info"), dict) else "",
                    "index": self.example.get("extra_info", {}).get("index", "") if isinstance(self.example.get("extra_info"), dict) else "",
                }
                self.all_results.append(error_result)
        
        # 保存汇总结果
        self.save_aggregated_results()
        
        return self.all_results
    
    def save_aggregated_results(self):
        """保存汇总的结果"""
        # 分析结果
        unique_answers = self.get_unique_answers()
        answer_stats = self.get_answer_statistics()
        best_candidates = self.get_best_candidates()
        correct_count = sum(1 for r in self.all_results if r.get("is_correct", False))
        total_count = len([r for r in self.all_results if "error" not in r])
        
        aggregated_data = {
            "total_simulations": self.num_simulations,
            "completed_simulations": len([r for r in self.all_results if "error" not in r]),
            "error_simulations": len([r for r in self.all_results if "error" in r]),
            "correct_count": correct_count,
            "total_valid_count": total_count,
            "accuracy": correct_count / total_count if total_count > 0 else 0.0,
            "unique_answers": len(unique_answers),
            "answer_statistics": answer_stats,
            "best_candidates": best_candidates,
            "all_results": self.all_results,
            "unique_answer_details": unique_answers,
            "ground_truth": self.example.get("reward_model", {}).get("ground_truth", None),
            "question_info": {
                "data_source": self.example.get("data_source", ""),
                "split": self.example.get("extra_info", {}).get("split", ""),
                "index": self.example.get("extra_info", {}).get("index", ""),
                "question": self.example.get("extra_info", {}).get("question", ""),
            }
        }
        
        # 保存汇总结果
        aggregated_path = self.save_path.replace('.json', '_candidates.json')
        with open(aggregated_path, "w", encoding="utf-8") as f:
            json.dump(aggregated_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved aggregated results to {aggregated_path}")
        print(f"Summary: {len(self.all_results)} total results, {len(unique_answers)} unique answers")
        print(f"Accuracy: {correct_count}/{total_count} = {aggregated_data['accuracy']:.4f}" if total_count > 0 else "Accuracy: N/A")
        
        # 打印答案统计
        if answer_stats:
            print("Answer frequency statistics:")
            for answer, count in list(answer_stats.items())[:5]:  # 显示前5个最频繁的答案
                print(f"  Answer '{answer}': {count} occurrences")
        
        return aggregated_path
    
    def get_unique_answers(self):
        """获取唯一答案及其详细信息"""
        unique_answers = {}
        for result in self.all_results:
            if "error" in result:
                continue
            answer = result.get('predicted_answer')
            if answer is None:
                continue
            if answer not in unique_answers:
                unique_answers[answer] = {
                    "first_occurrence": result,
                    "occurrences": 1,
                    "simulation_ids": [result['simulation_id']]
                }
            else:
                unique_answers[answer]["occurrences"] += 1
                unique_answers[answer]["simulation_ids"].append(result['simulation_id'])
        
        return unique_answers
    
    def get_answer_statistics(self):
        """获取答案统计信息"""
        answer_counts = {}
        for result in self.all_results:
            if "error" in result:
                continue
            answer = result.get('predicted_answer')
            if answer is None:
                continue
            answer_counts[answer] = answer_counts.get(answer, 0) + 1
        
        # 按出现次数排序
        return dict(sorted(answer_counts.items(), key=lambda x: x[1], reverse=True))
    
    def get_best_candidates(self):
        """获取最佳候选答案"""
        valid_results = [r for r in self.all_results if "error" not in r and r.get('predicted_answer') is not None]
        if not valid_results:
            return []
        
        # 按照是否正确、然后按simulation_id排序
        sorted_results = sorted(
            valid_results, 
            key=lambda x: (not x.get('is_correct', False), x.get('simulation_id', 0))
        )
        
        return sorted_results[:10]  # 返回前10个最佳候选答案
    
    def save_individual_results(self):
        """保存每个模拟的单独结果"""
        for result in self.all_results:
            if "error" in result:
                continue
            sim_id = result['simulation_id']
            individual_path = self.save_path.replace('.json', f'_sim_{sim_id}.json')
            with open(individual_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)


class BatchParallelDirectInferenceProcessor:
    def __init__(self, model_path: str, max_tokens: int = 4096):
        """
        初始化批量并行直接推理处理器
        """
        self.processor = ParallelDirectInferenceProcessor(model_path, max_tokens)
    
    def process_problems_batch(self, problem_configs: List[Dict[str, Any]]):
        """
        为每个问题批处理多个直接推理实例
        """
        managers = []
        
        for config in problem_configs:
            manager = BatchDirectInferenceManager(
                example=config['example'],
                save_path=config['save_path'],
                num_simulations=config.get('num_simulations', 5)
            )
            managers.append(manager)
        
        # 为每个问题运行批处理模拟
        all_results = []
        for i, manager in enumerate(managers):
            print(f"\n=== Processing Problem {i+1}/{len(managers)} ===")
            results = manager.run_batch_simulations(self.processor)
            all_results.append(results)
            
            # 统计信息
            valid_results = [r for r in results if "error" not in r]
            correct_results = [r for r in valid_results if r.get("is_correct", False)]
            print(f"Problem {i+1}: {len(correct_results)}/{len(valid_results)} correct answers")
        
        print(f"\n=== Batch Processing Complete ===")
        print(f"Total problems processed: {len(managers)}")
        print(f"Results saved to respective output directories")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description='Parallel Direct Inference for AIME 2025')
    parser.add_argument('--parquet_path', type=str, 
                        default='../../dataset/aime2025/aime2025_vanilla.parquet',
                        help='Path to parquet file')
    parser.add_argument('--output_path', type=str, 
                        default='./aime_parallel_output',
                        help='Output directory for results')
    parser.add_argument('--model_path', type=str,
                        default='Qwen/Qwen2.5-7B-Instruct',
                        help='Path to the model')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of examples to process')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Maximum tokens to generate')
    parser.add_argument('--num_simulations', type=int, default=5, 
                        help='Number of parallel simulations per problem')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 读取数据
    df = pd.read_parquet(args.parquet_path)
    if args.limit:
        df = df.head(args.limit)
    
    print(f"Processing {len(df)} AIME examples with {args.num_simulations} simulations each...")
    print(f"Total simulations: {len(df) * args.num_simulations}")
    print(f"Output directory: {args.output_path}")
    
    # 准备问题配置
    problem_configs = []
    for idx, row in df.iterrows():
        # 将 pandas Series 转换为字典，确保数据类型正确
        example_dict = row.to_dict()
        
        # 特别处理 prompt 字段
        if "prompt" in example_dict and hasattr(example_dict["prompt"], 'tolist'):
            example_dict["prompt"] = example_dict["prompt"].tolist()
        
        save_path = os.path.join(args.output_path, f"aime_parallel_{example_dict.get('extra_info', {}).get('split', 'test')}_{idx}.json")
        
        config = {
            'example': example_dict,
            'save_path': save_path,
            'num_simulations': args.num_simulations
        }
        problem_configs.append(config)
    
    if not problem_configs:
        print("No valid examples found.")
        exit(0)
    
    # 创建批处理器并处理
    batch_processor = BatchParallelDirectInferenceProcessor(args.model_path, args.max_tokens)
    batch_processor.process_problems_batch(problem_configs)
    
    print("All parallel direct inference processing completed!")


if __name__ == "__main__":
    main()
