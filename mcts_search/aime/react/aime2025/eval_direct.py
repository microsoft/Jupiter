# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import argparse
import pandas as pd
from typing import Dict, List, Any, Optional
import numpy as np
from collections import defaultdict


def load_results(results_path: str) -> List[Dict[str, Any]]:
    """加载推理结果"""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def calculate_accuracy(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """计算准确率"""
    # 过滤掉有错误的结果
    valid_results = [r for r in results if "error" not in r and "is_correct" in r]

    # 新增：统计 ground_truth 不为空的项数
    non_empty_gt = sum(1 for r in valid_results if r.get("ground_truth") is not None)

    if not valid_results:
        return {
            "overall_accuracy": 0.0,
            "total_samples": 0,
            "correct_samples": 0,
            "error_samples": len(results) - len(valid_results),
            "non_empty_ground_truths": 0,
            "non_empty_predicted_answers": 0
        }

    correct_count = sum(1 for r in valid_results if r["is_correct"])
    total_count = len(valid_results)

    # 新增：统计预测答案不为空的项数
    non_empty_pred = sum(1 for r in valid_results if r.get("predicted_answer") is not None)

    return {
        "overall_accuracy": correct_count / total_count,
        "total_samples": total_count,
        "correct_samples": correct_count,
        "error_samples": len(results) - total_count,
        "non_empty_ground_truths": non_empty_gt,
        "non_empty_predicted_answers": non_empty_pred
    }


def calculate_accuracy_by_split(results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """按数据集split计算准确率"""
    split_results = defaultdict(list)
    
    # 按split分组
    for result in results:
        if "error" not in result and "is_correct" in result:
            split = result.get("split", "unknown")
            split_results[split].append(result)
    
    accuracy_by_split = {}
    for split, split_data in split_results.items():
        correct_count = sum(1 for r in split_data if r["is_correct"])
        total_count = len(split_data)
        
        accuracy_by_split[split] = {
            "accuracy": correct_count / total_count if total_count > 0 else 0.0,
            "total_samples": total_count,
            "correct_samples": correct_count
        }
    
    return accuracy_by_split


def analyze_errors(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """分析错误类型"""
    error_analysis = {
        "parsing_errors": 0,  # 无法提取答案的情况
        "computation_errors": 0,  # 计算错误
        "total_errors": 0,
        "examples": []
    }
    
    for result in results:
        if "error" in result:
            error_analysis["total_errors"] += 1
            continue
            
        if not result.get("is_correct", False):
            if result.get("predicted_answer") is None:
                error_analysis["parsing_errors"] += 1
            else:
                error_analysis["computation_errors"] += 1
            
            # 保存一些错误样例
            if len(error_analysis["examples"]) < 10:
                error_analysis["examples"].append({
                    "index": result.get("index", ""),
                    "question": result.get("question", "")[:200] + "..." if len(result.get("question", "")) > 200 else result.get("question", ""),
                    "ground_truth": result.get("ground_truth"),
                    "predicted_answer": result.get("predicted_answer"),
                    "generated_text": result.get("generated_text", "")[:300] + "..." if len(result.get("generated_text", "")) > 300 else result.get("generated_text", "")
                })
    
    return error_analysis


def generate_detailed_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """生成详细的评估报告"""
    report = {}
    
    # 整体准确率
    report["overall_metrics"] = calculate_accuracy(results)
    
    # 按split的准确率
    report["accuracy_by_split"] = calculate_accuracy_by_split(results)
    
    # 错误分析
    report["error_analysis"] = analyze_errors(results)
    
    # 答案长度统计
    valid_results = [r for r in results if "error" not in r and r.get("generated_text")]
    if valid_results:
        answer_lengths = [len(r["generated_text"]) for r in valid_results]
        report["answer_length_stats"] = {
            "mean": np.mean(answer_lengths),
            "std": np.std(answer_lengths),
            "min": np.min(answer_lengths),
            "max": np.max(answer_lengths),
            "median": np.median(answer_lengths)
        }
    
    return report


def compare_with_ground_truth(results_path: str, ground_truth_path: Optional[str] = None) -> Dict[str, Any]:
    """
    与ground truth进行详细比较
    
    Args:
        results_path: 推理结果路径
        ground_truth_path: 原始数据路径（可选，用于获取更多上下文）
    """
    results = load_results(results_path)
    
    comparison_report = {
        "summary": generate_detailed_report(results),
        "detailed_results": []
    }
    
    # 如果提供了原始数据路径，进行更详细的比较
    if ground_truth_path and ground_truth_path.endswith('.parquet'):
        try:
            df = pd.read_parquet(ground_truth_path)
            
            for result in results:
                if "error" not in result:
                    original_idx = result.get("original_index")
                    if original_idx is not None and original_idx < len(df):
                        original_data = df.iloc[original_idx].to_dict()
                        
                        detailed_result = {
                            "index": result.get("index"),
                            "question": result.get("question"),
                            "ground_truth": result.get("ground_truth"),
                            "predicted_answer": result.get("predicted_answer"),
                            "is_correct": result.get("is_correct"),
                            "original_prompt": original_data.get("prompt"),
                            "original_answer": original_data.get("extra_info", {}).get("answer", "")
                        }
                        comparison_report["detailed_results"].append(detailed_result)
        except Exception as e:
            print(f"Warning: Could not load ground truth file: {e}")
    
    return comparison_report


def print_report(report: Dict[str, Any]):
    """打印评估报告"""
    print("=" * 60)
    print("DIRECT INFERENCE EVALUATION REPORT")
    print("=" * 60)

    # 整体指标
    overall = report["summary"]["overall_metrics"]
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Accuracy: {overall['overall_accuracy']:.4f} ({overall['correct_samples']}/{overall['total_samples']})")
    print(f"  Ground truth non-empty: {overall['non_empty_ground_truths']}")
    print(f"  Predicted answer non-empty: {overall['non_empty_predicted_answers']}")
    if overall['error_samples'] > 0:
        print(f"  Error samples: {overall['error_samples']}")

    # 按 split 的结果
    if "accuracy_by_split" in report["summary"]:
        print(f"\nPERFORMANCE BY SPLIT:")
        for split, metrics in report["summary"]["accuracy_by_split"].items():
            print(f"  {split}: {metrics['accuracy']:.4f} ({metrics['correct_samples']}/{metrics['total_samples']})")

    # 错误分析
    if "error_analysis" in report["summary"]:
        error_analysis = report["summary"]["error_analysis"]
        print(f"\nERROR ANALYSIS:")
        print(f"  Parsing errors: {error_analysis['parsing_errors']}")
        print(f"  Computation errors: {error_analysis['computation_errors']}")
        print(f"  Total processing errors: {error_analysis['total_errors']}")

    # 逐条打印 ground_truth 与 predicted
    print("\nGROUND TRUTH vs PREDICTED:")
    for item in report.get("detailed_results", []):
        gt = item.get("ground_truth")
        pred = item.get("predicted_answer")
        print(f"  GT: {gt}  |  Pred: {pred}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Direct Inference Results')
    parser.add_argument('--results_path', type=str, default="../../mcts/aime_direct_results.json",
                        help='Path to the inference results JSON file')
    parser.add_argument('--ground_truth_path', type=str, default="../../dataset/aime2025/aime2025_vanilla.parquet",
                        help='Path to the original parquet file (optional)')
    parser.add_argument('--output_report', type=str, default=None,
                        help='Path to save detailed report (optional)')
    
    args = parser.parse_args()
    
    # 生成评估报告
    report = compare_with_ground_truth(args.results_path, args.ground_truth_path)
    
    # 打印报告
    print_report(report)
    
    # 保存详细报告
    if args.output_report:
        with open(args.output_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nDetailed report saved to: {args.output_report}")


if __name__ == "__main__":
    main()
