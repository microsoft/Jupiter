# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import os
import json
import random
import numpy as np
from transformers import AutoTokenizer
import base64
from hashlib import md5
from vllm import LLM, SamplingParams
from sandbox_fusion import run_concurrent, RunJupyterRequest, set_endpoint, run_jupyter

from node import MCTSNode
from check_result import check_correct
from config import (
    MAX_SIMULATIONS, 
    EXPANSION_NUM,  
    MAX_TOKENS, 
    NEGETIVE_REWARD,
    POSITIVE_REWARD, 
    NEUTRAL_REWARD,
    ERROR_THRESHOLD, 
    MAX_DEPTH,
    TIME_LIMIT,
    instruction_generate,
)

from config import (
    INPUT_EXCEEDS_TOKEN_LIMIT,
    PARSABLE_TEXT_FAIL,
    CODE_BLOCK_FAIL,
    CODE_BLOCK_EMPTY,
    SANDBOX_ERROR,
    SANDBOX_EXECUTION_FAILED,
    TOO_MANY_CODE_ERRORS,
    FAIL_WITHIN_LIMITED_STEPS,
)

class MCTSTree:
    def __init__(self, initial_input, ground_truth, files_dict, save_path, is_test_mode=False):
        self.root = self.create_node()
        self.root.state["messages"] = initial_input
        
        self.ground_truth = ground_truth
        self.files_dict = files_dict
        self.save_path = save_path
        self.is_test_mode = is_test_mode  # 添加测试模式参数
        self.candidate_answers = []  # 存储候选答案

    def create_node(self, parent=None):
        return MCTSNode(
            parent=parent,
        )

    def eval_final_answer(self, node):
        if node.state.get("final_answer") in [
            INPUT_EXCEEDS_TOKEN_LIMIT,
            PARSABLE_TEXT_FAIL,
            CODE_BLOCK_FAIL,
            CODE_BLOCK_EMPTY,
            SANDBOX_ERROR,
            SANDBOX_EXECUTION_FAILED,
            TOO_MANY_CODE_ERRORS,
            FAIL_WITHIN_LIMITED_STEPS,
        ]:
            node.update_recursive(NEGETIVE_REWARD, self.root)
            return

        # 处理有最终答案的情况
        final_answer = node.state.get("final_answer")
        if final_answer is None:
            print(f"Warning: Final answer is None for node {node.tag}")
            node.update_recursive(NEGETIVE_REWARD, self.root)
            return

        if self.ground_truth:
            # 训练模式：有ground_truth，进行正确性检查
            correct_sub_problem = check_correct(self.ground_truth, final_answer)
            reward = POSITIVE_REWARD * correct_sub_problem / self.ground_truth.count("@") if correct_sub_problem > 0 else NEGETIVE_REWARD
            node.update_recursive(reward, self.root)
        elif self.is_test_mode:
            # 测试模式：没有ground_truth，收集候选答案并给予中性奖励
            # print(f"Test mode: Final answer collected for node {node.tag}")
            self.collect_candidate_answer(node, final_answer)
            node.update_recursive(NEUTRAL_REWARD, self.root)  # 中性奖励，不影响搜索
        else:
            raise ValueError("Invalid state: ground_truth is None and is_test_mode is False")

    # 遍历树上所有节点，收集到该节点为止的 messages 拼接、value、is_terminal
    def collect_all_node_contexts(self):
        all_infos = []
        
        def dfs(node):
            # 从根节点到当前节点构建完整的消息历史
            messages = []
            path_nodes = []
            
            # 从当前节点向上追溯到根节点
            current = node
            while current is not None:
                path_nodes.append(current)
                current = current.parent
            
            # 反转路径，从根节点到当前节点的顺序
            path_nodes.reverse()
            
            # 按顺序拼接所有messages
            for n in path_nodes:
                if "messages" in n.state:
                    messages.extend(n.state["messages"])
            
            # 保存信息
            all_infos.append({
                "tag": node.tag,
                "depth": node.depth,
                "messages": messages,
                "is_terminal": node.is_terminal,
                "visit_count": node.visit_count,
                "value_sum": node.value_sum,
                "value": node.value,
            })
            
            # 递归处理所有子节点
            for child in node.children:
                dfs(child)
        
        # 从根节点开始遍历
        dfs(self.root)
        return all_infos

    def save_all_node_contexts(self, save_path):
        all_infos = self.collect_all_node_contexts()
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(all_infos, f, ensure_ascii=False, indent=2)
        
    def selection(self):
        node = self.root

        while node.has_children() and not node.is_terminal:
            next_node = self.select_child(node)
            if next_node is None:
                node.is_terminal = True
                break
            node = next_node

        return None if node.is_terminal else node

    def select_child(self, node):
        best_value = -float("inf")
        best_childs = []

        for child in node.children:
            if child.is_terminal:
                continue

            puct_value = child.puct()
            if puct_value == best_value:
                best_childs.append(child)
            elif puct_value > best_value:
                best_value = puct_value
                best_childs = [child]

        return random.choice(best_childs) if best_childs else None

    def get_input_prompt(self, node, tokenizer):
        """Get the full input prompt for a node"""
        input_messages = []
        tmp_node = node
        while tmp_node is not None:
            if "messages" in tmp_node.state:
                input_messages = tmp_node.state["messages"] + input_messages
            tmp_node = tmp_node.parent
        
        return tokenizer.apply_chat_template(input_messages, tokenize=False, add_generation_prompt=True)

    def create_child_without_execution(self, output_text, node, prior_prob, idx):
        """Create child node without executing code (for batch processing)"""
        new_node = self.create_node(parent=node)
        new_node.tag = f"{node.tag}.{idx}"
        new_node.depth = node.depth + 1
        new_node.prior = prior_prob

        if "Formatted answer:" in output_text:
            formatted = output_text.split("Formatted answer:")[-1].strip()
            print(f"formatted answer found: {formatted}")
            new_node.is_terminal = True
            new_node.state["messages"] = [
                {"role": "assistant", "content": output_text},
            ]
            new_node.state["final_answer"] = formatted
            self.eval_final_answer(new_node)
            node.children.append(new_node)
            return new_node, None  # No code execution needed

        if "thought:" not in output_text.lower() and "action:" not in output_text.lower():
            print(f"Invalid output format: {output_text}")
            #new_node.is_terminal = True
            new_node.state["messages"] = [
                {"role": "assistant", "content": output_text},
                {"role": "user", "content": "Observation: You response should contain both 'Thought:' and 'Action:' sections."}
            ]
            new_node.state["final_answer"] = PARSABLE_TEXT_FAIL
            self.eval_final_answer(new_node)
            node.children.append(new_node)
            return new_node, None

        code_match = re.search(r'```python\s*(.*?)```', output_text, re.DOTALL)

        if code_match:
            code_block = code_match.group(1).strip()
            if not code_block:
                print(f"Empty code block in output: {output_text}")
                #new_node.is_terminal = True
                new_node.state["messages"] = [
                    {"role": "assistant", "content": output_text},
                    {"role": "user", "content": "Observation: You need to provide a non-empty python code snippet wrapped in triple backticks."}
                ]
                new_node.state["final_answer"] = CODE_BLOCK_EMPTY
                self.eval_final_answer(new_node)
                node.children.append(new_node)
                return new_node, None
            else:
                new_node.state["code"] = code_block
                
                # extract the code field from the root node to the now node
                codes = []
                tmp_node = new_node
                while tmp_node is not None:
                    if "code" in tmp_node.state:
                        codes.append(tmp_node.state["code"])
                    tmp_node = tmp_node.parent
                codes.reverse()

                # Return the node and execution request for batch processing
                exec_request = RunJupyterRequest(
                    cells=codes,
                    total_timeout=TIME_LIMIT,
                    files=self.files_dict,
                )
                node.children.append(new_node)
                return new_node, exec_request
        else:
            print(f"No code block found in output: {output_text}")
            #new_node.is_terminal = True
            new_node.state["messages"] = [
                {"role": "assistant", "content": output_text},
                {"role": "user", "content": "Observation: You need to provide a valid python code snippet wrapped in triple backticks."}
            ]
            new_node.state["final_answer"] = CODE_BLOCK_FAIL
            self.eval_final_answer(new_node)
            node.children.append(new_node)
            return new_node, None

    def process_execution_result(self, new_node, output_text, exec_result):
        """Process the execution result for a node"""
        code_output = "Observation:\n"
        have_error = False

        try:
            exec_result = json.loads(exec_result.json())

            if exec_result.get("status") != "Success":
                print(f"Execution failed with execution result: {exec_result}")
                status_info = exec_result.get("driver", {}).get("status", "Unknown")
                code_output += f"Code execution failed with status: {status_info}"

                #new_node.is_terminal = True
                new_node.state["messages"] = [
                    {"role": "assistant", "content": output_text},
                    {"role": "user", "content": code_output}
                ]
                new_node.state["final_answer"] = SANDBOX_EXECUTION_FAILED
                self.eval_final_answer(new_node)
                return

            if exec_result.get("cells"):
                last_output = exec_result["cells"][-1]
                if last_output["stdout"] != "":
                    code_output += last_output["stdout"] + "\n"
                if last_output["stderr"] != "":
                    code_output += last_output["stderr"] + "\n"
                if last_output["error"] != []:
                    have_error = True
                    for err in last_output["error"]:
                        if "ename" in err and "evalue" in err:
                            code_output += f"{err['ename']}: {err['evalue']}\n"
                        # if "traceback" in err:
                        #     code_output += "\n".join(err["traceback"]) + "\n"
        except Exception as e:
            code_output += f"Sandbox error: {str(e)}"
            new_node.state["messages"] = [
                {"role": "assistant", "content": output_text},
                {"role": "user", "content": code_output}
            ]
            new_node.state["final_answer"] = SANDBOX_ERROR
            self.eval_final_answer(new_node)
            return

        new_node.state["messages"] = [
            {"role": "assistant", "content": output_text},
            {"role": "user", "content": code_output}
        ]

        if have_error:
            new_node.consecutive_errors = getattr(new_node.parent, 'consecutive_errors', 0) + 1
            if new_node.consecutive_errors > ERROR_THRESHOLD:
                new_node.is_terminal = True
                new_node.state["final_answer"] = TOO_MANY_CODE_ERRORS
                self.eval_final_answer(new_node)
                return
        
        if not new_node.is_terminal and new_node.depth > MAX_DEPTH:
            new_node.is_terminal = True
            new_node.state["final_answer"] = FAIL_WITHIN_LIMITED_STEPS
            self.eval_final_answer(new_node)

    def collect_candidate_answer(self, node, final_answer):
        """收集候选答案信息"""
        candidate_info = {
            "node_tag": node.tag,
            "node_depth": node.depth,
            "final_answer": final_answer,
            "visit_count": node.visit_count,
            "value_sum": node.value_sum,
            "value": node.value,
            "path_to_answer": self.get_node_path_messages(node)
        }
        self.candidate_answers.append(candidate_info)
        #print(f"Collected candidate answer for node {node.tag}: {final_answer[:100]}...")

    def get_node_path_messages(self, node):
        """获取从根节点到当前节点的完整消息路径"""
        messages = []
        path_nodes = []
        
        # 从当前节点向上追溯到根节点
        current = node
        while current is not None:
            path_nodes.append(current)
            current = current.parent
        
        # 反转路径，从根节点到当前节点的顺序
        path_nodes.reverse()
        
        # 按顺序拼接所有messages
        for n in path_nodes:
            if "messages" in n.state:
                messages.extend(n.state["messages"])
        
        return messages

    def save_candidate_answers(self):

        save_path = os.path.join(os.path.dirname(self.save_path), self.save_path.split('/')[-1].replace('.json', '_candidates.json'))
        
        candidates_data = {
            "total_candidates": len(self.candidate_answers),
            "unique_candidates": len(self.get_all_unique_answers()),
            "best_candidates": self.get_best_candidate_answers(),
            "all_candidates": self.candidate_answers,
            "is_test_mode": self.is_test_mode,
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(candidates_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(self.candidate_answers)} candidate answers to {save_path}")
        return save_path

    def get_best_candidate_answers(self):
        """获取评分最高的候选答案"""
        if not self.candidate_answers:
            return []
        
        # 按照 value_sum / visit_count = q value排序，q value 越大越靠前
        sorted_candidates = sorted(
            self.candidate_answers, 
            key=lambda x: (x['value_sum'] / x['visit_count'] if x['visit_count'] > 0 else -1),
            reverse=True
        )

        return sorted_candidates

    def get_all_unique_answers(self):
        """获取所有唯一的候选答案"""
        unique_answers = {}
        for candidate in self.candidate_answers:
            answer = candidate['final_answer']
            if answer not in unique_answers:
                unique_answers[answer] = candidate
            else:
                # 如果答案相同，保留visit_count更高的
                if candidate['visit_count'] > unique_answers[answer]['visit_count']:
                    unique_answers[answer] = candidate
        
        return list(unique_answers.values())

class BatchMCTSProcessor:
    def __init__(self, llm, tokenizer, sampling_params):
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
    
    def process_trees_concurrently(self, mcts_trees):
        """Process multiple MCTS trees concurrently"""
        active_trees = list(mcts_trees)
        
        for simulation_idx in range(MAX_SIMULATIONS):
            if not active_trees:
                break
                
            print(f"Simulation {simulation_idx + 1}/{MAX_SIMULATIONS}, processing {len(active_trees)} active trees...")
            
            # Selection phase: collect nodes to expand from all active trees
            nodes_to_expand = []
            tree_node_mapping = []
            
            for tree in active_trees:
                node = tree.selection()
                if node is not None:
                    nodes_to_expand.append(node)
                    tree_node_mapping.append((tree, node))
            
            if not nodes_to_expand:
                print("No nodes to expand, ending simulation.")
                break
            
            # Batch generation phase
            prompts = []
            valid_indices = []
            
            for i, (tree, node) in enumerate(tree_node_mapping):
                input_prompt = tree.get_input_prompt(node, self.tokenizer)
                
                if len(self.tokenizer(input_prompt)["input_ids"]) > MAX_TOKENS:
                    print(f"Input exceeds {MAX_TOKENS} tokens, marking node as terminal.")
                    node.is_terminal = True
                    node.state["final_answer"] = INPUT_EXCEEDS_TOKEN_LIMIT
                    tree.eval_final_answer(node)
                    continue
                
                prompts.append(input_prompt)
                valid_indices.append(i)
            
            if not prompts:
                continue
            
            # Generate outputs for all valid prompts at once
            batch_outputs = self.llm.generate(prompts, self.sampling_params)
            
            # Expansion phase: create child nodes and collect execution requests
            execution_requests = []
            execution_metadata = []
            
            for batch_idx, output in enumerate(batch_outputs):
                tree_idx = valid_indices[batch_idx]
                tree, node = tree_node_mapping[tree_idx]
                
                # Create child nodes without executing code
                for idx, single_output in enumerate(output.outputs):
                    prior_prob = np.exp(single_output.cumulative_logprob / len(single_output.token_ids))
                    child_node, exec_request = tree.create_child_without_execution(
                        single_output.text.strip(), node, prior_prob, idx
                    )
                    
                    if exec_request is not None:
                        execution_requests.append(exec_request)
                        execution_metadata.append((tree, child_node, single_output.text.strip()))
            
            # Batch execution phase
            if execution_requests:
                print(f"Executing {len(execution_requests)} code requests concurrently...")
                # Wrap each request in a list to match the expected format
                requests_to_run = [[req] for req in execution_requests]
                execution_results = run_concurrent(
                    run_jupyter, 
                    args=requests_to_run, 
                    concurrency=5,
                )
                
                # Process execution results
                for (tree, child_node, output_text), exec_result in zip(execution_metadata, execution_results):
                    tree.process_execution_result(child_node, output_text, exec_result)
            
            # Update trees with backpropagation
            for tree, node in tree_node_mapping:
                if hasattr(node, 'children') and node.children:
                    node.update_recursive(NEUTRAL_REWARD, tree.root)  # Not using value model now
            
            # Remove completed trees
            active_trees = [tree for tree in active_trees if not self.is_tree_completed(tree)]
        
        # Save all trees
        for tree in mcts_trees:
            # 如果是测试模式，保存候选答案
            if tree.is_test_mode and tree.candidate_answers:
                candidates_path = tree.save_candidate_answers()
                print(f"Saved candidates to {candidates_path}")
            elif tree.is_test_mode:
                print(f"No candidate answers found for tree {tree.save_path}")
            else:
                tree.save_all_node_contexts(tree.save_path)
                print(f"Saved tree to {tree.save_path}")
    
    def is_tree_completed(self, tree):
        """Check if a tree has completed all simulations or reached terminal states"""
        # For simplicity, consider a tree completed if it has no expandable nodes
        def has_expandable_nodes(node):
            if not node.is_terminal and not node.has_children():
                return True
            for child in node.children:
                if has_expandable_nodes(child):
                    return True
            return False
        
        return not has_expandable_nodes(tree.root)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MCTS Inference for InfiAgent Evaluation Dataset')
    parser.add_argument('--test_mode', action='store_true', help='Enable test mode (evaluation without ground truth)')
    parser.add_argument('--questions_path', type=str, default='../../infiagent_evaluation/data/da-dev-questions.jsonl', 
                       help='Path to questions JSONL file')
    parser.add_argument('--tables_path', type=str, default='../../infiagent_evaluation/data/da-dev-tables', 
                       help='Path to tables directory')
    parser.add_argument('--output_path', type=str, default='./mcts_output', 
                       help='Output directory for MCTS results')
    parser.add_argument('--model_path', type=str, 
                       default='../sft_models/Qwen_25-14b-instruct_lora_sft_checkpoint-1500_NEW',
                       help='Path to the model')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions to process')
    args = parser.parse_args()
    
    set_endpoint("http://localhost:8080")
    os.environ['HF_TOKEN'] = "fill-in-your-huggingface-token"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        n=EXPANSION_NUM,
        temperature=0.7,
        top_p=0.95,
        max_tokens=8192,
        skip_special_tokens=True,
        logprobs=1,
    )

    llm = LLM(model=args.model_path)
    #llm = LLM(model=args.model_path, tensor_parallel_size=2)

    # Create batch processor
    batch_processor = BatchMCTSProcessor(llm, tokenizer, sampling_params)

    # Load questions from inference.py format
    questions = []
    with open(args.questions_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    if args.limit:
        questions = questions[:args.limit]

    print(f"Loaded {len(questions)} questions from {args.questions_path}")

    # Create output directory
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    MCTSTrees = []
    for question in questions:
        # Check if already processed
        output_file = os.path.join(args.output_path, f"{question['id']}_candidates.json")
        if os.path.exists(output_file):
            print(f"Skipping question {question['id']} as it has already been processed.")
            continue

        instruction = instruction_generate(question)
        initial_input = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]

        # For inference.py dataset, we run in test mode without ground truth
        ground_truth = None

        # Load file data
        file_name = question['file_name']
        file_path = os.path.join(args.tables_path, file_name)
        files_dict = {}
        
        if file_name != 'none' and os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                files_dict[file_name] = base64.b64encode(f.read()).decode('utf-8')
        elif file_name != 'none':
            print(f"Warning: File {file_name} not found at {file_path}")

        save_path = os.path.join(args.output_path, f"{question['id']}.json")
        mcts_tree = MCTSTree(initial_input, ground_truth, files_dict, save_path, is_test_mode=True)
        MCTSTrees.append(mcts_tree)

    if not MCTSTrees:
        print("No questions to process (all already completed or no valid questions found).")
        exit(0)

    # Process all trees concurrently
    print(f"Processing {len(MCTSTrees)} trees concurrently with MCTS...")
    batch_processor.process_trees_concurrently(MCTSTrees)