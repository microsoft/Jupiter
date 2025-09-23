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
    ERROR_THRESHOLD, 
    MAX_DEPTH,
    TREE_PER_TASK,
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
    def __init__(self, initial_input, ground_truth, files_dict, save_path):
        self.root = self.create_node()
        self.root.state["messages"] = initial_input
        
        self.ground_truth = ground_truth
        self.files_dict = files_dict
        self.save_path = save_path

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

        if self.ground_truth:
            final_answer = node.state["final_answer"]
            if final_answer is None:
                print(f"Warning: Final answer is None for node {node.tag}")
                node.update_recursive(NEGETIVE_REWARD, self.root)
                return
            
            correct_sub_problem = check_correct(self.ground_truth, final_answer)
            # backup
            node.update_recursive(POSITIVE_REWARD * correct_sub_problem /  self.ground_truth.count("@") if correct_sub_problem > 0 else NEGETIVE_REWARD, self.root)

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
            new_node.is_terminal = True
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
                new_node.is_terminal = True
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
            new_node.is_terminal = True
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

                new_node.is_terminal = True
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
                    concurrency=24
                )
                
                # Process execution results
                for (tree, child_node, output_text), exec_result in zip(execution_metadata, execution_results):
                    tree.process_execution_result(child_node, output_text, exec_result)
            
            # Update trees with backpropagation
            for tree, node in tree_node_mapping:
                if hasattr(node, 'children') and node.children:
                    node.update_recursive(0, tree.root)  # Not using value model now
            
            # Remove completed trees
            active_trees = [tree for tree in active_trees if not self.is_tree_completed(tree)]
        
        # Save all trees
        for tree in mcts_trees:
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
    set_endpoint("http://localhost:8080")
    os.environ['HF_TOKEN'] = "fill-in-your-huggingface-token"

    model_path = 'Path/to/your/model'  # fill-in-your-model-path
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    sampling_params = SamplingParams(
        n=EXPANSION_NUM,
        temperature=1,
        top_p=0.95,
        max_tokens=8192,
        skip_special_tokens=True,
        logprobs=1,
    )

    llm = LLM(model=model_path)
    #llm = LLM(model=model_path, tensor_parallel_size=2)

    # Create batch processor
    batch_processor = BatchMCTSProcessor(llm, tokenizer, sampling_params)

    data_to_collect_path = "../../dataset/train_collect_data.jsonl"
    data_to_collect = []

    with open(data_to_collect_path, 'r') as f:
        for line in f:
            data_to_collect.append(json.loads(line))

    # 
    total = len(data_to_collect)
    split = total // 4
    # specify which split to collect this time
    # data_to_collect = data_to_collect[:split]
    # data_to_collect = data_to_collect[split:2*split]
    # data_to_collect = data_to_collect[2*split:3*split]
    data_to_collect = data_to_collect[3*split:]
    
    data_files_base_path = "path/to/your/data_files"  # fill-in-your-data-files-path
    save_base_path = "./collected_output"
    if not os.path.exists(save_base_path):
        os.makedirs(save_base_path)

    for tree_idx in range(TREE_PER_TASK):
        print(f"Generating MCTS tree {tree_idx + 1}/{TREE_PER_TASK} for each task...")
        tree_save_path = os.path.join(save_base_path, f"tree_{tree_idx + 1}")
        if not os.path.exists(tree_save_path):
            os.makedirs(tree_save_path)

        MCTSTrees = []
        for data in data_to_collect:
            instruction = instruction_generate(data)
            initial_input = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": instruction},
            ]
            ground_truth = data["label"]

            if not ('@' in ground_truth):
                continue

            data_files_path = [os.path.join(data_files_base_path, file_path) for file_path in data["data_files"]]

            files_dict = {}
            for file_path in data_files_path:
                if not os.path.exists(file_path):
                    print(f"Warning: File {file_path} does not exist.")
                    continue
                with open(file_path, 'r') as f:
                    files_dict[os.path.basename(file_path)] = base64.b64encode(f.read().encode('utf-8')).decode('utf-8')

            
            save_path = os.path.join(
                tree_save_path,
                f"{int(md5(data['task'].encode()).hexdigest()[:8], 16)}.json"
            )
            mcts_tree = MCTSTree(initial_input, ground_truth, files_dict, save_path)
            MCTSTrees.append(mcts_tree)

        # Process all trees concurrently
        print(f"Processing {len(MCTSTrees)} trees concurrently...")
        batch_processor.process_trees_concurrently(MCTSTrees)