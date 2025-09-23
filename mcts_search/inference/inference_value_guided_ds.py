# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import re
import os
import json
import random
import torch
import numpy as np
from transformers import AutoTokenizer
import base64
from hashlib import md5
from vllm import LLM, SamplingParams
from sandbox_fusion import run_concurrent, RunJupyterRequest, set_endpoint, run_jupyter

from node import MCTSNode
from check_result import check_correct
from estimate import ValueModel

from config import (
    MAX_SIMULATIONS, 
    EXPANSION_NUM,  
    MAX_TOKENS, 
    NEGETIVE_REWARD,
    #POSITIVE_REWARD, 
    NEUTRAL_REWARD,
    ERROR_THRESHOLD, 
    MAX_DEPTH,
    TIME_LIMIT,
    instruction_generate_ds,
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

TIME_LIMIT = 60 * 15  # 15 minutes for each task
MAX_SIMULATIONS = 50
MAX_DEPTH = 25

class MCTSTree:
    def __init__(self, initial_input, files_dict, save_path, task_folder, use_value_model=False):
        self.root = self.create_node()
        self.root.state["messages"] = initial_input
        
        self.files_dict = files_dict
        #self.save_path = save_path
        self.task_folder = task_folder  # task folder path
        self.use_value_model = use_value_model
        self.candidate_answers = []  # store candidate answers
        self.pending_value_nodes = []  # store nodes pending for value model evaluation
        self.submission_counter = 0  # couter for generating unique submission files

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

        final_answer = node.state.get("final_answer")
        if final_answer is None:
            print(f"Warning: Final answer is None for node {node.tag}")
            node.update_recursive(NEGETIVE_REWARD, self.root)
            return

        if self.use_value_model:
            self.pending_value_nodes.append(node)
        else:
            node.update_recursive(NEUTRAL_REWARD, self.root)

    # # collect all node contexts from root to each node
    # def collect_all_node_contexts(self):
    #     all_infos = []
        
    #     def dfs(node):
    #         messages = []
    #         path_nodes = []
            
    #         current = node
    #         while current is not None:
    #             path_nodes.append(current)
    #             current = current.parent
            
    #         path_nodes.reverse()
            
    #         for n in path_nodes:
    #             if "messages" in n.state:
    #                 messages.extend(n.state["messages"])
            
    #         all_infos.append({
    #             "tag": node.tag,
    #             "depth": node.depth,
    #             "messages": messages,
    #             "is_terminal": node.is_terminal,
    #             "visit_count": node.visit_count,
    #             "value_sum": node.value_sum,
    #             "value": node.value,
    #         })
            
    #         for child in node.children:
    #             dfs(child)
        
    #     dfs(self.root)
    #     return all_infos

    # def save_all_node_contexts(self, save_path):
    #     all_infos = self.collect_all_node_contexts()
    #     with open(save_path, "w", encoding="utf-8") as f:
    #         json.dump(all_infos, f, ensure_ascii=False, indent=2)
        
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
        """Create child node for batch processing without executing code"""
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
                    fetch_files=['submission.csv']  # add fetching submission.csv file
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

    def process_execution_result(self, new_node, output_text, exec_result, iteration_idx=None):
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

            # deal with file generated by the code execution, especially submission.csv
            if exec_result.get("files"):
                for file_name, file_content in exec_result["files"].items():
                    if file_name == "submission.csv":
                        self.submission_counter += 1
                        if iteration_idx is not None:
                            submission_filename = f"submission_{iteration_idx}_{self.submission_counter}.csv"
                        else:
                            submission_filename = f"submission_{self.submission_counter}.csv"
                        submission_path = os.path.join(self.task_folder, submission_filename)
                        
                        # use base64 to decode the content
                        try:
                            decoded_content = base64.b64decode(file_content).decode('utf-8')
                            with open(submission_path, 'w', encoding='utf-8', newline='') as f:
                                f.write(decoded_content)
                            print(f"Saved submission file: {submission_path}")
                            
                            # recode the submission file information in the node state
                            new_node.state["submission_file"] = submission_filename
                            new_node.state["submission_path"] = submission_path
                            
                            # collect candidate answer information immediately because the submission file is generated
                            fake_answer = f"Generated submission file: {submission_filename}"
                            self.collect_candidate_answer(new_node, fake_answer)
                        except Exception as file_err:
                            print(f"Error saving submission file: {file_err}")
                            
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
        """collect candidate answer information"""
        candidate_info = {
            "node_tag": node.tag,
            "node_depth": node.depth,
            "final_answer": final_answer,
            "visit_count": node.visit_count,
            "value_sum": node.value_sum,
            "value": node.value,
            "submission_file": node.state.get("submission_file", None),
            "submission_path": node.state.get("submission_path", None),
            "path_to_answer": self.get_node_path_messages(node)
        }
        self.candidate_answers.append(candidate_info)
        #print(f"Collected candidate answer for node {node.tag}: {final_answer[:100]}...")

    def get_node_path_messages(self, node):
        """get the full message path from root to current node"""
        messages = []
        path_nodes = []
        
        # traverse from current node to root
        current = node
        while current is not None:
            path_nodes.append(current)
            current = current.parent
        
        # reverse the path to get from root to current node
        path_nodes.reverse()
        
        # concatenate all messages in order
        for n in path_nodes:
            if "messages" in n.state:
                messages.extend(n.state["messages"])
        
        return messages

    def save_candidate_answers(self):
        # save to task folder's candidates.json
        save_path = os.path.join(self.task_folder, "candidates.json")
        
        candidates_data = {
            "total_candidates": len(self.candidate_answers),
            "unique_candidates": len(self.get_all_unique_answers()),
            "best_candidates": self.get_best_candidate_answers(),
            "all_candidates": self.candidate_answers,
        }
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(candidates_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved {len(self.candidate_answers)} candidate answers to {save_path}")
        return save_path

    def get_best_candidate_answers(self):
        """ get the best candidate answers based on value_sum / visit_count = q value """
        if not self.candidate_answers:
            return []
        
        sorted_candidates = sorted(
            self.candidate_answers, 
            key=lambda x: (x['value_sum'] / x['visit_count'] if x['visit_count'] > 0 else -1),
            reverse=True
        )

        return sorted_candidates

    def get_all_unique_answers(self):
        """ get all unique candidate answers"""
        unique_answers = {}
        for candidate in self.candidate_answers:
            answer = candidate['final_answer']
            if answer not in unique_answers:
                unique_answers[answer] = candidate
            else:
                # if answer are equal, keep the one with higher visit_count
                if candidate['visit_count'] > unique_answers[answer]['visit_count']:
                    unique_answers[answer] = candidate
        
        return list(unique_answers.values())

class BatchMCTSProcessor:
    def __init__(self, llm, tokenizer, sampling_params, value_model=None):
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.value_model = value_model
    
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
                tree.submission_counter = 0
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
                    concurrency=24,
                )
                
                # Process execution results
                for (tree, child_node, output_text), exec_result in zip(execution_metadata, execution_results):
                    tree.process_execution_result(child_node, output_text, exec_result, iteration_idx=simulation_idx)
            
            # Update trees with backpropagation and value model evaluation
            for tree, node in tree_node_mapping:
                if hasattr(node, 'children') and node.children:
                    if tree.use_value_model and self.value_model:
                        # tree.pending_value_nodes.extend(node.children)
                        tree.pending_value_nodes.append(node)
                    else:
                        # give neutral reward when not using value model
                        node.update_recursive(NEUTRAL_REWARD, tree.root)
            
            # batch process pending nodes with value model
            self.batch_evaluate_with_value_model(active_trees)
            
            # Remove completed trees
            active_trees = [tree for tree in active_trees if not self.is_tree_completed(tree)]
                
        # # Save all trees
        for tree in mcts_trees:
            # save candidate answers if in test mode
            if tree.candidate_answers:
                candidates_path = tree.save_candidate_answers()
                print(f"Saved candidates to {candidates_path}")
            # else:
            #     tree.save_all_node_contexts(tree.save_path)
            #     print(f"Saved tree to {tree.save_path}")
    
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

    def batch_evaluate_with_value_model(self, active_trees):
        """Batch evaluate nodes using value model"""
        if not self.value_model:
            return
            
        # collect all nodes that need evaluation
        all_nodes_to_evaluate = []
        tree_node_pairs = []
        
        for tree in active_trees:
            if tree.use_value_model and tree.pending_value_nodes:
                for node in tree.pending_value_nodes:
                    all_nodes_to_evaluate.append(node)
                    tree_node_pairs.append((tree, node))
                tree.pending_value_nodes.clear()  # clear pending nodes after processing
        
        if not all_nodes_to_evaluate:
            return
        
        # construct prompts for all nodes
        prompts = []
        for tree, node in tree_node_pairs:
            prompt = tree.get_node_path_messages(node)
            prompts.append(prompt)

        def format_messages_simple(messages):
            #print(f"Formatting messages for value model evaluation:\n{messages}")
            return "\n\n".join([f"{m['role']}:\n{m['content']}" for m in messages])

        prompts = [ format_messages_simple(msg) for msg in prompts ]
        
        if prompts:
            print(f"Evaluating {len(prompts)} nodes with value model...")
            try:
                scores = self.value_model.score_mean_pooling(prompts, max_length=8000, batch_size=5, show_progress=True)
                
                # apply scores to corresponding nodes
                for (tree, node), score in zip(tree_node_pairs, scores):
                    node.update_recursive(score, tree.root)
                    
            except Exception as e:
                print(f"Error during value model evaluation: {e}")
                # if evaluation fails, use neutral reward
                for tree, node in tree_node_pairs:
                    node.update_recursive(NEUTRAL_REWARD, tree.root)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MCTS Inference for InfiAgent Evaluation Dataset')
    parser.add_argument('--questions_path', type=str, default='../../../DSBench/data_modeling/simplify_results.jsonl', 
                       help='Path to questions JSONL file')
    parser.add_argument('--tables_path', type=str, default='../../../DSBench/data_modeling/data/data_resplit', 
                       help='Path to tables directory')
    parser.add_argument('--output_path', type=str, default='./mcts_output_ds', 
                       help='Output directory for MCTS results')
    parser.add_argument('--model_path', type=str, 
                       default='Qwen/Qwen2.5-7B-Instruct',
                       help='Path to the model')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of questions to process')
    parser.add_argument('--use_value_model', action='store_true', help='Enable value model for node evaluation')
    parser.add_argument('--value_model_path', type=str, default='../value_model/value_model_round1_7B_NEW/checkpoint-63309/epoch_3.0', 
                       help='Path to the base model of value model')
    parser.add_argument('--value_head_path', type=str, default='./v-shuocli/mcts_search/value_model/value_model_round1_7B_NEW/checkpoint-63309/epoch_3.0/value_head.pt',
                       help='Path to the value head .pt checkpoint of value model')
    args = parser.parse_args()
    
    set_endpoint("http://localhost:8080")
    os.environ['HF_TOKEN'] = "fill_in_your_huggingface_token"

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

    # Initialize value model if requested
    value_model = None
    if args.use_value_model:
        if not args.value_model_path or not args.value_head_path:
            print("Error: --value_model_path and --value_head_path are required when using value model")
            exit(1)
        
        print(f"Loading value model from {args.value_model_path}")
        print(f"Loading value head from {args.value_head_path}")
        try:
            # specify the cuda:1 here
            # should modified later to support multiple GPUs
            value_model = ValueModel(
                model_name_or_path=args.value_model_path,
                dtype=torch.bfloat16,
                device="cuda:1",
            )
            value_model.load_value_head(args.value_head_path)
            print("Value model loaded successfully")
        except Exception as e:
            print(f"Error loading value model: {e}")
            print("Continuing without value model...")
            value_model = None
            args.use_value_model = False

    # Create batch processor
    batch_processor = BatchMCTSProcessor(llm, tokenizer, sampling_params, value_model)

    # Load questions from inference.py format
    questions = []
    with open(args.questions_path, 'r') as f:
        for line in f:
            questions.append(json.loads(line))
    
    if args.limit:
        questions = questions[:args.limit]

    print(f"Loaded {len(questions)} questions from {args.questions_path}")

    # Create output directory structure similar to inference.py
    model_name = args.model_path.split('/')[-1]
    inference_output_path = os.path.join(args.output_path, model_name)
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(inference_output_path):
        os.makedirs(inference_output_path)

    MCTSTrees = []
    for question in questions:
        # create task folder for each question under inference_output_path
        task_folder = os.path.join(inference_output_path, question['task'])
        if not os.path.exists(task_folder):
            os.makedirs(task_folder)

        instruction = instruction_generate_ds(question)
        initial_input = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": instruction},
        ]

        # Load file data similar to inference.py
        # Initialize file_names list
        question['file_names'] = []
        task_data_path = os.path.join(args.tables_path, question['task'])
        files_dict = {}
        
        if os.path.exists(task_data_path):
            for file_name in os.listdir(task_data_path):
                file_name = file_name.strip()
                full_file_path = os.path.join(task_data_path, file_name)
                if not os.path.exists(full_file_path):
                    print(f"Warning: File {full_file_path} not found.")
                    continue
                
                # read the file and encode it in base64
                with open(full_file_path, 'rb') as f:
                    files_dict[file_name] = base64.b64encode(f.read()).decode('utf-8')

                question['file_names'].append(file_name)
        else:
            print(f"Warning: Task data path {task_data_path} not found.")

        # save_path = os.path.join(task_folder, "history.json")
        # mcts_tree = MCTSTree(initial_input, files_dict, save_path, task_folder, use_value_model=args.use_value_model)
        save_path = None
        mcts_tree = MCTSTree(initial_input, files_dict, save_path, task_folder, use_value_model=args.use_value_model)
        MCTSTrees.append(mcts_tree)

    if not MCTSTrees:
        print("No questions to process (all already completed or no valid questions found).")
        exit(0)

    # Process all trees concurrently
    print(f"Processing {len(MCTSTrees)} trees concurrently with MCTS...")
    batch_processor.process_trees_concurrently(MCTSTrees)