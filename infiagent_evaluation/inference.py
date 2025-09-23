# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import tqdm
import json
import base64

from sandbox_fusion import run_concurrent, RunJupyterRequest, set_endpoint, run_jupyter
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

set_endpoint("http://localhost:8080")

os.environ['HF_TOKEN'] = "fill-in-your-huggingface-token"

questions_path = 'data/da-dev-questions.jsonl'
#labels_path = 'data/da-dev-labels.jsonl'
tables_path = 'data/da-dev-tables'

questions = []

with open(questions_path, 'r') as f:
    for line in f:
        questions.append(json.loads(line))

#questions = questions[:3]  # Limit to 3 questions for testing

def instruction_generate(question: dict) -> str:
    raw_question = question['question']
    constraints = question.get('constraints', 'none')
    output_format = question.get('format', 'none')
    file_name = question.get('file_name', 'none')

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
Format: {output_format}  
Available Local Files: {file_name}
"""

model_path = 'meta-llama/Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

MAX_TOKENS= 100000

sampling_params = SamplingParams(
    n=1,
    temperature=0.2,
    top_p=0.95,
    max_tokens=8192,
    skip_special_tokens=True,
)

llm = LLM(model=model_path) # for model size <= 14B
#llm = LLM(model=model_path, tensor_parallel_size=2) # for model size > 14B

max_iterations = 25
 active_contexts = []


ouput_path = './inference_output'
if not os.path.exists(ouput_path):
    os.makedirs(ouput_path)

model_name = model_path.split('/')[-1]
infernce_output_path = os.path.join(ouput_path, model_name)

if not os.path.exists(infernce_output_path):
    os.makedirs(infernce_output_path)

# 初始化每个问题的上下文
for question in questions:
    output_file = os.path.join(infernce_output_path, f"{question['id']}.json")
    if os.path.exists(output_file):
        print(f"Skipping question {question['id']} as it has already been processed.")
        continue

    instruction = instruction_generate(question)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": instruction}
    ]

    file_name = question.get('file_name', 'none')
    file_path = os.path.join(tables_path, file_name)
    files_dict = {}
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            files_dict[file_name] = base64.b64encode(f.read()).decode('utf-8')
    else:
        print(f"Warning: File {file_name} not found.")

    active_contexts.append({
        "id": question["id"],
        "question": question["question"],
        "messages": messages,
        "files_dict": files_dict,
        "codes": [],
        "finished": False,
        "result": {
            "id": question["id"],
            "question": question["question"],
            "conversation_history": [],
            "formatted_answer": None,
        }
    })

print(f"Processing {len(active_contexts)} questions in batch mode...")

from sandbox_fusion import run_concurrent  # 加在顶部

# 多轮推理主循环
for iteration in range(max_iterations):
    print(f"\n--- Iteration {iteration+1} ---")

    # 构建 prompts
    prompts = []
    running_contexts = []
    for ctx in active_contexts:
        if ctx["finished"]:
            continue
        prompt = tokenizer.apply_chat_template(ctx["messages"], tokenize=False, add_generation_prompt=True)

        token_len = len(tokenizer(prompt)["input_ids"])

        if token_len > MAX_TOKENS:
            print(f"[{ctx['id']}] Prompt too long ({token_len} tokens). Marking as failed.")
            ctx["result"]["formatted_answer"] = "Failed"
            ctx["result"]["error"] = f"Prompt too long: {token_len} tokens"
            ctx["result"]["conversation_history"] = ctx["messages"]
            ctx["finished"] = True

            output_file = os.path.join(infernce_output_path, f"{ctx['id']}.json")
            with open(output_file, 'w') as f:
                json.dump(ctx["result"], f, indent=4)
            continue


        prompts.append(prompt)
        running_contexts.append(ctx)

    if not prompts:
        print("All questions finished.")
        break

    # 批量生成
    try:
        responses = llm.generate(prompts=prompts, sampling_params=sampling_params)
    except Exception as e:
        print(f"Batch generation error: {e}")
        break

    # 分开处理：先判断需要执行代码的 ctx，统一并发执行
    ctxs_to_execute = []
    requests_to_run = []

    for response, ctx in zip(responses, running_contexts):
        text = response.outputs[0].text.strip()
        ctx["messages"].append({"role": "assistant", "content": text})

        if "formatted answer:" in text.lower():
            formatted = text.split("Formatted answer:")[-1].strip()
            ctx["result"]["formatted_answer"] = formatted
            ctx["finished"] = True
            ctx["result"]["conversation_history"] = ctx["messages"]
            print(f"[{ctx['id']}] Final answer found. Saving result...")

            output_file = os.path.join(infernce_output_path, f"{ctx['id']}.json")
            with open(output_file, 'w') as f:
                json.dump(ctx["result"], f, indent=4)
            continue

        if "thought:" not in text.lower() or "action:" not in text.lower():
            ctx["messages"].append({"role": "user", "content": "Observation: You response should contain both 'Thought:' and 'Action:' sections."})
            continue

        code_match = re.search(r'```python\s*(.*?)```', text, re.DOTALL)
        if not code_match:
            ctx["messages"].append({"role": "user", "content": "Observation: You need to provide a valid python code snippet wrapped in triple backticks."})
            continue

        code_snippet = code_match.group(1).strip()
        #ctx["last_code"] = code_snippet
        if "codes" not in ctx:
            ctx["codes"] = []
        ctx["codes"].append(code_snippet)

        

        # 准备并发执行的请求
        ctxs_to_execute.append(ctx)
        # requests_to_run.append([
        #     RunJupyterRequest(
        #         cells=[code_snippet],
        #         files=ctx["files_dict"],
        #         total_timeout=60 * 10,
        #     )
        # ])

        requests_to_run.append([
            RunJupyterRequest(
                cells=ctx["codes"],  # 传所有历史代码
                files=ctx["files_dict"],
                #total_timeout=60 * 10,
                total_timeout=30,
            )
        ])

    # 批量并发执行 sandbox
    if requests_to_run:
        print(f"Running {len(requests_to_run)} sandbox executions concurrently...")
        try:
            sandbox_results = run_concurrent(run_jupyter, args=requests_to_run, concurrency=24)
        except Exception as e:
            print(f"Concurrent sandbox execution failed: {e}")
            for ctx in ctxs_to_execute:
                ctx["messages"].append({"role": "user", "content": f"Observation: Sandbox execution failed: {e}"})
            continue

        # 逐个处理 sandbox 返回结果
        for ctx, exec_result in zip(ctxs_to_execute, sandbox_results):
            code_output = "Observation:\n"
            try:
                exec_result = json.loads(exec_result.json())

                if exec_result.get("status") != "Success":
                    status_info = exec_result.get("driver", {}).get("status", "Unknown")
                    ctx["messages"].append({"role": "user", "content": f"Observation: Code execution failed with status: {status_info}"})
                    
                    ctx["result"]["formatted_answer"] = "Failed"
                    ctx["result"]["error"] = f"Sandbox execution failed with status: {status_info}"
                    ctx["result"]["conversation_history"] = ctx["messages"]
                    ctx["finished"] = True
                    output_file = os.path.join(infernce_output_path, f"{ctx['id']}.json")
                    with open(output_file, 'w') as f:
                        json.dump(ctx["result"], f, indent=4)
                    print(f"[{ctx['id']}] Sandbox execution failed with status: {status_info}. Saving result...")
                    
                    continue

                if exec_result.get("cells"):
                    last_output = exec_result["cells"][-1]
                    if last_output["stdout"] != "":
                        code_output += last_output["stdout"] + "\n"
                    if last_output["stderr"] != "":
                        code_output += last_output["stderr"] + "\n"
                    if last_output["error"] != []:
                        for err in last_output["error"]:
                            if "ename" in err and "evalue" in err:
                                code_output += f"{err['ename']}: {err['evalue']}\n"
            except Exception as e:
                code_output += f"Sandbox error: {str(e)}"

            ctx["messages"].append({"role": "user", "content": code_output})

# 保存剩余未完成的问题（迭代用尽）
for ctx in active_contexts:
    if not ctx["finished"]:
        ctx["result"]["formatted_answer"] = "Failed"
        ctx["result"]["error"] = f"Did not get formatted answer in {max_iterations} iterations"
        ctx["result"]["conversation_history"] = ctx["messages"]
        print(f"[{ctx['id']}] Max iterations reached. Saving failed result...")
        output_file = os.path.join(infernce_output_path, f"{ctx['id']}.json")
        with open(output_file, 'w') as f:
            json.dump(ctx["result"], f, indent=4)
