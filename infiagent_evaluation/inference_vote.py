# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import re
import json
import base64
import logging
from collections import Counter
from typing import List, Dict, Any

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from sandbox_fusion import (
    run_concurrent,
    RunJupyterRequest,
    set_endpoint,
    run_jupyter,
)
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

set_endpoint("http://localhost:8080")   
os.environ["HF_TOKEN"] = "hf_xxx"  


questions_path = "data/da-dev-questions.jsonl"
tables_path    = "data/da-dev-tables"


model_path = "meta-llama/Llama-3.1-8B-Instruct"


NUM_VOTES        = 5
MAX_CONCURRENCY  = 64          
MAX_ITERATIONS   = 25            
MAX_TOKENS       = 100000          


output_root = "./inference_output_vote"
model_tag   = model_path.rstrip("/").split("/")[-1]

output_dir  = os.path.join(output_root, model_tag+str(MAX_ITERATIONS))
os.makedirs(output_dir, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

def instruction_generate(question: Dict[str, Any]) -> str:
    raw_question  = question["question"]
    constraints   = question.get("constraints", "none")
    output_format = question.get("format", "none")
    file_name     = question.get("file_name", "none")

    return f"""Please complete the following data analysis task as best as you can. On each turn, you can use the python code sandbox to execute the python code snippets you generate and get the output.

### Format to follow
Thought: your reasoning about what to do next based on the current context.
Action: the code snippet you want to execute in the python code sandbox. Use ```python ...```.

Then you will get the Observation based on the execution results and can iterate. Once you have enough information to answer, finish with:
Thought: ...
Formatted answer: ...

### Important Notes
- **Do NOT** include Observation content yourself.
- For any code output, always use print().
- Write files to "./" if produced.

### Task
Question: {raw_question}
Constraints: {constraints}
Format: {output_format}
Available Local Files: {file_name}
"""

questions: List[Dict[str, Any]] = []
with open(questions_path, "r", encoding="utf-8") as f:
    for ln in f:
        questions.append(json.loads(ln))

log.info("Loaded %d questions.", len(questions))


tokenizer       = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
sampling_params = SamplingParams(
    n=1,
    temperature=0.7,
    top_p=0.95,
    max_tokens=8192,
    skip_special_tokens=True,
)

llm = LLM(
    model=model_path,
    gpu_memory_utilization=0.7,  
    tensor_parallel_size=1       
)

active_contexts: List[Dict[str, Any]] = []

for q in questions:
    file_name  = q.get("file_name", "none")
    file_path  = os.path.join(tables_path, file_name)
    files_dict = {}

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            files_dict[file_name] = base64.b64encode(f.read()).decode("utf-8")
    else:
        log.warning("File %s not found (qid=%s).", file_name, q["id"])

    for run_idx in range(NUM_VOTES):
        ctx_id      = f"{q['id']}_{run_idx}"
        output_file = os.path.join(output_dir, f"{ctx_id}.json")
        if os.path.exists(output_file):              
            log.info("Skip %s (already processed).", ctx_id)
            continue

        instruction = instruction_generate(q)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",   "content": instruction},
        ]

        active_contexts.append({
            "id": ctx_id,      
            "qid": q["id"],      
            "question": q["question"],
            "messages": messages,
            "files_dict": files_dict,
            "codes": [],
            "finished": False,
            "result": {
                "id": ctx_id,
                "question": q["question"],
                "conversation_history": [],
                "formatted_answer": None,
            },
        })

log.info("Total active contexts: %d (=%d questions × %d runs)", len(active_contexts), len(questions), NUM_VOTES)

for iteration in range(MAX_ITERATIONS):
    log.info("--- Iteration %d ---", iteration + 1)

    prompts, running_ctxs = [], []
    for ctx in active_contexts:
        if ctx["finished"]:
            continue
        prompt = tokenizer.apply_chat_template(ctx["messages"], tokenize=False, add_generation_prompt=True)
        if len(tokenizer(prompt)["input_ids"]) > MAX_TOKENS:
            log.error("[%s] Prompt too long, mark failed.", ctx["id"])
            ctx["result"]["formatted_answer"] = "Failed (prompt too long)"
            ctx["finished"] = True
            with open(os.path.join(output_dir, f"{ctx['id']}.json"), "w") as f:
                json.dump(ctx["result"], f, indent=4, ensure_ascii=False)
            continue

        prompts.append(prompt)
        running_ctxs.append(ctx)

    if not prompts:
        log.info("All contexts finished.")
        break

    try:
        responses = llm.generate(prompts=prompts, sampling_params=sampling_params)
    except Exception as e:
        log.exception("LLM batch generation error → abort: %s", e)
        break

    ctxs_to_execute, requests_to_run = [], []
    for resp, ctx in zip(responses, running_ctxs):
        text = resp.outputs[0].text.strip()
        ctx["messages"].append({"role": "assistant", "content": text})

        # 若已产生最终答案
        if "formatted answer:" in text.lower():
            formatted = text.split("Formatted answer:")[-1].strip()
            ctx["result"]["formatted_answer"] = formatted
            ctx["result"]["conversation_history"] = ctx["messages"]
            ctx["finished"] = True

            with open(os.path.join(output_dir, f"{ctx['id']}.json"), "w") as f:
                json.dump(ctx["result"], f, indent=4, ensure_ascii=False)
            log.info("[%s] Completed (formatted answer).", ctx["id"])
            continue

        if "thought:" not in text.lower() or "action:" not in text.lower():
            ctx["messages"].append({"role": "user", "content": "Observation: Response must contain 'Thought:' and 'Action:' sections."})
            continue

        m = re.search(r"```python\s*(.*?)```", text, re.DOTALL)
        if not m:
            ctx["messages"].append({"role": "user", "content": "Observation: Provide python code snippet inside triple backticks."})
            continue
        code_snippet = m.group(1).strip()
        ctx.setdefault("codes", []).append(code_snippet)

        ctxs_to_execute.append(ctx)
        requests_to_run.append([
            RunJupyterRequest(
                cells=ctx["codes"],
                files=ctx["files_dict"],
                #total_timeout=60 * 5,
                total_timeout=3,
            )
        ])

    # execute concurrent sandbox jobs
    if requests_to_run:
        log.info("Executing %d sandbox jobs (concurrency=%d)...", len(requests_to_run), MAX_CONCURRENCY)
        try:
            sandbox_results = run_concurrent(run_jupyter, args=requests_to_run, concurrency=MAX_CONCURRENCY)
        except Exception as e:
            log.exception("Sandbox run_concurrent failed: %s", e)
            for ctx in ctxs_to_execute:
                ctx["messages"].append({"role": "user", "content": f"Observation: Sandbox execution failed: {e}"})
            continue

        # write the execution result to the observation
        for ctx, exec_result in zip(ctxs_to_execute, sandbox_results):
            obs_text = "Observation:\n"
            try:
                exec_result = json.loads(exec_result.json())
                if exec_result.get("status") != "Success":
                    status = exec_result.get("driver", {}).get("status", "Unknown")
                    obs_text += f"Code execution failed (status={status})\n"
                else:
                    # only get the last cell output
                    last = exec_result.get("cells", [])[-1] if exec_result.get("cells") else {}
                    obs_text += last.get("stdout", "")
                    if last.get("stderr"):
                        obs_text += last["stderr"]
                    if last.get("error"):
                        for err in last["error"]:
                            obs_text += f"{err.get('ename', '')}: {err.get('evalue', '')}\n"
            except Exception as e:
                obs_text += f"Sandbox parse error: {e}"

            ctx["messages"].append({"role": "user", "content": obs_text})


# marked as failed if not finished
for ctx in active_contexts:
    if not ctx["finished"]:
        ctx["result"]["formatted_answer"] = "Failed (max iterations)"
        ctx["result"]["conversation_history"] = ctx["messages"]
        with open(os.path.join(output_dir, f"{ctx['id']}.json"), "w") as f:
            json.dump(ctx["result"], f, indent=4, ensure_ascii=False)
        log.warning("[%s] Reached max iterations.", ctx["id"])

# majority vote aggregation
aggregated: Dict[str, List[str]] = {}
for ctx in active_contexts:
    ans = ctx["result"].get("formatted_answer")
    if ans:
        aggregated.setdefault(ctx["qid"], []).append(ans)

majority_results = {}
for qid, answers in aggregated.items():
    winner, votes = Counter(answers).most_common(1)[0]
    majority_results[qid] = {
        "majority_answer": winner,
        "votes": votes,
        "all_answers": answers,
    }

maj_path = os.path.join(output_dir, "majority_results.json")
with open(maj_path, "w", encoding="utf-8") as f:
    json.dump(majority_results, f, indent=4, ensure_ascii=False)

log.info("Majority‑vote aggregation saved → %s", maj_path)
