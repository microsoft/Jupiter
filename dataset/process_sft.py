import json
import os

def read_jsonl(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"file {file_path} does not exist.")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    
    return data

train_sft_data = read_jsonl('./train_sft_data.jsonl')

print(f"read {len(train_sft_data)} records.")

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

import ast

processed_train_sft_data = []

for sample in train_sft_data:
    sample_dict = {}

    sample_dict["input"] = ""
    sample_dict["instruction"] = "Observation:\n" + (sample["solution"][-2]['code_output'] if sample["solution"][-2]['code_output'] else "")

    sample_dict['output'] = "Thought: " + sample["solution"][-1]['thought'] + "\n" + "Formatted answer: " + sample["solution"][-1]['label']

    sample_dict["system"] = "You are a helpful assistant."

    sample_dict["history"] = []

    file_names = sample["file_names"]
    file_names = ast.literal_eval(file_names) if isinstance(file_names, str) else file_names

    if len(file_names) > 1:
        file_names = ', '.join(file_names)
    else:
        file_names = file_names[0] if file_names else "none"
    
    question = {
        "question": sample["task"],
        "constraints": sample["constraints"],
        "format": sample["format"],
        "file_name": file_names
    }

    sample_dict["history"].append([
        instruction_generate(question),
        "Thought: " + sample["solution"][0]['thought'] + "\n" + "Action: " + sample["solution"][0]['python_code']
    ])

    for i in range(0, len(sample["solution"])-2):
        sample_dict["history"].append([
            "Observation:\n" + (str(sample["solution"][i]['code_output'] if sample["solution"][i]['code_output'] else "")),
            "Thought: " + sample["solution"][i+1]['thought'] + "\n" + "Action: " + sample["solution"][i+1]['python_code']
        ])

    processed_train_sft_data.append(sample_dict)


output_file_path = './train_sft_processed.json'
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    json.dump(processed_train_sft_data, output_file, ensure_ascii=False, indent=4)


