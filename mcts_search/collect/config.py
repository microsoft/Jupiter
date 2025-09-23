# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Search configuration constants
MAX_SIMULATIONS = 50
EXPANSION_NUM = 3
MAX_TOKENS=100000
MAX_DEPTH = 10
ERROR_THRESHOLD = 3
NEGETIVE_REWARD = -1.0
POSITIVE_REWARD = 1.0
TREE_PER_TASK = 1
TIME_LIMIT = 60 * 3  # 3 minutes

# Failure messages
INPUT_EXCEEDS_TOKEN_LIMIT = "Input exceeds maximum token limit."
PARSABLE_TEXT_FAIL = "Fail to generate parsable text for next step."
CODE_BLOCK_FAIL = "Fail to generate valid code block for next step."
CODE_BLOCK_EMPTY = "Code block is empty."
SANDBOX_ERROR = "Sandbox error."
SANDBOX_EXECUTION_FAILED = "Sandbox execution failed."
TOO_MANY_CODE_ERRORS = "Too many consecutive steps have code errors."
FAIL_WITHIN_LIMITED_STEPS = "Fail to solve the problem within limited steps."

# Instruction generation function
def instruction_generate(question: dict) -> str:
    raw_question = question['task']
    constraints = question.get('constraints', 'none')
    output_format = question.get('format', 'none')
    file_name = question.get('file_names', 'none')

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