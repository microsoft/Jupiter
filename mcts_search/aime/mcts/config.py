# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Search configuration constants
MAX_SIMULATIONS = 40
EXPANSION_NUM = 3
MAX_TOKENS = 100000
MAX_DEPTH = 8
ERROR_THRESHOLD = 3
NEGETIVE_REWARD = -1.0
POSITIVE_REWARD = 1.0
NEUTRAL_REWARD = 0.0
TREE_PER_TASK = 10
TIME_LIMIT = 60 * 3  # 60 seconds

# Failure messages
INPUT_EXCEEDS_TOKEN_LIMIT = "Input exceeds maximum token limit."
PARSABLE_TEXT_FAIL = "Fail to generate parsable text for next step."
CODE_BLOCK_FAIL = "Fail to generate valid code block for next step."
CODE_BLOCK_EMPTY = "Code block is empty."
SANDBOX_ERROR = "Sandbox error."
SANDBOX_EXECUTION_FAILED = "Sandbox execution failed."
TOO_MANY_CODE_ERRORS = "Too many consecutive steps have code errors."
FAIL_WITHIN_LIMITED_STEPS = "Fail to solve the problem within limited steps."