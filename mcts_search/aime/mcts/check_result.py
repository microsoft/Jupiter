# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json

import ast

def compare_values(v1, v2):
    if v1 is None and v2 is None:
        return True
    if isinstance(v1, str) and isinstance(v2, str):
        return v1 == v2
    try:
        if isinstance(v1, int):
            v1 = float(v1)
        if isinstance(v2, int):
            v2 = float(v2)
        if isinstance(v1, float) and isinstance(v2, float):
            return abs(v1 - v2) < 1e-3 or abs(v1 - v2) / (abs(v1) + abs(v2) + 1e-6) < 0.001
    except:
        return False
    if isinstance(v1, (list, tuple)) and isinstance(v2, (list, tuple)) and len(v1) == len(v2):
        # 
        v1_list, v2_list = list(v1), list(v2)
        unmatched = list(v2_list)
        for a in v1_list:
            for i, b in enumerate(unmatched):
                if compare_values(a, b):
                    unmatched.pop(i)
                    break
            else:
                return False
        return True
    if isinstance(v1, dict) and isinstance(v2, dict) and set(v1.keys()) == set(v2.keys()):
        return all(compare_values(v1[k], v2[k]) for k in v1)
    return False


def check_correct(solution_str, ground_truth):
    answer_items = ground_truth.split("@")[1:]
    answer_items = [item.strip() for item in answer_items]

    answer_dict = {}

    for item in answer_items:
        if "[" in item and "]" in item:
            first_bracket_index = item.index("[")
            last_bracket_index = item.rindex("]")

            if len(item[:first_bracket_index].strip()) == 0 or len(item[first_bracket_index + 1:last_bracket_index].strip()) == 0:
                continue

            key = item[:first_bracket_index].strip()
            value = item[first_bracket_index + 1:last_bracket_index].strip()

            try:
                value = ast.literal_eval(value)
            except Exception:
                pass

            answer_dict[key] = value

    solution_items = solution_str.split("@")[1:]
    solution_items = [item.strip() for item in solution_items]

    solution_dict = {}

    for item in solution_items:
        if "[" in item and "]" in item:
            first_bracket_index = item.index("[")
            last_bracket_index = item.rindex("]")

            if len(item[:first_bracket_index].strip()) == 0 or len(item[first_bracket_index + 1:last_bracket_index].strip()) == 0:
                continue

            key = item[:first_bracket_index].strip()
            value = item[first_bracket_index + 1:last_bracket_index].strip()

            try:
                value = ast.literal_eval(value)
            except Exception:
                pass

            solution_dict[key] = value

    correct_sub_questions = 0

    for key, value in answer_dict.items():
        if key in solution_dict:
            if compare_values(solution_dict[key], value):
                correct_sub_questions += 1

    return correct_sub_questions