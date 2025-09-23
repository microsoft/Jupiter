# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import argparse
import re
import os

def read_jsonl(file_name):
    """
    Read a file with each line containing a JSON string representing a dictionary,
    and return a list of dictionaries.

    :param file_name: Name of the file to read from.
    :return: List of dictionaries.
    """
    dict_list = []
    with open(file_name, 'r') as file:
        for line in file:
            # Convert the JSON string back to a dictionary.
            dictionary = json.loads(line.rstrip('\n'))
            dict_list.append(dictionary)
    return dict_list

def write_jsonl(dict_list, file_name):
    """
    Write a list of dictionaries to a file, with each dictionary on a new line.

    :param dict_list: List of dictionaries to write to the file.
    :param file_name: Name of the file to write to.
    """
    with open(file_name, 'w') as file:
        for dictionary in dict_list:
            # Convert the dictionary to a JSON string.
            dict_str = json.dumps(dictionary)
            # Write the JSON string to the file followed by a new line character.
            file.write(dict_str + '\n')


# Function to extract formatted text using regular expressions
def extract_format(input_string):
    pattern = r"@(\w+)\[(.*?)\]"
    matches = re.findall(pattern, input_string)
    answer_names = [match[0] for match in matches]
    answers = [match[1] for match in matches]
    return answer_names, answers


# added
def normalize_list_string(s):
    if not isinstance(s, str):
        return [str(s)]
    s = s.strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1]
    return [x.strip().strip('"').strip("'") for x in s.split(",") if x.strip()]

# added
def is_empty_value(val):
    if val is None:
        return True
    if isinstance(val, str):
        v = val.strip()
        return v in ['', '[]', '{}', 'None', 'none', 'nan', 'NaN', 'null', 'Null']
    
    if isinstance(val, (list, dict)) and len(val) == 0:
        return True
    return False

def is_equal(response, label):
    if response == label:
        return True
    if is_empty_value(response) and is_empty_value(label):
        return True
    try:
        return abs(float(response) - float(label)) < 1e-6
    except Exception:
        pass
    try:
        resp_list = normalize_list_string(response)
        label_list = normalize_list_string(label)
        if len(resp_list) != len(label_list):
            return False
        for r, l in zip(resp_list, label_list):
            try:
                if abs(float(r) - float(l)) > 1e-6:
                    return False
            except Exception:
                if r != l:
                    return False
        return True
    except Exception:
        return False
    
# Function to evaluate responses with the adjusted label structure
def evaluate_responses(labels, responses):
    results = []
    for label in labels:
        label_id = label["id"]
        label_answers = {ans[0]: ans[1] for ans in label["common_answers"]}

        corresponding_response = next(
            (resp for resp in responses if "id" in resp.keys() and resp["id"] == label_id), None)

        if corresponding_response and corresponding_response['response']:
            # have corresponding response and response is not empty
            answer_names, answers = extract_format(corresponding_response["response"])
            extracted_answers = dict(zip(answer_names, answers))
            correct_answers = {ans_name: is_equal(extracted_answers.get(ans_name), label_answers[ans_name]) for ans_name
                               in label_answers.keys()}
            result = {
                "id": label_id,
                "label_answers": label_answers,
                "predicted_answers": extracted_answers,
                "correctness": correct_answers
            }
        else:
            # no corresponding response or response is empty
            correct_answers = {ans_name: False for ans_name in label_answers.keys()}
            result = {
                "id": label_id,
                "label_answers": label_answers,
                "predicted_answers": {},
                "correctness": correct_answers
            }
        results.append(result)
    return results


# Function to read concepts data from a JSONL file
def read_concepts_from_file(file_name):
    concepts_data = {}
    with open(file_name, 'r') as file:
        for line in file:
            question_data = json.loads(line.rstrip('\n'))
            question_id = question_data["id"]
            concepts = question_data["concepts"]
            concepts_data[question_id] = concepts
    return concepts_data


# Function to analyze concept accuracy
def analyze_concepts_accuracy(results, concepts_data):
    concept_accuracy = {}
    for result in results:
        question_id = result["id"]
        if question_id in concepts_data:
            for concept in concepts_data[question_id]:
                if concept not in concept_accuracy:
                    concept_accuracy[concept] = {"total": 0, "correct": 0}
                concept_accuracy[concept]["total"] += 1
                if 'correctness' in result and all(result['correctness'].values()):
                    concept_accuracy[concept]["correct"] += 1
    return {concept: round(acc["correct"] / acc["total"], 4) for concept, acc in concept_accuracy.items()}


def analyze_concepts_count_accuracy(results, concepts_data):
    concept_count_accuracy = {}
    two_or_more_concepts_accuracy = {"total": 0, "correct": 0}

    for result in results:
        question_id = result["id"]
        if question_id in concepts_data:
            concept_count = len(concepts_data[question_id])
            if concept_count not in concept_count_accuracy:
                concept_count_accuracy[concept_count] = {"total": 0, "correct": 0}
            concept_count_accuracy[concept_count]["total"] += 1

            if 'correctness' in result and all(result['correctness'].values()):
                concept_count_accuracy[concept_count]["correct"] += 1
            if concept_count >= 2:
                two_or_more_concepts_accuracy["total"] += 1
                if 'correctness' in result and all(result['correctness'].values()):
                    two_or_more_concepts_accuracy["correct"] += 1

    concept_count_accuracy = {count: round(acc["correct"] / acc["total"], 4) for count, acc in
                              concept_count_accuracy.items() if
                              acc["total"] > 0}
    if two_or_more_concepts_accuracy["total"] > 0:
        two_or_more_concepts_accuracy = round(two_or_more_concepts_accuracy["correct"] / two_or_more_concepts_accuracy[
            "total"], 4)
    else:
        two_or_more_concepts_accuracy = None

    return concept_count_accuracy, two_or_more_concepts_accuracy


# Functions to evaluate accuracy
def evaluate_accuracy_by_question(results):
    correct = sum('correctness' in result and all(result['correctness'].values()) for result in results)
    total = len(results)
    return round(correct / total, 4) if total > 0 else 0


def evaluate_accuracy_by_sub_question(results):
    correct = sum(sum(result['correctness'].values()) for result in results if 'correctness' in result)
    total = sum(len(result['correctness']) for result in results if 'correctness' in result)
    return round(correct / total, 4) if total > 0 else 0


def evaluate_accuracy_proportional_by_sub_question_adjusted(results):
    total_score = 0
    for result in results:
        if 'correctness' in result:
            sub_question_count = len(result['correctness'])
            score_per_sub_question = 1 / sub_question_count if sub_question_count > 0 else 0
            question_score = sum(result['correctness'].values()) * score_per_sub_question
            total_score += question_score
    return round(total_score / len(results), 4) if results else 0


# Main function to run the evaluation
def main():
    parser = argparse.ArgumentParser(description="Evaluate closed-form dataset responses.")
    parser.add_argument('--questions_file_path', type=str, default="data/da-dev-questions.jsonl",
                        help='Path to the questions file (JSONL format)')
    parser.add_argument('--labels_file_path', type=str, default="data/da-dev-labels.jsonl",
                        help='Path to the labels file (JSONL format)')
    parser.add_argument('--response_files_path', type=str, default="inference_output/Llama-3.1-8B-Instruct",
                        help='Path to the response files (JSON format)')
    # parser.add_argument('--responses_file_path', type=str, required=True,
    #                     help='Path to the responses file (JSONL format)')
    args = parser.parse_args()

    os.makedirs("eval_outputs", exist_ok=True)

    # Reading files
    labels = read_jsonl(args.labels_file_path)
    
    # seach for response files of each question based on 'id' in label
    responses = []
    for label in labels:
        qid = label["id"]
        response_file = os.path.join(args.response_files_path, f"{qid}.json")
        
        if os.path.exists(response_file):
            # if the file exists, read the response
            with open(response_file, 'r') as f:
                data = json.load(f)
                response = data['formatted_answer']
                responses.append({"id": qid, "response": response})
        else:
            # if the file does not exist, mark as empty response
            responses.append({"id": qid, "response": ""})
    
    print(f"Loaded {len(labels)} labels and {len(responses)} responses.")
    print(f"Missing response files: {len([r for r in responses if not r['response']])}")
    concepts_data = read_concepts_from_file(args.questions_file_path)

    # Evaluating responses
    results = evaluate_responses(labels, responses)

    print("\n========== Incorrect Predictions ==========\n")
    for r in results:
        if "correctness" in r and not all(r["correctness"].values()):
            print(f"ID: {r['id']}:")
            wrong_fields = [k for k, v in r["correctness"].items() if not v]
            for k in wrong_fields:
                print(f"Answer: {r['predicted_answers'].get(k)} | Correct: {r['label_answers'][k]}\n")
    print("\n===========================================\n")
    
    print(f"Total questions evaluated: {len(results)}")
    print(f"Questions with responses: {len([r for r in results if r['predicted_answers']])}")
    print(f"Questions without responses: {len([r for r in results if not r['predicted_answers']])}")

    # Calculate accuracies
    accuracy_by_question = evaluate_accuracy_by_question(results)
    accuracy_by_sub_question = evaluate_accuracy_by_sub_question(results)
    accuracy_proportional_by_sub_question = evaluate_accuracy_proportional_by_sub_question_adjusted(results)


    # Print results
    print(f"Accuracy by Question: {accuracy_by_question:.2%}")
    print(f"Accuracy Proportional by Sub-Question: {accuracy_proportional_by_sub_question:.2%}")
    print(f"Accuracy by Sub-Question: {accuracy_by_sub_question:.2%}")
    main_results = {"Accuracy by Question": accuracy_by_question,
                    "Accuracy Proportional by Sub-Question": accuracy_proportional_by_sub_question,
                    "Accuracy by Sub-Question": accuracy_by_sub_question}

    # Analyzing concepts accuracy
    concept_accuracy_analysis = analyze_concepts_accuracy(results, concepts_data)

    # Analyzing concepts count accuracy
    concept_count_accuracy_analysis, two_or_more_concepts_accuracy = analyze_concepts_count_accuracy(results,
                                                                                                     concepts_data)

    # Print results
    print("Concept Accuracy Analysis:")
    for concept, accuracy in concept_accuracy_analysis.items():
        print(f"{concept}: {accuracy:.2%}")

    print("\nConcept Count Accuracy Analysis:")
    for count, accuracy in concept_count_accuracy_analysis.items():
        print(f"{count} Concept(s): {accuracy:.2%}")

    # Print accuracy for questions with >= 2 concepts
    if two_or_more_concepts_accuracy is not None:
        print(f"\nAccuracy for Questions with >= 2 Concepts: {two_or_more_concepts_accuracy:.2%}")
    else:
        print("\nNo data available for questions with >= 2 Concepts.")

    #responses_file_name = os.path.splitext(os.path.basename(args.responses_file_path))[0]
    responses_file_name = os.path.basename(args.response_files_path)
    eval_file_name = f'eval_outputs/{responses_file_name}_evaluation_analysis.json'

    with open(eval_file_name, 'w') as file:
        json.dump({
            "main_results": main_results,
            "concept_accuracy_analysis": concept_accuracy_analysis,
            "concept_count_accuracy_analysis": concept_count_accuracy_analysis,
            "two_or_more_concepts_accuracy": two_or_more_concepts_accuracy
        }, file, indent=4)


if __name__ == "__main__":
    main()