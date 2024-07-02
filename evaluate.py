import os
import json

from utils import nested_dict

from collections import defaultdict
from argparse import ArgumentParser

# def format_number(input_string: str) -> int:
#     input_string = input_string \
#         .replace(',', '') \
#         .replace('%', '') \
#         .replace('$', '') \
#         .replace('美元', '') \
#         .replace('4800/1000=', '') \
#         .replace('7:00 AM', '7') \
#         .replace('100-30=<<100-30=70>>70 more than Jill', '70') \
#         .replace('150kg', '150') \
#         .replace('th place', '') \
#         .replace('/year', '') \
#         .replace('/month', '') \
#         .replace('/week', '') \
#         .replace('/day', '') \
#         .replace('/hour', '') \
#         .replace('/minute', '') \
#         .replace('cm', '') \
#         .replace('ml', '') \
#         .replace('m', '') \
#         .replace('kg', '') \
#         .replace('g', '') \
#         .replace('/task', '') \
#         .replace('\"', '') \
#         .replace('/sandwich', '') \
#         .split()[0]
#     if input_string.endswith("."):
#         input_string = input_string[:-1]
#     return float(input_string)

# def filter_(by: str, step: int, question: str) -> str:
#     return format_number(question.split(by)[step].split("\n")[0])

def evaluate(final_results, eval_folder_path, ground_truth_path, postfix):

    comparison = defaultdict(lambda: defaultdict(list))
    
    # Predictions
    for final_result in final_results:
        with open(final_result, "r") as f:
            data = json.load(f)
        for element in data:
            question = element["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:\n")[0]
            if "result" in element:
                prediction = element["result"]
            else:
                split_by_answer = element["prediction"].split("Final answer: ")
                if len(split_by_answer) > 1:
                    prediction = split_by_answer[-1]
                else:
                    continue
            comparison[question]["prediction"] = prediction
                
    # Ground truth
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    for element in ground_truth:
        question = element["source_question"]
        comparison[question]["ground_truth"] = element["target_result"].replace(",", "")
    
    # Write the comparison to a file
    with open(os.path.join(eval_folder_path, f"comparison{postfix}.json"), "w") as f:
        json.dump(comparison, f, indent=4, ensure_ascii=False)
    
    # Now calculate the accuracy
    correct = 0
    predicted = 0
    total = len(ground_truth)
    for question, data in comparison.items():
        if "prediction" not in data or "ground_truth" not in data:
            continue
        predicted += 1
        if data["prediction"] == data["ground_truth"]:
            correct += 1
    
    accuracy = correct / total
    
    # Write the accuracy to a file
    with open(os.path.join(eval_folder_path, f"accuracy{postfix}.json"), "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "correct": correct,
                "false": total - correct,
                "predicted": predicted,
                "missed": total - predicted,
                "n": total,
            }
            , f, indent=4, ensure_ascii=False)
    
    print(f"Accuracy: {round(accuracy * 100, 4)}%")
    print()


def calc_metrics(
    test_data_path = "/cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/data/test.json",
    output_dir ="/cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it"
    ):
    results = nested_dict(n=2, type_=dict)
    with open(test_data_path, "r") as f:
        ground_truth = json.load(f)
        print(len(ground_truth))
    for element in ground_truth:
        question = element["source_question"]
        results[question]["ground_truth"] = element["target_result"]

    # Get prediction
    predicted_result_path = os.path.join(output_dir, "final_results.json")
    with open(predicted_result_path, "r") as f:
        predicted_result = json.load(f)

    for element in predicted_result:
        question = element["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:\n")[0]
        results[question]["predicted"] = element["result"]

    # For those where there is no prediction, add "None" as a prediction to the results dict
    for question in results:
        if "predicted" not in results[question]:
            results[question]["predicted"] = "None"

    # Calculate error
    for question in results:
        results[question]["correct"] = results[question]["predicted"] == results[question]["ground_truth"]

    # Write results to file
    results_path = os.path.join(output_dir, "final_comparison.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    # Write metrics to file
    metrics = {
        "n": len(results),
        "correct": sum([results[question]["correct"] for question in results]),
        "incorrect": len(results) - sum([results[question]["correct"] for question in results]),
        "accuracy": sum([results[question]["correct"] for question in results]) / len(results)
    }

    metrics_path = os.path.join(output_dir, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print("Done.")

if __name__ == "__main__":

    argparser = ArgumentParser()
    argparser.add_argument("--eval_folder_path", type=str, required=True)
    argparser.add_argument("--ground_truth_path", type=str, required=True)
    argparser.add_argument("--final_result", type=str, required=True, nargs='+')
    argparser.add_argument("--postfix", type=str, default="")
    args = argparser.parse_args()
    # print(args.final_result[1])
    evaluate(args.final_result, args.eval_folder_path, args.ground_truth_path, args.postfix)

"""
Example:
sbatch --time=00:01:00 --wrap="python3 sft/evaluate.py \
--eval_folder_path  /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/moe--filter-previous-data/8/1 \
--final_result      /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/moe--filter-previous-data/8/1/final_results.json \
--ground_truth_path /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/moe--filter-previous-data/8/test.json"
"""
