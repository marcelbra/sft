import os
import json

from utils import nested_dict

from collections import defaultdict
from argparse import ArgumentParser

def format_number(input_string: str) -> int:
    input_string = input_string \
        .replace(',', '') \
        .replace('%', '') \
        .replace('$', '') \
        .replace('美元', '') \
        .replace('4800/1000=', '') \
        .replace('7:00 AM', '7') \
        .replace('100-30=<<100-30=70>>70 more than Jill', '70') \
        .replace('150kg', '150') \
        .replace('th place', '') \
        .replace('/year', '') \
        .replace('/month', '') \
        .replace('/week', '') \
        .replace('/day', '') \
        .replace('/hour', '') \
        .replace('/minute', '') \
        .replace('cm', '') \
        .replace('ml', '') \
        .replace('m', '') \
        .replace('kg', '') \
        .replace('g', '') \
        .replace('/task', '') \
        .replace('\"', '') \
        .replace('/sandwich', '') \
        .split()[0]
    if input_string.endswith("."):
        input_string = input_string[:-1]
    return float(input_string)

def filter_(by: str, step: int, question: str) -> str:
    return format_number(question.split(by)[step].split("\n")[0])

def evaluate(final_results, eval_folder_path, ground_truth_path, postfix):

    comparison = defaultdict(lambda: defaultdict(list))
    
    # Get the predictions
    for final_result in final_results:
        with open(final_result, "r") as f:
            data = json.load(f)
        for element in data:
            question = element["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:\n")[0]
            prediction = element["result"]
            comparison[question]["prediction"] = prediction
                
    # Get the ground truth
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)
    
    for element in ground_truth:
        question = element["source_question"]
        ground_truth = element["target_result"].replace(",", "")
        comparison[question]["ground_truth"] = ground_truth
    
    # Write the comparison to a file
    with open(os.path.join(eval_folder_path, f"comparison{postfix}.json"), "w") as f:
        json.dump(comparison, f, indent=4, ensure_ascii=False)
    
    # Now calculate the accuracy
    correct = 0
    total = 0
    for question, data in comparison.items():
        total += 1
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
                "n": total,
            }
            , f, indent=4, ensure_ascii=False)
    
    print(f"Accuracy: {round(accuracy * 100, 4)}%")


def calc_metrics(
    test_data_path = "/cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/data/test.json",
    output_dir ="/cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it"
    ):
    results = nested_dict(n=2, type_=dict)
    with open(test_data_path, "r") as f:
        ground_truth = json.load(f)
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
    argparser.add_argument("--final_result", type=str, required=True)
    argparser.add_argument("--postfix", type=str, required=True)
    args = argparser.parse_args()

    evaluate([args.final_result], args.eval_folder_path, args.ground_truth_path, args.postfix)

    """
    Example:
    sbatch --time=00:01:00 --wrap="python3 sft/evaluate.py \
    --eval_folder_path /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/test_best_2_and_3_step_solver--4/2_steps \
    --final_result     /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/test_best_2_and_3_step_solver--4/2_steps/final_results_test_2_steps.json \
    --ground_truth_path /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/test_2_steps.json \
    --postfix _2_steps"

    sbatch --time=00:01:00 --wrap="python3 sft/evaluate.py \
    --eval_folder_path /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/test_best_2_and_3_step_solver--4/23_steps \
    --final_result     /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/test_best_2_and_3_step_solver--4/23_steps/final_results_test_2_steps.json \
    --ground_truth_path /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/test_2_steps.json \
    --postfix _2_steps"

    sbatch --time=00:01:00 --wrap="python3 sft/evaluate.py \
    --eval_folder_path /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/test_best_2_and_3_step_solver--4/23_steps \
    --final_result     /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/test_best_2_and_3_step_solver--4/23_steps/final_results_test_3_steps.json \
    --ground_truth_path /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/test_3_steps.json \
    --postfix _3_steps"
    """
