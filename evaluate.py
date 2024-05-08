import os
import json

from typing import Tuple
from argparse import ArgumentParser, Namespace
from functools import partial

def get_arguments() -> ArgumentParser:
    """
    Adds all the arguments to the parser.

    :param parser: The parser to add the arguments to.
    :return: The parser with the added arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, help="Specifies the run name. Used to load right folder from output dir.")
    parser.add_argument("--verbose", type=bool, help="Specifies whether to print errors verbosely (for debugging).", default=False)
    parser.add_argument("--output_dir", type=str, help="Specifies the path to the directory where everything is happenung..", default="/cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output/")
    parser.add_argument("--predictions_name", type=str, help="(Source) Specifies the name of predictions file in the output directory.", default="predictions.json")
    parser.add_argument("--postprocessed_name", type=str, help="(Target) File name of evaluation (by sample).", default="postprocessed.json")
    parser.add_argument("--metrics_name", type=str, help="(Target) File name of summarized metrics.", default="model_metrics.json")
    parser.add_argument("--step", type=bool, help="Which step to evaluate.", default=None)
    return parser.parse_args()

def format_number(input_string: str) -> int:
    return int(input_string \
        .replace(',', '') \
        .replace('.', '') \
        .replace('%', '') \
        .replace('$', '') \
        .replace('g', '') \
        .replace('美元', '') \
        .replace('4800/1000=', '') \
        .replace('7:00 AM', '7') \
        .replace('100-30=<<100-30=70>>70 more than Jill', '70') \
        .replace('th place', '') \
        .replace('/year', '') \
        .replace('/week', '') \
        .replace('/month', '') \
        .replace('cm', '') \
        .replace('m', '') \
        .split()[0])

def filter_(by: str, step: int, question: str) -> str:
    return format_number(question.split(by)[step].split("\n")[0])

def create_processed_results(args: Namespace) -> int:
    data_path = os.path.join(args.output_dir, args.run_name, args.predictions_name)
    print(f"Loading data from {data_path}")
    with open(data_path, "r") as f:
        samples = json.load(f)

    print(f"Starting to process predictions.")
    counter = 0
    fail_counter = 0
    formatted_results = []
    for i, sample in enumerate(samples):

        if ">>" not in sample["ground_truth"]:
            print("Skipping test sample because it does not have '>>' in it.")
            print(f"Sample:\n{sample['ground_truth']}")
            continue
        counter += 1

        if args.step:
            filter = partial(filter_, ">>", args.step)
        else:
            filter = partial(filter_, "#### ", 1)

        failed = False
        ground_truth_value = None
        try:
            ground_truth_value = filter(sample["ground_truth"])
        except:
            failed = True
        
        predicted_value = None
        try:
            predicted_value = filter(sample["prediction"])
        except:
            failed = True

        if failed:
            print(f"Prediction for data point {i} failed.")
            fail_counter += 1
            if args.verbose:
                print(filter)
                print(f"Ground truth:\n{ground_truth_value}")
                print(f"Predicted:\n{predicted_value}")
                print(f"Correct:\n{ground_truth_value == predicted_value}")
                print(f"Input:\n{sample['instruction']}")
                print(f"Ground truth:\n{sample['ground_truth']}")
                print(f"Prediction:\n{sample['prediction']}\n") #.replace("\n<|EOT|>", "")}\n")
            continue

        print(f"Prediction for data point {i} processed correctly.")

        formatted_results.append(
            {
                "ground_truth": ground_truth_value,
                "prediction": predicted_value,
                "is_correct": ground_truth_value == predicted_value
            }
        )

    failed = (round(fail_counter/counter, 4), fail_counter)
    print(f"Failed: {failed}.")

    target_path = os.path.join(args.output_dir, args.run_name, args.postprocessed_name)
    print(f"Writing postprocessed result to {target_path}")
    with open(target_path, "w") as f:
        json.dump(formatted_results, f, indent=4, ensure_ascii=False)
    
    return failed

def create_final_metrics(
    args: Namespace,
    failed: Tuple,
    write: bool = True
    ):

    source_path = os.path.join(args.output_dir, args.run_name, args.postprocessed_name)
    with open(source_path, "r") as f:
        data = json.load(f)

    correct = [x["is_correct"] for x in data].count(True)
    false = len(data) - correct
    accuracy = correct / len(data)
    if not write:
        return

    target_path = os.path.join(args.output_dir, args.run_name, args.metrics_name)
    print(f"Writing metrics to {target_path}.")
    metrics = {
        "correct": correct,
        "false": false,
        "accuracy": accuracy,
        "rel_fail": failed[0],
        "abs_fail": failed[1],
    }
    print(f"Metrics:\n{metrics}")
    with open(target_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = get_arguments()
    failed = create_processed_results(args=args)
    create_final_metrics(args=args, failed=failed)
    print("Done.")

"""
# Evaluate baseline performance overall
sbatch --wrap="python3 sft/evaluate.py --run_name deepseek-7b-base-baseline"

# Evaluate baseline step 1
sbatch --wrap=" \
    python3 sft/evaluate.py \
        --run_name deepseek-7b-base-baseline \
        --step 1 \
        --postprocessed_name postprocessed_step1.json \
        --metrics_name model_metrics_step1.json"
"""