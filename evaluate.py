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
    parser.add_argument("--ground_truth_file", type=str, help="(Source) Specifies the name of the file that has the ground truth.")
    parser.add_argument("--instruction_key", type=str, help="Specifies the name of the key that has the instruction.", default="instruction")
    parser.add_argument("--ground_truth_key", type=str, help="Specifies the name of the key that has the ground truth.", default="output")
    parser.add_argument("--verbose", type=bool, help="Specifies whether to print errors verbosely (for debugging).", default=False)
    parser.add_argument("--output_dir", type=str, help="Specifies the path to the directory where everything is happenung..", default="/cluster/work/lawecon/Work/mbraasch/output/")
    parser.add_argument("--predictions_name", type=str, help="(Source) Specifies the name of predictions file in the output directory.", default="predictions.json")
    parser.add_argument("--postprocessed_name", type=str, help="(Target) File name of evaluation (by sample).", default="postprocessed.json")
    parser.add_argument("--metrics_name", type=str, help="(Target) File name of summarized metrics.", default="model_metrics.json")
    parser.add_argument("--step", type=bool, help="Which step to evaluate.", default=None)
    return parser.parse_args()

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
    
def create_processed_results(args: Namespace) -> int:
    
    data_path = os.path.join(args.output_dir, args.run_name, args.predictions_name)
    print(f"Loading data from {data_path}")
    with open(data_path, "r") as f:
        samples = json.load(f)

    # If there is a ground_truth_file load it with its key
    if args.ground_truth_file:
        print(f"Loading ground truth from {args.ground_truth_file}")
        with open(args.ground_truth_file, "r") as f:
            ground_truth = json.load(f)

        # Make mapping from question to answer
        mapping = {x[args.instruction_key]: x[args.ground_truth_key] for x in ground_truth}

        # Replace ground truth with answer
        for i, sample in enumerate(samples):
            current_instruction = sample["instruction"].replace("### Instruction:\n", "").replace("\n### Response:\n", "")
            if current_instruction in mapping:
                samples[i]["ground_truth"] = mapping[current_instruction]
            # else:
            #     raise ValueError(f"Could not find ground truth for question {i}:\n{current_instruction}")
            # TODO: for some reason there are 3 questions missing in the ground truth file

    print(f"Starting to process predictions.")
    counter = 0
    fail_counter = 0
    skip_counter = 0
    formatted_results = []
    for i, sample in enumerate(samples):

        if ">>" not in sample["ground_truth"]:
            print("Skipping test sample because no '>>' found.")
            print(sample)
            skip_counter += 1
            continue
        counter += 1

        if ">>" not in sample["prediction"]:
            print("No intermediate step generated from model. Error.")
            formatted_results.append(
                {
                    "ground_truth": "not computed",
                    "prediction": None,
                    "is_correct": False
                }
            )
            continue

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

    print(f"Failed: {fail_counter}.")
    print(f"Skipped: {skip_counter}.")

    target_path = os.path.join(args.output_dir, args.run_name, args.postprocessed_name)
    print(f"Writing postprocessed result to {target_path}")
    with open(target_path, "w") as f:
        json.dump(formatted_results, f, indent=4, ensure_ascii=False)
    
    return fail_counter, skip_counter

def create_final_metrics(
    args: Namespace,
    failed: Tuple,
    write: bool = True
    ):

    data_path = os.path.join(args.output_dir, args.run_name, args.predictions_name)
    print(f"Loading original data from {data_path}")
    with open(data_path, "r") as f:
        samples = json.load(f)

    source_path = os.path.join(args.output_dir, args.run_name, args.postprocessed_name)
    print(f"Loading evaluated data from {source_path}")
    with open(source_path, "r") as f:
        data = json.load(f)

    correct = [x["is_correct"] for x in data].count(True)
    false = [x["is_correct"] for x in data].count(False)
    total_computed = correct + false
    accuracy = correct / total_computed
    adj_accuracy = correct / len(samples) # This conservatively assumes all skipped instances are just wrong

    if not write:
        return

    target_path = os.path.join(args.output_dir, args.run_name, args.metrics_name)
    print(f"Writing metrics to {target_path}.")
    metrics = {
        "correct": correct,
        "false": false,
        "total_evaluated": total_computed,
        "total": len(samples),
        "accuracy": accuracy,
        "adj_accuracy": adj_accuracy,
        "failed": failed[0],
        "skipped": failed[1],
    }
    print(f"Metrics:\n{metrics}")
    with open(target_path, "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    print(f"Wrote result to {target_path}")

if __name__ == "__main__":
    args = get_arguments()
    failed = create_processed_results(args=args)
    create_final_metrics(args=args, failed=failed)
    print("Done.")
