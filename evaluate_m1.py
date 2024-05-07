import os
import json

from typing import Tuple

from argparse import ArgumentParser, Namespace

def get_arguments() -> Namespace:
    """
    Gets the arguments from the command line or accepts a pre-defined list
    of arguments such that it can be used programatically.

    :param predefined_args: The pre-defined arguments.
    :return: The arguments.
    """
    parser = ArgumentParser()
    _get_args_from_cl(parser)
    return parser.parse_args()


def _get_args_from_cl(parser: ArgumentParser) -> ArgumentParser:
    """
    Adds all the arguments to the parser.

    :param parser: The parser to add the arguments to.
    :return: The parser with the added arguments.
    """
    parser.add_argument(
        "--predictions",
        type=str,
        help="Specifies the path to the config.",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        help="Specifies the path to the config.",
    )
    parser.add_argument(
        "--processed",
        type=str,
        help="Specifies the path to the config.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        help="Specifies the path to the config.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Specifies the path to the config.",
        default="/cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output/"
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="Specifies the path to the config."
    )
    parser.add_argument(
        "--predictions_name",
        type=str,
        help="Specifies the path to the config.",
        default="predictions.json"
    )
    parser.add_argument(
        "--postprocessed_name",
        type=str,
        help="Specifies the path to the config.",
        default="postprocessed.json"
    )
    parser.add_argument(
        "--metrics_name",
        type=str,
        help="Specifies the path to the config.",
        default="model_metrics.json"
    )
    return parser

def format_number(input_string):
    input_string = input_string \
        .replace(',', '') \
        .replace('.', '') \
        .replace('%', '') \
        .replace('$', '') \
        .replace('g', '') \
        .replace('美元', '') \
        .replace('4800/1000=', '') \
        .replace('7:00 AM', '7') \
        .replace('100-30=<<100-30=70>>70 more than Jill', '70') \
        .replace('th place', '')
    return input_string.split()[0]
    

def filter_number_m1(question: str):
    return int(format_number(question.split(">>")[-1].split()[0]))

def create_processed_results(    
    args: Namespace,
    verbose: bool = False
    ):
    data_path = os.path.join(args.output_dir, args.run_name, args.predictions_name)
    print(f"Loading data from {data_path}")
    with open(data_path, "r") as f:
        data = json.load(f)

    print(f"Starting to process predictions.")
    fail_counter = 0
    formatted_results = []
    for i, data_point in enumerate(data):

        failed = False
        try:
            ground_truth_value = filter_number_m1(data_point["input"])
        except:
            if verbose:
                print(f"Something went wrong with getting the ground truth value.")
                print(f"The input was:\n{data_point["input"]}")
            failed = True

        try:
            predicted_value = filter_number_m1(data_point["prediction"])
        except:
            if verbose:
                print(f"Something went wrong with getting sion.")
                print(f"The prediction was:\n{data_point["prediction"]}")
            failed = True

        if failed:
            print(f"Prediction for data point {i} failed.")
            fail_counter += 1
            if verbose:
                print(f"Ground truth:\n{ground_truth_value}")
                print(f"Predicted:\n{predicted_value}")
                print(f"Correct:\n{ground_truth_value == predicted_value}")
                print(f"Input:\n{data_point["input"]}")
                print(f"Prediction:\n{data_point["prediction"].replace("\n<|EOT|>", "")}\n")
            continue
        else:
            print(f"Prediction for data point {i} processed correctly.")

        formatted_results.append(
            {
                "ground_truth": ground_truth_value,
                "prediction": predicted_value,
                "is_correct": ground_truth_value == predicted_value
            }
        )

    failed = (round(fail_counter/len(data), 4), fail_counter)
    print(f"Failed: {failed}%")

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
Command:
sbatch --wrap="cd repos/sft; python3 evaluate_m1.py --run_name deepseek-7b-base-m1"
"""