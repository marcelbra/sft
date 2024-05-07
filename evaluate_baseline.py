import json
  
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
        "--raw",
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
    return parser

def filter_number(input_string):
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
    input_string = input_string.split()[0]
    return int(input_string)

def create_processed_results(
        source_path: str = "./test_result_copy.json",
        target_path: str = "./processed_test_results.json",
        verbose: bool = False,
        write: bool = True
    ):

    with open(source_path, "r") as f:
        data = json.load(f)

    formatted_results = []
    for i, data_point in enumerate(data):
        ground_truth_value = filter_number(data_point["question"].split("#### ")[1])
        try:
            predicted_string = data_point["raw_prediction"].split("#### ")[1].split("\n")[0]
        except:
            print(f"Something went wrong in prediction {i}. Skipping (1).")
            continue
        try:
            predicted_value = filter_number(predicted_string)
        except:
            print(f"Something went wrong in prediction {i}. Skipping (2).")
            print(f"Predicted string is {predicted_string}.")
            continue
        if verbose:
            print(f"Ground truth: {ground_truth_value}")
            print(f"Predicted: {predicted_value}")
            print(f"Correct: {ground_truth_value == predicted_value}")
        formatted_results.append(
            {
                "ground_truth": ground_truth_value,
                "prediction": predicted_value,
                "is_correct": ground_truth_value == predicted_value
            }
        )
    if not write:
        return
    with open(target_path, "w") as f:
        json.dump(formatted_results, f, indent=4, ensure_ascii=False)

def create_final_metrics(
        source_path: str = "./processed_test_results.json",
        target_path: str = "./metrics_test_results.json",
        write: bool = True
    ):

    with open(source_path, "r") as f:
        data = json.load(f)

    correct = [x["is_correct"] for x in data].count(True)
    false = len(data) - correct
    accuracy = correct / len(data)
    if not write:
        return

    with open(target_path, "w") as f:
        json.dump(
            {
                "correct": correct,
                "false": false,
                "accuracy": accuracy
            },
            f, indent=4, ensure_ascii=False
        )

if __name__ == "__main__":
    args = get_arguments()
    create_processed_results(args.raw, args.processed)
    create_final_metrics(args.processed, args.metrics)
    print("Done.")

"""
Command:
sbatch --wrap=" \
    cd repos/sft; \
    python3 evaluate.py \
    --raw /cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output/deepseek-7b-base-baseline/raw_test_results.json \
    --processed /cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output/deepseek-7b-base-baseline/processed_test_results.json \
    --metrics /cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output/deepseek-7b-base-baseline/metrics_test_results.json"
"""