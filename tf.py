import os
import json
import glob
import random

from tqdm import tqdm
from copy import deepcopy
from collections import defaultdict
from typing import Any, Optional

random.seed(42)

def nested_dict(n: int = 2, type_: Any = list, obj: Optional[dict] = None):
    if n <= 0:  # Base case
        return obj if obj else type_()

    if obj:  # Recursive case
        return defaultdict(lambda: type_(), {k: nested_dict(n=n - 1, type_=type_, obj=v) for k, v in obj.items()})

    return defaultdict(lambda: nested_dict(n=n - 1, type_=type_))

def create_tf_data_from_corresponding_model(
    previous_model_predictions: str,
    target_data_dir: str,
    ground_truth_data,
    target_model: str
):
    
    # Load the ground truth data
    with open(ground_truth_data, "r") as f:
        ground_truth_data = json.load(f)
    
    # Map question to its index for easy lookup
    question_to_index = {element["source_question"]: i for i, element in enumerate(ground_truth_data)}
    total_n = len(ground_truth_data)

    # Recursively get all full paths where there is a file "s1_next_step_predictions_s1.json" in the previous_model_predictions directory
    previous_model_predictions_paths = glob.glob(f"{previous_model_predictions}/**/s1_next_step_predictions_s1.json", recursive=True)
    
    for path in previous_model_predictions_paths:
        
        percentage_rest = path.split("-tf-")[1].split("-")
        percentage = float(percentage_rest[0])
        strategy = percentage_rest[1].split("/")[0]
        print("Calculating new dataset.")
        print(f"Percentage: {percentage}, Strategy: {strategy}")

        # Load predictions
        with open(path, "r") as f:
            previous_model_predictions = json.load(f)

        # Filter out predictions that are not in the ground truth data. We need to do this
        # because the distilled data is only a subset (~80%) of the ground truth data
        previous_model_predictions = [
            {
                "instruction": element["instruction"],
                "prediction": element["prediction"]
            }
            for element in previous_model_predictions
            if element["instruction"].split("\n\n### Input:\n")[1] \
            .split("\n\n### Response:\n")[0] in question_to_index
        ]

        amount = int(total_n * percentage)
        note = ""
        if amount > len(previous_model_predictions):
            note += f"Amount of predictions is smaller than the desired amount of {amount}.\n"
            note += f"Setting amount to the length of the predictions, which is {len(previous_model_predictions)}.\n"
            new_percentage = percentage * len(previous_model_predictions) / amount
            note += f"Corresponding overlap is {round(len(previous_model_predictions) / amount, 4)} "
            note += f"with a percentage of {round(new_percentage, 4)}."
            amount = len(previous_model_predictions)
            print(note)
        new_dataset = {
            "added": deepcopy(ground_truth_data),
            "replaced": deepcopy(ground_truth_data)
        }

        # Sample `amount` times from previous_model_predictions
        sampled_predictions = random.sample(previous_model_predictions, amount)

        # Check for those sampled predictions if the question is in the ground truth data
        count = 0
        for sampled_prediction in sampled_predictions:
            question = sampled_prediction["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:\n")[0]
            if question not in question_to_index:
                count += 1
        print(f"Amount of questions not in ground truth data: {count}")

        for sampled_prediction in sampled_predictions:

            # Get question, first step, and index of data point
            question = sampled_prediction["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:\n")[0]
            first_step = sampled_prediction["prediction"].split("\n")[0]
            index = question_to_index[question]

            # Strategy 1: Replace the first step of the data point with the prediction
            new_dataset["replaced"][index]["source_steps"] = first_step

            # Strategy 2: Add the prediction as a new data point
            data_point_copy = deepcopy(ground_truth_data[index])
            data_point_copy["source_steps"] = first_step
            new_dataset["added"].append(data_point_copy)

        # Save the new dataset
        print("Saving new dataset.")
        percentage_strategy_name = f"{target_model}-tf-{percentage}-{strategy}"
        save_path = os.path.join(target_data_dir, percentage_strategy_name)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_name = os.path.join(save_path, percentage_strategy_name + ".json")
        with open(file_name, "w") as f:
            json.dump(new_dataset[strategy], f, indent=4, ensure_ascii=False)
        if note:
            with open(os.path.join(save_path, "below_percentage_note.txt"), "w") as f:
                f.write(note)

def create_all_tf_data_from_single_model(
    previous_model_predictions: str,
    ground_truth_data: str,
    run_name: str,
    target_model: str,
    splits: list,
    strategies: list,
    only_first_step: bool = True
):

    with open(previous_model_predictions, "r") as f:
        previous_model_predictions = json.load(f)

    with open(ground_truth_data, "r") as f:
        ground_truth_data = json.load(f)

    # Map question to its index for easy lookup
    question_to_index = {element["source_question"]: i for i, element in enumerate(ground_truth_data)}

    # Filter out predictions that are not in the ground truth data. We need to do this
    # because the distilled data is only a subset (~80%) of the ground truth data
    previous_model_predictions = [
        {
            "instruction": element["instruction"],
            "prediction": element["prediction"]
        }
        for element in previous_model_predictions
        if element["instruction"].split("\n\n### Input:\n")[1] \
        .split("\n\n### Response:\n")[0] in question_to_index
    ]

    total_n = len(ground_truth_data)
    amounts = [int(total_n * split) for split in splits]

    new_datasets = nested_dict(n=2, type_=list)

    for amount, split in zip(amounts, splits):

        new_datasets["added"][split] = deepcopy(ground_truth_data)
        new_datasets["replaced"][split] = deepcopy(ground_truth_data)

        # Sample `amount` times from previous_model_predictions
        sampled_predictions = random.sample(previous_model_predictions, amount)

        # Check for those sampled predictions if the question is in the ground truth data
        count = 0
        for sampled_prediction in sampled_predictions:
            question = sampled_prediction["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:\n")[0]
            if question not in question_to_index:
                count += 1
        print(f"Amount of questions not in ground truth data: {count}")

        for sampled_prediction in sampled_predictions:

            # Get question, first step, and index of data point
            question = sampled_prediction["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:\n")[0]
            steps = sampled_prediction["prediction"]
            if only_first_step:
                steps = steps.split("\n")[0]
            index = question_to_index[question]

            # Strategy 1: Replace the first step of the data point with the prediction
            new_datasets["replaced"][split][index]["source_steps"] = steps

            # Strategy 2: Add the prediction as a new data point
            data_point_copy = deepcopy(ground_truth_data[index])
            data_point_copy["source_steps"] = steps
            new_datasets["added"][split].append(data_point_copy)

    # Save the new datasets
    print("Saving new datasets.")
    for split in tqdm(splits):
        for strategy in strategies:
            split_name = f"{target_model}-tf-{split}-{strategy}"
            folder_name = f"{run_name}/{target_model}/{split_name}"
            file_name = os.path.join(folder_name, split_name[1:] + ".json")
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
            with open(file_name, "w") as f:
                json.dump(new_datasets[strategy][split], f, indent=4, ensure_ascii=False)

if __name__ == "__main__":

    # split_size = 0.05
    # previous_model = "m234"
    # target_model = "m345"
    # previous_model_predictions = "/Users/marcelbraasch/decomposition-local/3s-overlap split tf/m234-m345/m234/m234-tf-0.05-added/s1_next_step_predictions_s1.json"
    # ground_truth_data = "/Users/marcelbraasch/decomposition-local/3s-overlap split tf/m234-m345/m345/345.json"
    # run_name = "/Users/marcelbraasch/decomposition-local/3s-overlap split tf/m234-m345"
    # strategies = ["added", "replaced"]
    # splits = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

    # create_all_tf_data_from_single_model(
    #     previous_model_predictions=previous_model_predictions,
    #     ground_truth_data=ground_truth_data,
    #     run_name=run_name,
    #     target_model=target_model,
    #     splits=splits,
    #     strategies=strategies
    # )

    previous_model_predictions_dir = "/Users/marcelbraasch/decomposition-local/3s-overlap split tf/m234-m345/m234"
    target_data_dir = "/Users/marcelbraasch/decomposition-local/3s-overlap split tf/m234-m345/m345"
    ground_truth_data = "/Users/marcelbraasch/decomposition-local/3s-overlap split tf/m234-m345/m345/345.json"
    # previous_model_predictions_dir = "/cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m234"
    # target_data_dir = "/cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m345"
    # ground_truth_data = "/cluster/work/lawecon/Work/mbraasch/output/gemma-2b-it/gsm8k-dl/3s-overlap-split-tf/m345/345.json"
    target_model = "m345"

    create_tf_data_from_corresponding_model(
        previous_model_predictions=previous_model_predictions_dir,
        target_data_dir=target_data_dir,
        ground_truth_data=ground_truth_data,
        target_model=target_model
    )