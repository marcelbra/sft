"""
Splits the filtered data into a nested dict with the sub-data.
"""
import os
import json

from collections import defaultdict
from typing import Any, Optional

from sft.arguments import get_arguments
from sft.settings import DATA_DIR, OUTPUT_DIR

def nested_dict(n: int = 2, type_: Any = list, obj: Optional[dict] = None):
    
    """
    Creates nested dicts (`defaultdict`s of any depth with any type).
    Additionally, can take an arbitrarily deep nested dictionary and
    turn it into its `defaultdict` correspondence.

    :param n: Number of depth levels.
    :param type_: Type of the nested dict
    :param obj: The nested dict to turn into a `defaultdict`.
    :return:
    """
    if n <= 0:  # Base case
        return obj if obj else type_()

    if obj:  # Recursive case
        return defaultdict(lambda: type_(), {k: nested_dict(n=n - 1, type_=type_, obj=v) for k, v in obj.items()})

    return defaultdict(lambda: nested_dict(n=n - 1, type_=type_))


def split_dataset(data):

    formatted_dataset = defaultdict(list)

    for data_point in data:
        
        question = data_point["instruction"]
        split_output = data_point["output"].split("\n")
        steps = len(split_output) - 1

        for i in range(steps):
            formatted_dataset[i+1].append(
                {
                    "source_question": question,
                    "source_steps": "\n".join(split_output[:i]),
                    "target_steps": "\n".join(split_output[i:i+1]),
                    "target_result": f"{split_output[-1].replace("#### ", "")}" if i == steps-1 else ""
                }
            )

    return formatted_dataset


def create_target_dirs(target_data_dir: str):
    splits = ["test", "train"]
    types = ["normal", "socratic"]
    for split in splits:
        for type_ in types:
            path = os.path.join(target_data_dir, split, type_)
            if not os.path.exists(path):
                os.makedirs(path)


def split(args):
    

    source_data_path = "/cluster/work/lawecon/Work/mbraasch/data/raw"
    data_paths = [f for f in os.listdir(source_data_path) if f.endswith(".json")]
    # data_paths = ["test.json", "train.json", "test_socratic.json", "train_socratic.json"]
    taget_data_dir = os.path.join(OUTPUT_DIR, args.run_name)
    create_target_dirs(target_data_dir=taget_data_dir)
    
    for data_path in data_paths:
        
        # Load source data
        with open(os.path.join(source_data_path, data_path), "r") as f:
            data = json.load(f)

        if args.filter:
            filtered_data = []
            for problem in data:
                if all([">>" in x for x in problem["output"].split("\n")[:-1]]) and len(problem["output"].split("\n")) <= 9:
                    filtered_data.append(problem)
            data = filtered_data
            # Save the data

        formatted_dataset = split_dataset(data)
        split = "test" if "test" in data_path else "train"
        type_ = "socratic" if "socratic" in data_path else "normal"
        for i in range(1, 9):
            path = os.path.join(args.run_name, f"data/decomposed/{split}/{type_}/{i}.json")
            with open(path, "w") as f:
                json.dump(formatted_dataset[i], f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = get_arguments()
    split(args=args)