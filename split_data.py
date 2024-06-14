"""
Splits the filtered data into a nested dict with the sub-data.
"""
import os
import json

from tqdm import tqdm
from collections import defaultdict

def split(data, merges):

    new_data = defaultdict(list)

    for element in tqdm(data):

        steps = element["target_steps"].count("\n") + 1

        for merge in merges:
        
            min_step  = min(merge)

            if steps < min_step:
                continue

            max_step = max(merge)
            split_steps = element["target_steps"].split("\n")

            new_data[merge].append(
                {
                    "source_question": element["source_question"],
                    "source_steps": "\n".join(split_steps[:min_step-1]),
                    "target_steps": "\n".join(split_steps[min_step-1:max_step]),
                    "target_result": element["target_result"] if max_step >= steps else ""
                }
            )

    # Rename the keys of `new_data`
    return {
        "".join(map(str, merge))[:5]: new_data[merge] for merge in merges
    }


if __name__ == "__main__":

    save_path = "teacher forcing data prep"
    type_ = "metamath"
    configs = []

    # amounts = [0.05, 0.1, 0.15, 0.2]
    # data_path = lambda x: f"/Users/marcelbraasch/decomposition-local/teacher forcing data prep/train_tf_{x}.json"

    # configs = [
    #     {
    #         "save_data_name": f"m123-m234-m345-final-t--tf-{amount}",
    #         "merges": [(1,2,3), (2,3,4), (3,4,5), tuple(range(4,71))],
    #         "data_path": data_path(amount)
    #     }
    #     for amount in amounts
    # ]

    ###### Distilled

    # if type_ == "distilled":

    #     data_path = "/Users/marcelbraasch/decomposition-local/data (to euler)/train_dl.json"

    #     configs = [
            # {
            #     "save_data_name": "dl-m1m2m3m4m5m6-",
            #     "merges": [(1,), (2,), (3,), (4,), (5,), tuple(range(6,100))]
            # },
            # {
            #     "save_data_name": "dl-m12-m34-m5-final",
            #     "merges": [(1,2), (3,4), tuple(range(5,71))]
            # },
        #     {
        #         "save_data_name": "dl-m123-m234-m345-m5-",
        #         "merges": [(1,2,3), (2,3,4), (3,4,5), tuple(range(5,100))]
        #     }
        # ]

    ###### Ground Truth

    # elif type_ == "ground_truth":

    #     data_path = "/Users/marcelbraasch/decomposition-local/data (to euler)/train_gt.json"

    #     configs = [
            # {
            #     "save_data_name": "gt-m1m2m3m4m5m6-2",
            #     "merges": [(1,), (2,), (3,), (4,), (5,), tuple(range(6,100))]
            # },
            # {
            #     "save_data_name": "m12-m34-final",
            #     "merges": [(1,2), (3,4), tuple(range(5,71))]
            # },
            # {
            #     "save_data_name": "gt-m123-m234-m345-m5-",
            #     "merges": [(1,2,3), (2,3,4), (3,4,5), tuple(range(5,100))]
            # }
        # ]

    ###### MetaMath

    # if type_ == "metamath":

    #     data_path = "/Users/marcelbraasch/decomposition-local/data (to euler)/train_mm.json"
        
    #     configs = [
            # {
            #     "save_data_name": "mm-m1-m2-m3-m4-m5-m6-",
            #     "merges": [(1,), (2,), (3,), (4,), (5,), tuple(range(6,100))]
            # },
            # {
            #     "save_data_name": "mm-m12-m34-m5-",
            #     "merges": [(1,2), (3,4), tuple(range(5,71))]
            # },
            # {
            #     "save_data_name": "mm-m123-m234-m345-m5-",
            #     "merges": [(1,2,3), (2,3,4), (3,4,5), tuple(range(5,100))]
            # },
        #     {
        #         "data_path": "/Users/marcelbraasch/decomposition-local/data (to euler)/train_mm.json",
        #         "save_data_name": "mm-m123-m234-m345-m4-",
        #         "merges": [(1,2,3), (2,3,4), (3,4,5), tuple(range(4,100))]
        #     }
        # ]


    for config in configs:
        data_path = config["data_path"]
        with open(os.path.join(data_path), "r") as f:
            data = json.load(f)
        save_data_name = config["save_data_name"]
        merges = config["merges"]
        formatted_dataset = split(data, merges)

        # Save data
        curr_save_path = os.path.join(save_path, save_data_name)
        if not os.path.exists(curr_save_path):
            os.makedirs(curr_save_path)
        for step, data in formatted_dataset.items():
            with open(os.path.join(curr_save_path, f"{step}.json"), "w") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
