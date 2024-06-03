"""
Some of the data is in a format that is very nice to evaluate on so we filter it out.
Specifically, some of the problem instances do not have the the << ... >> format. 
It helps the model structure the problem better and indicates when a subsequent step is needed.

"""

import os
import json

raw_data_dir = "data/raw"
filtered_data_dir = "data/filtered/"

data_paths = ["test.json", "train.json", "test_socratic.json", "train_socratic.json"]

if not os.path.exists(filtered_data_dir):
    os.makedirs(filtered_data_dir)

for data_path in data_paths:

    with open(os.path.join(raw_data_dir, data_path), "r") as f:
        data = json.load(f)
    
    
    filtered_data = []
    for problem in data:
        if all([">>" in x for x in problem["output"].split("\n")[:-1]]) and len(problem["output"].split("\n")) <= 9:
            filtered_data.append(problem)
        
    #     # Filter out problems that do not have the `>>` format
    #     if  problem["output"].split("\n")[0]
    #     # Filter out problems that have more than 8 steps (error in dataset)
    #     and len(problem["output"].split("\n")) <= 9
    # ]

    with open(os.path.join(filtered_data_dir, data_path), "w") as f:
        json.dump(filtered_data, f, indent=4, ensure_ascii=False)