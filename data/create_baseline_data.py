import os
import json
from collections import defaultdict

filtered_data_dir = "data/filtered/"
data_paths = ["test.json", "train.json", "test_socratic.json", "train_socratic.json"]

def create(data):

    formatted_dataset = defaultdict(list)
    counter = defaultdict(int)

    baseline_data = []
    for data_point in data:
        steps, result = data_point["output"].split("\n#### ")
        baseline_data.append(
            {
                "question": data_point["instruction"],
                "steps": steps,
                "result": result
            }
        )
    return baseline_data

for data_path in data_paths:

    with open(os.path.join(filtered_data_dir, data_path), "r") as f:
        data = json.load(f)

    baseline_data = create(data)

    # Create directory if it doesn't exist
    dir_path = "data/baseline/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    save_path = os.path.join(dir_path, data_path)
    with open(save_path, "w") as f:
        json.dump(baseline_data, f, indent=4, ensure_ascii=False)


