"""
Script that turns the data from train_formatted_data.json into a directory structure with the following format:'

type_/
    m/
        data.json

where data.json contains a list of dictionaries with the following format:

{
    "input": "question",
    "output": "answer"
}
"""

import json
import os

with open("train_formatted_data.json", 'r') as f:
    data = json.load(f)

for type_, m_data in data.items():
    # Create directory with name type_ if it doesn't exist yet
    if not os.path.exists(type_):
        os.makedirs(type_)

    for m, data in m_data.items():

        # Create directory with name m in type_ if it doesn't exist yet
        if not os.path.exists(type_ + "/" + m):
            os.makedirs(type_ + "/" + m)

        # Write data to file

        # In data (list of dicts), for every dict change the key question to input and the key answer to output
        for d in data:
            d["input"] = d.pop("question")
            d["output"] = d.pop("answer")


        with open(type_ + "/" + m + "/data.json", 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        