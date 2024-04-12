"""
Splits the raw data into a nested dict with the sub-data.
"""

import json
from utils import nested_dict

# Counts for test dataset, compares the newly formatted dataset with the original.

# COUNTS = {
#     2: 326,
#     3: 370,
#     4: 298,
#     5: 174,
#     6: 88,
#     7: 40,
#     8: 20
# }

# CUMMULATED_AMOUNT = {
#     2: 326 + 370 + 298 + 174 + 88 + 40 + 20,
#     3: 370 + 298 + 174 + 88 + 40 + 20,
#     4: 298 + 174 + 88 + 40 + 20,
#     5: 174 + 88 + 40 + 20,
#     6: 88 + 40 + 20,
#     7: 40 + 20,
#     8: 20
# }

def split_dataset(
    dataset_path: str = 'data/test.jsonl',
    save_path: str = 'data/formatted_data.json',
    save_to_disk: bool = False
):

    # Load the jsonl
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]

    formatted_dataset = nested_dict()
    counter = nested_dict(n=1, type_=int)
    for data_point in data:
        question = data_point["question"]
        split_answer = data_point["answer"].split("\n")
        m = len(split_answer) - 1
        if m > 8:
            # There are samples in the OpenAI dataset that do not
            # comply with the claimed format. Here we filter them out.
            continue
        counter[m] += 1
        for i in range(m):
            previous_steps = "\n".join(split_answer[:i])
            last_step = split_answer[i]
            for type_ in ["without_result", "with_result"]:
                formatted_dataset[type_][i+1].append(
                    {
                        "question": question + ("\n" if previous_steps else "") + previous_steps,
                        "answer": last_step
                    }
                )
            if i == m-1:
                # Append the extracted answer, e.g., '#### 16' to the last step
                formatted_dataset["with_result"][i+1][-1]["answer"] += "\n"+ split_answer[-1]

    # The tests for the counts
    # TODO: write test for this
    # for i in range(2, 9):
    #     if CUMMULATED_AMOUNT[i] != len(formatted_dataset["without_result"][i]):
    #         raise ValueError(f"Amount of {i} step questions is not the same as expected")
    #     if CUMMULATED_AMOUNT[i] != len(formatted_dataset["with_result"][i]):
    #         raise ValueError(f"Amount of {i} step questions is not the same as expected")
        
    # Save the formatted dataset
    if save_to_disk:
        with open(save_path, 'w') as f:
            json.dump(formatted_dataset, f, ensure_ascii=False, indent=4)

    return formatted_dataset
