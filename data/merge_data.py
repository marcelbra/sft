import os
import json

start, end = 5, 8
splits = ["train", "test"]
types = ["normal", "socratic"]
data_dir  = "data/decomposed/"

for split in splits:
    for type_ in types:

        seen_questions = []

        data_path = os.path.join(data_dir, split, type_, f"{start}.json")
        with open(data_path, "r") as f:
            start_data = json.load(f)

        reversed_steps = list(reversed(range(start + 1, end + 1)))
        
        for step in reversed_steps:

            data_path = os.path.join(data_dir, split, type_, f"{step}.json")
            with open(data_path, "r") as f:
                end_data = json.load(f)
            
            for data_point in end_data:
                question = data_point["source_question"]
                if question in seen_questions:
                    continue
                seen_questions.append(question)
                
                # Find the index of the question in the start data
                idx = -1
                for i, d in enumerate(start_data):
                    if d["source_question"] == question:
                        idx = i
                        break
                
                new_steps = data_point["source_steps"] \
                    .replace(start_data[idx]["source_steps"], "") \
                    + "\n" + data_point["target_steps"]
                if new_steps.startswith("\n"):
                    new_steps = new_steps[1:]
                start_data[idx]["target_steps"] = new_steps
                start_data[idx]["target_result"] = data_point["target_result"]

        # Save the merged data
        save_path = os.path.join(data_dir, split, type_, f"{start}_{end}.json") 
        with open(save_path, "w") as f:
            json.dump(start_data, f, indent=4, ensure_ascii=False)