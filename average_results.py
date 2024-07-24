import os
import json

experiment_name = "/cluster/work/lawecon/Work/mbraasch/output/TEC"

runs = list(range(1, 6))
experts = ["m2345678", "m345678", "m45678"]
datasets = ["socratic", "dl", "gt"]
models = [
    "google--gemma-1.1-2b-it",
    "mistralai--Mistral-7B-Instruct-v0.3",
    "deepseek-ai--deepseek-llm-7b-chat"
]
postfixes = [
    "_TEC_from_gt_1_BL",
    "_TEC_from_gt_2_BL",
    "_TEC_from_gt_3_BL"
]

for model in models:
    for dataset in datasets:
        for postfix in postfixes:
            
            accs = []
            for run in runs:
                file_path = os.path.join(experiment_name, model, dataset, "m12345678", str(run), f"accuracy{postfix}.json")
                with open(file_path, "r") as f:
                    accs.append(json.load(f)["accuracy"])
            avg_acc = sum(accs) / len(accs)
            
            # Write to the output file
            output_file = os.path.join(experiment_name, model, dataset, "m12345678", f"accuracy{postfix}.json")
            with open(output_file, "w") as f:
                json.dump({"accuracy": avg_acc}, f)

            # Print accuracy to the console for easy copy-pasting
            print(model, dataset, postfix)
            print(avg_acc)

"""
sbatch -wrap="python3 sft/average_results.py"
"""