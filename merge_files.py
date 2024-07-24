import os
import glob
import json

what = "test"

path = f"/cluster/work/lawecon/Work/mbraasch/output/sample_decoding/mistralai--Mistral-7B-Instruct-v0.3/m1/nsp_{what}"

# Get all json files from the path
files = glob.glob(f"{path}/*.json")

# Iterate over the files and load from json
general_dict = {}
for file in files:
    if f"next_step_predictions_{what}.json" in file or what not in file:
        continue
    with open(file, "r") as f:
        data = json.load(f)
    print(len(data))
    # Merge the contents of data into general_dict
    general_dict.update(data)

print(f"There are {len(general_dict)} entries in the general_dict.")

# Write the file to disk
target_path = os.path.join(path, f"next_step_predictions_{what}.json")
print(f"Writing to {target_path}")
with open(target_path, "w") as f:
    json.dump(general_dict, f, ensure_ascii=False, indent=4)

# sbatch --wrap="python3 sft/merge_files.py"