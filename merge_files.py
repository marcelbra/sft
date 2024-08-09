import os
import glob
import json

# from argparse import ArgumentParser
# parser = ArgumentParser()
# parser.add_argument("--source_path", type=str)
# parser.add_argument("--target_name", type=str)
# parser.add_argument("--type", type=str, default="dict")
# args = parser.parse_args()

hf_models= ["Qwen/Qwen2-0.5B-Instruct", "Qwen/Qwen2-1.5B-Instruct", "Qwen/Qwen2-7B-Instruct"]
temps = ["0.3", "0.4", "0.5", "0.6", "0.7"]
test_train = ["test", "train"]
hf_models= ["Qwen/Qwen2-7B-Instruct"]
temps = ["0.5"]
test_train = ["train"]
type_ = "dict"

for hf_model in hf_models:
    for temp in temps:
        for tt in test_train:
        
            # Get all json files from the path
            model_name = hf_model.replace("/", "--")
            source_path = f"/cluster/work/lawecon/Work/mbraasch/output/sample_decoding/{model_name}/m1/nsp_{tt}_t={temp}"
            target_name = f"nsp_{tt}_t={temp}.json"
            files = glob.glob(f"{source_path}/*.json")
            print(f"There are {len(files)} files.")

            # Iterate over the files and load from json
            general = {} if type_ == "dict" else []
            for file in files:
                with open(file, "r") as f:
                    data = json.load(f)
                if type_ == "dict":
                    general.update(data)
                else:
                    general += data

            # Write the file to disk
            print(f"There are {len(general)} entries ({hf_model, temp, tt}).")
            target_path = os.path.join(source_path, target_name)
            print(f"Writing to {target_path}")
            with open(target_path, "w") as f:
                json.dump(general, f, ensure_ascii=False, indent=4)

"""
sbatch --gpus=rtx_4090:1 --wrap="python3 sft/merge_files.py"
"""