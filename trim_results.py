import json

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--source_path", type=str)
parser.add_argument("--target_path", type=str)
args = parser.parse_args()

with open(args.source_path, "r") as f:
    final_result = json.load(f)

steps = [
    {
        "instruction": element["instruction"],
        "prediction": element["prediction"].split("\n")[0]
    }
    for element in final_result
]

with open(args.target_path, "w") as f:
    json.dump(steps, f, indent=4, ensure_ascii=False)

print(f"Done writing to {args.target_path}")

"""
sbatch --wrap="python3 sft/turn_final_result_to_steps.py \
    --source_path /cluster/work/lawecon/Work/mbraasch/output/sample_decoding/Qwen--Qwen2-7B-Instruct/m1/final_results.json \
    --target_path /cluster/work/lawecon/Work/mbraasch/output/sample_decoding/Qwen--Qwen2-7B-Instruct/m1/first_steps.json"
"""