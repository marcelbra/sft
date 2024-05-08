import os
import json

from typing import List

from tqdm import tqdm
from vllm import SamplingParams, LLMEngine, EngineArgs
from vllm.lora.request import LoRARequest

from arguments import get_inference_arguments
from inference_hf import (
    get_max_step_adapter_name
)


def format_question(question: str, formatting_template: str):
    return formatting_template.format(question.strip()).lstrip()


def initialize_engine(args) -> LLMEngine:
    print("Initialize the LLMEngine.")
    engine_args = EngineArgs(
        model=args.model_name_or_path,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        max_num_seqs=256
    )
    return LLMEngine.from_engine_args(engine_args)


def get_sampling_params():
    print("Initalizating sampling params.")
    return SamplingParams(
        temperature=0,
        max_tokens=2048,
        stop=["\n<|EOT|>"]
    )


def get_formatted_data(args, sampling_params, lora_request):
    data_path = os.path.join(args.data_dir, args.data_path)
    print(f"Load data from {data_path}")
    with open(data_path) as f:
        data = json.load(f)
    print("Format data.")
    formatted_data = [
        {
            "instruction": format_question(element["instruction"], args.formatting_template),
            "ground_truth": element["output"],
            "sampling_params": sampling_params,
            "lora_request": lora_request
        }
        for element in data
    ]
    print("Sample data:")
    print(*formatted_data[:10], sep="\n")
    if args.amount_samples:
        formatted_data = formatted_data[:args.amount_samples]
    return formatted_data


def get_lora_request():
    print("Get lora adapter.")
    adapter_path = os.path.join(args.output_dir, args.run_name)
    print(f"Getting max adapter checkpoint from {adapter_path}.")
    adapter_ckpt = get_max_step_adapter_name(adapter_path)
    print(f"Loaded ckpt {adapter_ckpt}.")
    return LoRARequest("lora", 1, adapter_ckpt)


def process_requests(engine: LLMEngine, data: List):
    print("Start processing.")
    request_id = 0
    results = []
    instruction_to_ground_truth = {}
    with tqdm(total=len(data)) as pbar:
        while data or engine.has_unfinished_requests():
            if data:
                element = data.pop(0)
                instruction_to_ground_truth[element["instruction"]] = element["ground_truth"]
                engine.add_request(
                    str(request_id),
                    element["instruction"],
                    element["sampling_params"],
                    lora_request=element["lora_request"]
                )
                request_id += 1

            for request_output in engine.step():
                if request_output.finished:
                    results.append(
                        {
                            "instruction": request_output.prompt,
                            "prediction": request_output.outputs[0].text
                        }
                    )
                    pbar.update(1)
    
    # Add ground truth
    for element in results:
        element["ground_truth"] = instruction_to_ground_truth[element["instruction"]]
    
    target_path = os.path.join(args.output_dir, args.run_name, args.target_file_name)
    print(f"Writing results to {target_path}.")
    with open(target_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    print("Getting arguments")
    args = get_inference_arguments()
    sampling_params = get_sampling_params()
    lora_request = get_lora_request()
    data = get_formatted_data(args, sampling_params, lora_request)
    engine = initialize_engine(args)
    process_requests(engine, data)

"""
sbatch \
    --gpus=rtx_3090:1 \
    --mem-per-cpu=16G \
    --wrap="python3 sft/inference_vllm.py \
        --data_path baseline/test.json \
        --run_name deepseek-7b-base-baseline";
"""