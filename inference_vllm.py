import os
import json

from argparse import ArgumentParser, Namespace
from typing import List
from vllm import SamplingParams, LLMEngine, EngineArgs, LLM
from vllm.lora.request import LoRARequest


def _get_args_from_cl(parser: ArgumentParser) -> ArgumentParser:
    """
    Adds all the arguments to the parser.

    :param parser: The parser to add the arguments to.
    :return: The parser with the added arguments.
    """
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-base"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Assumes the adapter is in here, too.",
        default="/cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output"
    ),
    parser.add_argument(
        "--run_name",
        type=str
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Adds a default data directory to the front, only the json specification is needed.",
        default="/cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/data"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Adds a default data directory to the front, only the json specification is needed."
    )
    parser.add_argument(
        "--target_file_name",
        type=str,
        default="predictions_vllm.json"
    )
    return parser

def get_arguments() -> Namespace:
    """
    Gets the arguments from the command line or accepts a pre-defined list
    of arguments such that it can be used programatically.

    :param predefined_args: The pre-defined arguments.
    :return: The arguments.
    """
    parser = ArgumentParser()
    _get_args_from_cl(parser)
    return parser.parse_args()


def format_questions(questions: List[str]):
    return [
        '''
### Instruction:
{}
### Response:
        '''.format(question.strip()).lstrip()
        for question in questions
    ]

def get_max_step_adapter_name(output_dir):
    ckpt_name = 'checkpoint-'
    max_step = max([
        int(file_name.split(ckpt_name)[1])
        for file_name in os.listdir(output_dir)
        if file_name.startswith(ckpt_name)
    ])
    return os.path.join(output_dir, f"{ckpt_name}{max_step}", "adapter_model")

print("Getting arguments")
args = get_arguments()

print("Initalizating sampling params")
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=2048,
    stop=["\n<|EOT|>"]
)

data_path = os.path.join(args.data_dir, args.data_path)
print(f"Load data from {data_path}")
with open(data_path) as f:
    data = json.load(f)
questions = format_questions([element["input"] for element in data])
inputs = [(id_, text, sampling_params) for id_, text in enumerate(questions)][:10]

adapter_path = os.path.join(args.output_dir, args.run_name)
print(f"Getting max adapter checkpoint from {adapter_path}")
adapter_ckpt = get_max_step_adapter_name(adapter_path)
print(f"Loaded ckpt {adapter_ckpt}.")


print(f"Load model {args.model_name_or_path}")
engine_args = EngineArgs(
        model=args.model_name_or_path,
        enable_lora=True,
        max_lora_rank=64,
        max_num_seqs=256,
        # max_loras=1,
)
engine = LLMEngine.from_engine_args(engine_args)


print("Start processing inputs")
result = []
while True:
    if inputs:
         req_id, prompt, sampling_params = inputs.pop(0)
         engine.add_request(str(req_id), prompt, sampling_params)

    # continue the request processing
    outputs = engine.step()
    for output in outputs:
        if output.finished:
            result.append({
                "prompt": output.prompt,
                "generated_text": output.outputs[0].text
            })

    if not (engine.has_unfinished_requests() or inputs):
        break

print("Writing result")
target_path = os.path.join(args.output_dir, args.run_name, args.target_file_name)
with open(target_path, "w") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)

"""
sbatch \
    --gpus=rtx_3090:1 \
    --mem-per-cpu=16G \
    --wrap=" \
        cd repos/sft; \
        python3 inference_fast.py \
        --data_path decomposed/test/with_result/1/data.json \
        --run_name deepseek-7b-base-baseline"; \
"""
"""
sbatch \
    --gpus=rtx_4090:1 \
    --mem-per-cpu=16G \
    --wrap=" \
        cd repos/sft; \
        python3 inference_fast.py \
        --data_path decomposed/test/with_result/1/data.json \
        --run_name deepseek-7b-base-baseline";
"""