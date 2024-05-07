import os
import json

from typing import List

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from argparse import ArgumentParser, Namespace
from tqdm import tqdm


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
        default="predictions_hf.json"
    )
    parser.add_argument(
        "--formatting_template",
        type=str,
        default="### Instruction:\n{}\n### Response:\n"
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


def format_questions(questions: List[str], formatting_template: str):
    return [
        formatting_template.format(question.strip()).lstrip()
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


args = get_arguments()

adapter_path = os.path.join(args.output_dir, args.run_name)
print(f"Adapter search path: {adapter_path}")
adapter_model_name = get_max_step_adapter_name(adapter_path)
print(f"Loading adapters from: {adapter_model_name}")

print(f"Start: Loading model {args.model_name_or_path}")
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    cache_dir=os.environ["HF_HOME"], 
    device_map="auto"
)
print("Done: Loading model")

print("Start: Load adapter")
model = PeftModel.from_pretrained(model, adapter_model_name)
print("Done: Load adapter")

model.generation_config = GenerationConfig.from_pretrained(args.model_name_or_path)
model.generation_config.do_sample = False
model.generation_config.num_beams = 1
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.max_new_tokens = 1024
print(f"Generation config:\n{model.generation_config}")

print("Start: Load tokenizer")
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
print("Done: Load tokenizer")

data_path = os.path.join(args.data_dir, args.data_path)
print(f"Load data from {data_path}")
with open(data_path) as f:
    data = json.load(f)
questions = format_questions(
    questions=[element["input"] for element in data],
    formatting_template=args.formatting_template
)

print("Start eval")
target_path = os.path.join(args.output_dir, args.run_name, args.target_file_name)
print(f"Writing to target path {target_path}")
results = []
for i, data_point in enumerate(tqdm(questions)):
    inputs = tokenizer.encode(data_point, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)
    result = tokenizer.decode(outputs[0])
    results.append(
        {
            "input": data_point,
            "prediction": result
        }
    )
    with open(target_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

"""
sbatch \
    --gpus=rtx_4090:1 \
    --mem-per-cpu=8G \
    --wrap=" \
        cd repos/sft; \
            python3 inference_hf.py \
            --data_path decomposed/test/with_result/1/data.json \
            --run_name deepseek-7b-base-baseline";
sbatch \
    --gpus=rtx_3090:1 \
    --mem-per-cpu=8G \
    --wrap=" \
        cd repos/sft; \
            python3 inference_hf.py \
            --data_path decomposed/test/with_result/1/data.json \
            --run_name deepseek-7b-base-baseline";
"""