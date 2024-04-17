import os
import json

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from src.arguments import (
    ModelArguments,
    DataArguments,
    SFTTrainingArguments,
    H4ArgumentParser
)
  
from argparse import ArgumentParser, Namespace
from tqdm import tqdm

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


def _get_args_from_cl(parser: ArgumentParser) -> ArgumentParser:
    """
    Adds all the arguments to the parser.

    :param parser: The parser to add the arguments to.
    :return: The parser with the added arguments.
    """
    parser.add_argument(
        "--config",
        type=str
    )
    parser.add_argument(
        "--final_result",
        type=str
    )
    return parser


def get_number(str):
    return int(str.split("#### ")[1].split("\n")[0])

def format_question(question: str):
    return f"### Question: {question}\n ### Answer: "

def get_max_step_adapter_name(training_args, output_dir):
    ckpt_name = 'checkpoint-'
    max_step = max([
        int(file_name.split(ckpt_name)[1])
        for file_name in os.listdir(output_dir)
        if file_name.startswith(ckpt_name)
    ])
    return os.path.join(output_dir, f"{ckpt_name}{max_step}", "adapter_model")


args = get_arguments()
parser = H4ArgumentParser((
    ModelArguments, DataArguments, SFTTrainingArguments
))
model_args, data_args, training_args = parser.parse_yaml_and_args(
    yaml_arg=args.config, other_args=[], allow_extra_keys=True
)

output_dir = os.path.join(training_args.output_dir, training_args.run_name)
print(f"Output dir: {output_dir}")
adapter_model_name = get_max_step_adapter_name(training_args, output_dir)
print(f"Loading adapters from: {adapter_model_name}")

print(f"Start: Loading model {model_args.model_name_or_path}")
model = AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    cache_dir=os.environ["HF_HOME"], 
    device_map="auto"
)
print("Done: Loading model")

print("Start: Load adapter")
model = PeftModel.from_pretrained(model, adapter_model_name)
print("Done: Load adapter")

model.generation_config = GenerationConfig.from_pretrained(model_args.model_name_or_path)
model.generation_config.do_sample = False
model.generation_config.num_beams = 1
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.pad_token_id = model.generation_config.eos_token_id
model.generation_config.max_new_tokens = 1024
print(f"Generation config:\n{model.generation_config}")

print("Start: Load tokenizer")
tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
print("Done: Load tokenizer")

data_path = os.path.join(data_args.data_dir, "baseline/test.json")
print(f"Load data from {data_path}")
with open(data_path) as f:
    data = json.load(f)

print("Start eval")
results = []
for data_point in tqdm(data):
    formatted_question = format_question(data_point["input"])
    inputs = tokenizer.encode(formatted_question, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)
    result = tokenizer.decode(outputs[0])
    results.append(
        {
            "question": data_point["output"],
            "ground_truth": data_point["output"],
            "raw_prediction": result
        }
    )
    print("Writing result")
    with open(args.final_result, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

"""
sbatch \
    --gpus=rtx_4090:1 \
    --mem-per-cpu=8G \
    --time=0-10 \
    --wrap=" \
        cd repos/sft; \
            python3 inference.py \
                --config /cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output/deepseek-llm-7b-base-baseline-without-packing/config.yaml \
                --final_result /cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output/deepseek-llm-7b-base-baseline-without-packing/raw_test_results.json"
"""