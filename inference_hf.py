import os
import json

from dataclasses import dataclass, field, asdict
from typing import Optional, List
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


GENERATION_CONFIG_FILE_NAME = "generation_args.json"

@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field(default=1024)
    min_new_tokens: Optional[int] = field(default=None)
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)
    temperature: Optional[float] = field(default=None)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

    def as_dict(self) -> dict:  
        """Converts the dataclass instance to a dictionary."""  
        return asdict(self)  

def get_arguments() -> Namespace:
    """
    Gets the arguments from the command line or accepts a pre-defined list
    of arguments such that it can be used programatically.

    :param predefined_args: The pre-defined arguments.
    :return: The arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--data_path", type=str, help="Adds a default data directory to the front, only the json specification is needed.")
    parser.add_argument("--formatting_template", type=str, default="### Instruction:\n{}\n### Response:\n")
    parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/deepseek-llm-7b-base")
    parser.add_argument("--output_dir", type=str, help="Specifies the path to the directory where everything is happenung..", default="/cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output/")
    parser.add_argument("--data_dir", type=str, help="Adds a default data directory to the front, only the json specification is needed.",default="/cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/data")
    parser.add_argument("--target_file_name", type=str, default="predictions_hf.json")
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

  
print("Loading cli arguments")
cli_args = get_arguments()


print("Loading generation arguments")
generation_config = GenerationConfig(**vars(GenerationArguments()))

target_path = os.path.join(cli_args.output_dir, cli_args.run_name, GENERATION_CONFIG_FILE_NAME)
print("Saving generation arguments")
with open(target_path, "w") as f:
    json.dump(vars(generation_config), f, indent=4, ensure_ascii=False)

data_path = os.path.join(cli_args.data_dir, cli_args.data_path)
print(f"Load data from {data_path}")
with open(data_path) as f:
    data = json.load(f)
questions = format_questions(
    questions=[element["instruction"] for element in data],
    formatting_template=cli_args.formatting_template
)

print("Start: Load tokenizer")
tokenizer = AutoTokenizer.from_pretrained(cli_args.model_name_or_path)

adapter_path = os.path.join(cli_args.output_dir, cli_args.run_name)
print(f"Adapter search path: {adapter_path}")
adapter_model_name = get_max_step_adapter_name(adapter_path)
print(f"Loading adapters from: {adapter_model_name}")

print(f"Start: Loading model {cli_args.model_name_or_path}")
model = AutoModelForCausalLM.from_pretrained(
    cli_args.model_name_or_path,
    torch_dtype=torch.bfloat16,
    cache_dir=os.environ["HF_HOME"], 
    device_map="auto",
)
model.generation_config = generation_config

print("Start: Load adapter")
model = PeftModel.from_pretrained(model, adapter_model_name)

print("Start eval")
target_path = os.path.join(cli_args.output_dir, cli_args.run_name, cli_args.target_file_name)
print(f"Writing to target path {target_path}")
results = []
for i, data_point in enumerate(tqdm(questions)):
    inputs = tokenizer.encode(data_point, return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)
    result = tokenizer.decode(outputs[0])
    results.append(
        {
            "instruction": data_point,
            "prediction": result
        }
    )
    with open(target_path, "w") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

"""
sbatch \
    --gpus=rtx_4090:1 \
    --mem-per-cpu=8G \
    --time=0-12 \
    --wrap="python3 sft/inference_hf.py \
            --data_path baseline/test.json \
            --run_name deepseek-7b-base-baseline-2";
"""