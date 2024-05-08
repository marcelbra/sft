import os
import json

from dataclasses import dataclass, field, asdict
from typing import Optional, List
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from arguments import get_inference_arguments, GenerationArguments

GENERATION_CONFIG_FILE_NAME = "generation_args.json"

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

def inference():
  
    print("Loading cli arguments")
    cli_args = get_inference_arguments()

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
    answers = [element["output"] for element in data]

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

    with tqdm(total=len(questions)) as pbar:
        for i, (question, answer) in enumerate(zip(questions, answers)):
            if cli_args.start_from and i < cli_args.start_from:
                continue
            inputs = tokenizer.encode(question, return_tensors="pt").to("cuda")
            outputs = model.generate(inputs)
            result = tokenizer.decode(outputs[0])
            results.append(
                {
                    "instruction": question,
                    "ground_truth": answer,
                    "prediction": result
                }
            )
            with open(target_path, "w") as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            pbar.update(1)

if __name__ == "__main__":
    inference()

"""
Base line rest (TODO)
sbatch \
    --gpus=rtx_4090:1 \
    --mem-per-cpu=8G \
    --time=0-12 \
    --wrap="python3 sft/inference_hf.py \
            --data_path baseline/test.json \
            --run_name deepseek-7b-base-baseline-2 \
            --target_file_name predictions_hf_rest.json \
            --start_from 1074";

M1 test
sbatch \
    --gpus=rtx_4090:1 \
    --time=0-12 \
    --mem-per-cpu=8G \
    --wrap="python3 sft/inference_hf.py \
            --data_path decomposed/test/with_result/1/data.json \
            --run_name deepseek-7b-base-m1";
58139668

M1 instruct
sbatch \
    --gpus=rtx_4090:1 \
    --time=0-12 \
    --mem-per-cpu=8G \
    --wrap="python3 sft/inference_hf.py \
            --data_path decomposed/test/with_result/1/data.json \
            --run_name deepseek-7b-base-m1-instruct \
            --formatting_template 'Generate the first step of the reasoning chain.\n### Instruction:\n{}\n### Response:\n'";
58139749
"""