"""
MWE script for loading a model into the cache for the first time.
Call: sbatch --gpus=rtx_4090:1 --wrap="python3 /cluster/home/mbraasch/repos/sft/scripts/load_model.py --model_name deepseek-ai/deepseek-llm-7b-chat"
"""
import os
import torch

from argparse import ArgumentParser, Namespace

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def get_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--cache_dir", default=os.environ["HF_HOME"], type=str)
    return parser.parse_args()

args = get_arguments()

print("Start: loading model and tokenizer")
print(f"Using cache directory: {os.environ["HF_HOME"]}")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, cache_dir=args.cache_dir,  device_map="auto")
model.generation_config = GenerationConfig.from_pretrained(args.model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)
print("Done: loading model and tokenizer")