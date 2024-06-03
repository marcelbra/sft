import os
import json
import sys

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field, asdict
from typing import Optional

import argparse
import transformers

from settings import OUTPUT_DIR

def get_inference_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, help="Adds a default data directory to the front, only the json specification is needed.", default=None)
    parser.add_argument("--instruction", type=str, default="Solve the given math problem step by step and put your final answer after 'Final answer: '.")
    # parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/deepseek-llm-7b-base")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-2b-it")
    parser.add_argument("--output_dir", type=str, help="Specifies the path to the directory where everything is happening.", default="/cluster/work/lawecon/Work/mbraasch/output/")
    parser.add_argument("--data_dir", type=str, help="Adds a default data directory to the front, only the json specification is needed.",default="/cluster/work/lawecon/Work/mbraasch/data")
    parser.add_argument("--target_file_name", type=str, default="next_step_predictions.json")
    parser.add_argument("--start_from", type=int, default=None)
    parser.add_argument("--amount_samples", type=int, default=None)
    return parser.parse_args([])

def get_arguments() -> Namespace:
    parser = ArgumentParser()
    # parser.add_argument("--model_name_or_path", type=str, default="deepseek-ai/deepseek-llm-7b-base")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-2b-it")
    parser.add_argument("--run_name", type=str)
    parser.add_argument("--data_path", type=str, help="")
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--max_train_samples", type=int, default=None, help="For debugging. Cuts the amount of training samples.")
    cli_args = parser.parse_args()
    # print(f"Training model type {cli_args.model_type} (bl=baseline, m1=m1, mi=all but m1).")
    # TODO: refactor
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    training_args.run_name = cli_args.run_name
    training_args.output_dir = os.path.join(OUTPUT_DIR, training_args.run_name)
    delattr(cli_args, 'run_name')
    # print(f"Run name: {training_args.run_name}")
    # print(f"Formatting template:\n{cli_args.formatting_template}")
    # print(f"Data path: {cli_args.data_path}")
    # trainings_args_path = os.path.join(OUTPUT_DIR, training_args.run_name, TRAINING_ARGS_FILE_NAME)
    all_args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(cli_args)
    )
    # with open(trainings_args_path, "w") as f:
    #     json.dump(str(all_args), f, indent=4, ensure_ascii=False)
    return all_args, training_args


@dataclass
class ModelArguments:
    padding_side: Optional[str] = field(default='right')
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."})
    local_files_only: Optional[bool] = field(default=True, metadata={"help": ""})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})


@dataclass
class DataArguments:
    source_max_len: int = field(default=3000, metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."})
    target_max_len: int = field(default=1000, metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},)
    dataset: str = field(default='', metadata={"help": "Which dataset to finetune on."})
    packing: bool = field(default=False, metadata={"help": "Apply packing when fine-tuning or not"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    max_memory_MB: int = field(default=80000, metadata={"help": "Free memory per gpu."})
    bf16: bool = field(default=True)
    output_dir: str = field(default=OUTPUT_DIR, metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=4, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
    num_train_epochs: int = field(default=3, metadata={"help":""})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=1e-5, metadata={"help": 'The learnign rate'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='linear', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    max_seq_length: int = field(default=4096, metadata={"help": 'maximum sequence length for SFTTrainer'})


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
