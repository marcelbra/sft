import os
import json
import torch
import datasets

import pandas as pd

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Sequence, Optional
from argparse import ArgumentParser
from functools import partial

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

from prompting import build_source_prompt, build_target_prompt
from utils import OUTPUT_DIR, SEED, IGNORE_INDEX


def load_model_and_tokenizer_unsloth(model_name_or_path):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name_or_path,
        dtype = torch.bfloat16,
        load_in_4bit = False
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 32,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = 3407,
        use_rslora = False,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )

    return model, tokenizer

def load_model_and_tokenizer(model_name_or_path):
    kwargs = {}
    # if "gemma-1.1-7b-it" in model_name_or_path:
    #     kwargs = {
    #         "attn_implementation": "eager"
    #     }
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=False,
        **kwargs
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Settings
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    #setattr(model, 'model_parallel', True)
    #setattr(model, 'is_parallelizable', True)
    model.config.use_cache = False
    model.config.torch_dtype = torch.bfloat16 #(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    kwargs = {
        "use_fast": False
    }
    if "stablelm-3b-4e1t" in model_name_or_path:
        kwargs["use_fast"] = True
    if "llama" in model_name_or_path:
        kwargs["model_max_length"] = 4000
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        padding_side="right",
        trust_remote_code=True,
        local_files_only=False,
        **kwargs
    )

    added_pad_token = False
    if "Llama-3" in model_name_or_path:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
        model.resize_token_embeddings(len(tokenizer))
        added_pad_token = True
    elif "Mistral-7B" in model_name_or_path:
        tokenizer.add_special_tokens({'pad_token': '<pad>'})
        added_pad_token = True
    elif "stablelm-3b-4e1t" in model_name_or_path:
        tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
    if added_pad_token:
        print(f"Added pad token to tokenizer: {tokenizer.pad_token}")

    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    return model, tokenizer

def _tokenize_fn(strings: Sequence[str], tokenizer: AutoTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: AutoTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: AutoTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def train_tokenize_function(previous_step_n, first_step_data, type_, examples, tokenizer):
    
    sources = []
    targets = []
    replacements = []
    all_new_target_steps = []
    
    for question, steps in zip(examples['source_question'], examples['target_steps']):
        prompt, replaced, new_target_steps = build_source_prompt(
            question,
            steps,
            previous_step_n,
            first_step_data, 
            type_
        )
        sources.append(prompt)
        replacements.append(replaced)
        all_new_target_steps.append(new_target_steps)
    
    if new_target_steps:
        examples['target_steps'] = all_new_target_steps
    
    for steps, result, replace in zip(examples['target_steps'], examples['target_result'], replacements):
        prompt = build_target_prompt(steps, result, tokenizer.eos_token, replace, type_)
        targets.append(prompt)
    
    # Print formatted example for safety check
    source, target = list(zip(sources, targets))[1]
    print(f"Source:\n---\n{source}\n---\n")
    print(f"Target:\n---\n{target}\n---\n")
    
    return preprocess(sources, targets, tokenizer)

def load_data(
    data_path: str,
    tokenizer: AutoTokenizer,
    max_train_samples: Optional[int] = None,
    delete_longest_n: Optional[int] = None,
    previous_step_n: Optional[str] = None,
    first_step_path: Optional[str] = None,
    type_: Optional[str] = None
):

    dataset = load_dataset(
        'json',
        data_files=data_path,
        split="train",
    )
    if delete_longest_n:
        lengths = {
            i: len(element["target_steps"]) + len(element["source_question"])
            for i, element in enumerate(dataset)
        }
        delete = [
            index for index, _ in 
            sorted(lengths.items(), key=lambda x: x[1], reverse=True)[:delete_longest_n]
        ]
        dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=[
            element for i, element in enumerate(dataset) if i not in delete
        ]))

    if max_train_samples:
        dataset = dataset.select(range(max_train_samples))

    first_step_data = None
    if first_step_path:
        print(f"Loaded first step data from {first_step_path}.")
        with open(first_step_path, "r") as f:
            first_step_data = json.load(f)

    return dataset.map(
        partial(
            train_tokenize_function,
            previous_step_n,
            first_step_data,
            type_
        ),
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running Encoding",
        fn_kwargs={"tokenizer": tokenizer}
    )

def setup_trainer(model, tokenizer, train_dataset, data_collator, output_dir):

    logging = {
        "logging_strategy": "epoch",
        "save_strategy": "epoch"
    }
    # logging = {
    #     "logging_strategy": "steps",
    #     "save_strategy": "steps",
    #     "evaluation_strategy": "steps",
    #     "logging_steps": 50,
    #     "eval_steps": 50
    # }

    train_args = TrainingArguments(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 8,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        weight_decay = 0.01,
        lr_scheduler_type = "constant",
        seed = SEED,
        output_dir = output_dir,
        report_to = "wandb",
        save_total_limit = 1,
        **logging
    )

    return Trainer(
        model=model,
        tokenizer=tokenizer,
        args=train_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

def run(config):

    print("Config:")
    print(*list(config.items()), sep="\n", end="\n\n")
    output_path = os.path.join(OUTPUT_DIR, config["run_name"])
    print("Output Path: ", output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print("Loading model and tokenizer")
    if "unsloth" in config["model_name_or_path"]:
        model, tokenizer = load_model_and_tokenizer_unsloth(config["model_name_or_path"])
    else:
        model, tokenizer = load_model_and_tokenizer(config["model_name_or_path"])

    print("Loading data")
    train_dataset = load_data(
        config["data_path"],
        tokenizer,
        max_train_samples=None,
        delete_longest_n=None,
        previous_step_n=config["previous_step_n"],
        first_step_path=config["first_step_path"],
        type_=config["type"]
        )
    data_collator = DataCollatorForSupervisedDataset(tokenizer)
    
    print("Setting up training")
    trainer = setup_trainer(model, tokenizer, train_dataset, data_collator, output_path)
    
    print("Start training")
    trainer.train()

if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--previous_step_n", type=int, default=None)
    parser.add_argument("--first_step_path", type=str, default=None)
    parser.add_argument("--type", type=str, default="training", choices=["training", "inference", "training_tf", "new_dawn"])
    args = parser.parse_args()
    
    config = {
        "data_path": args.data_path,
        "run_name": args.run_name,
        "model_name_or_path": args.model_name_or_path,
        "previous_step_n": args.previous_step_n,
        "first_step_path": args.first_step_path,
        "type": args.type
    }
    run(config)
