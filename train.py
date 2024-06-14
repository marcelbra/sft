import os
import json
import torch

from copy import deepcopy
from functools import partial
from dataclasses import dataclass
from typing import Dict, Sequence
from argparse import ArgumentParser

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)

from prompting import build_source_prompt, build_target_prompt
from sft.utils import OUTPUT_DIR, SEED, IGNORE_INDEX


def load_model_and_tokenizer(model_name_or_path):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        device_map="auto",
        torch_dtype=torch.bfloat16
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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        padding_side="right",
        use_fast=False,
        # trust_remote_code=args.trust_remote_code,
        # local_files_only=args.local_files_only,
    )
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


def train_tokenize_function(instruction, examples, tokenizer):
    sources = [
        build_source_prompt(question, steps, instruction, eos)
        for question, steps in zip(
            examples['source_question'],
            examples['source_steps']
        )
    ]
    targets = [
        build_target_prompt(steps, result, tokenizer.eos_token)
        for steps, result in zip(
            examples['target_steps'],
            examples['target_result']
        )
    ]
    
    # Print formatted example for safety check
    #source, target = list(zip(sources, targets))[1]
    #print(f"Source:\n---\n{source}\n---\n")
    #print(f"Target:\n---\n{target}\n---\n")
    
    return preprocess(sources, targets, tokenizer)

def load_data(config, max_train_samples=None):

    dataset = load_dataset(
        'json',
        data_files=config["data_path"],
        split="train",
    )

    if max_train_samples:
        dataset = dataset.select(range(max_train_samples))

    return dataset.map(
        partial(train_tokenize_function, config["instruction"]),
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
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
        num_train_epochs = 2,
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



if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--run_name", type=str, required=True)
    args = parser.parse_args()

    config = {
        "data_path": args.data_path,
        "run_name": args.run_name,
        "instruction": "Solve the following math word problem step-by-step.",
        "experiment_name": "gemma-2b-it",
        "model_name_or_path": "google/gemma-2b-it",
    }
    print("Config:")
    print(*list(config.items()), sep="\n", end="\n\n")
    output_path = os.path.join(OUTPUT_DIR, config["experiment_name"], config["run_name"])
    print("Output Path: ", output_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with open(os.path.join(output_path, "config.json"), "w") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    print("Loading model and tokenizer")
    model, tokenizer = load_model_and_tokenizer(config["model_name_or_path"])

    print("Loading data")
    train_dataset = load_data(config, max_train_samples=None)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    
    print("Setting up training")
    trainer = setup_trainer(model, tokenizer, train_dataset, data_collator, output_path)
    
    print("Start training")
    trainer.train()
