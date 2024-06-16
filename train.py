import os
import json
import torch

from copy import deepcopy
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
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=False
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
        trust_remote_code=True,
        local_files_only=False
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


def train_tokenize_function(examples, tokenizer):
    sources = [
        build_source_prompt(question, steps)
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

def load_data(data_path, tokenizer, max_train_samples=None):

    dataset = load_dataset(
        'json',
        data_files=data_path,
        split="train",
    )

    if max_train_samples:
        dataset = dataset.select(range(max_train_samples))

    return dataset.map(
        train_tokenize_function,
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
    train_dataset = load_data(config["data_path"], tokenizer, max_train_samples=None)
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
    args = parser.parse_args()
    
    config = {
        "data_path": args.data_path,
        "run_name": args.run_name,
        "model_name_or_path": args.model_name_or_path,
    }
    run(config)
