import json
import os
import logging


from functools import partial
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
# import bitsandbytes as bnb

from accelerate import PartialState
from datasets import load_dataset
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    BitsAndBytesConfig,
    LlamaTokenizer,
    Trainer
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)

from prompting import build_source_prompt, build_target_prompt
from arguments import get_arguments
from settings import TRAIN_METRICS_FILE_NAME

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items():
        total += v
    for k, v in dtypes.items():
        print(k, v, v / total)


def find_all_linear_names(args, model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(args.run_name, state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.run_name, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")
        kwargs["model"].save_pretrained(peft_model_path)
        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, 'a'):
                os.utime(fname, times)

        touch(os.path.join(args.run_name, 'completed'))
        self.save_model(args, state, kwargs)


def get_accelerate_model(args):
    n_gpus = 0
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if n_gpus > 1:
        device_string = PartialState().process_index
        device_map = {'': device_string}

    print(f'Loading base model {args.model_name_or_path}...')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        device_map="auto",
        # quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=args.bits == 4,
        #     load_in_8bit=args.bits == 8,
        #     llm_int8_threshold=6.0,
        #     llm_int8_has_fp16_weight=False,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_use_double_quant=args.double_quant,
        #     bnb_4bit_quant_type=args.quant_type,
        # ),
        torch_dtype=torch.bfloat16
        # local_files_only=args.local_files_only,
        # trust_remote_code=args.trust_remote_code,
    )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)
    model.config.use_cache = False
    model.config.torch_dtype = (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        padding_side=args.padding_side,
        use_fast=False,
        # trust_remote_code=args.trust_remote_code,
        # local_files_only=args.local_files_only,
    )
    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    print(f'Initalizing LoRA.')
    modules = find_all_linear_names(args, model)
    print(f'The {len(modules)} modules are: {modules}')
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules = modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    
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
        build_source_prompt(question, steps, instruction)
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
    source, target = list(zip(sources, targets))[1]
    print(f"Source:\n---\n{source}\n---\n")
    print(f"Target:\n---\n{target}\n---\n")
    
    return preprocess(sources, targets, tokenizer)


def train(args, training_args):

    set_seed(0)

    # Model

    model, tokenizer = get_accelerate_model(args)

    # Data

    dataset_path = os.path.join(args.data_path)
    dataset = load_dataset(
        'json',
        data_files=dataset_path,
        split="train",
    )

    if args.max_train_samples:
        dataset = dataset.select(range(args.max_train_samples))

    train_dataset = dataset.map(
        partial(train_tokenize_function, args.instruction),
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=dataset.column_names,
        load_from_cache_file=False,
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # Training

    print_trainable_parameters(args, model)

    if args.local_rank > 0: 
        torch.distributed.barrier()
    
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )
    
    logger.info("*** Train ***")
    trainer.add_callback(SavePeftModelCallback)
    all_metrics = {"run_name": args.run_name}
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    all_metrics.update(metrics)
    
    with open(os.path.join(args.run_name, TRAIN_METRICS_FILE_NAME), "w") as f:
        f.write(json.dumps(all_metrics))

def main():
    args, training_args = get_arguments()
    train(args=args, training_args=training_args)

if __name__ == "__main__":
    main()