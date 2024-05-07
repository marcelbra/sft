import json
import os
import logging
import argparse

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence

import torch
import transformers
import bitsandbytes as bnb

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

from settings import OUTPUT_DIR, DATA_DIR

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
EOT_TOKEN = "<|EOT|>"

# Arguments


def get_arguments() -> Namespace:
    """
    Gets the arguments from the command line or accepts a pre-defined list
    of arguments such that it can be used programatically.

    :param predefined_args: The pre-defined arguments.
    :return: The arguments.
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="deepseek-ai/deepseek-llm-7b-base"
    )
    parser.add_argument(
        "--run_name",
        type=str
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Adds a default data directory to the front, only the specification starting _from_ data directory is needed."
    )
    parser.add_argument(
        "--formatting_template",
        type=str,
        default="### Instruction:\n{}\n### Response:\n"
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="For debugging. Cuts the amount of training samples."
    )
    return parser.parse_args()


@dataclass
class ModelArguments:
    padding_side: Optional[str] = field(default='right')
    trust_remote_code: Optional[bool] = field(default=True, metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."})
    continue_on: str = field(default='Empty')


@dataclass
class DataArguments:
    source_max_len: int = field(default=3000, metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."})
    target_max_len: int = field(default=1000, metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},)
    dataset: str = field(default='', metadata={"help": "Which dataset to finetune on."})
    packing: bool = field(default=False, metadata={"help": "Apply packing when fine-tuning or not"})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    train_on_source: Optional[bool] = field(default=False, metadata={"help": "Whether to train on the input in addition to the target text."})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    quant_type: str = field(default="nf4", metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."})
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    max_memory_MB: int = field(default=80000, metadata={"help": "Free memory per gpu."})
    bf16: bool = field(default=True)
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=4, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=8, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
    num_train_epochs: int = field(default=3, metadata={"help":""})
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=True, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    max_seq_length: int = field(default=4096, metadata={"help": 'maximum sequence length for SFTTrainer'})


@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field(default=512)
    min_new_tokens: Optional[int] = field(default=None)
    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)
    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)

# # #

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


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def find_all_linear_names(args, model):
    cls = bnb.nn.Linear4bit if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(OUTPUT_DIR, args.run_name, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

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

        touch(os.path.join(OUTPUT_DIR, args.run_name, 'completed'))
        self.save_model(args, state, kwargs)


def get_accelerate_model(args):#, checkpoint_dir):
    n_gpus = 0
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()

    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if n_gpus > 1:
        device_string = PartialState().process_index
        device_map = {'': device_string}

    print(f'loading base model {args.model_name_or_path}...')
    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        device_map=device_map,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)),
        local_files_only=True,
        trust_remote_code=True,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print('=' * 80)
            print('Your GPU supports bfloat16, you can accelerate training with the argument --bf16')
            print('=' * 80)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        padding_side="right",
        use_fast=False,  # Fast tokenizer giving issues.
        # tokenizer_type='llama' if 'llama' in args.model_name_or_path else None,  # Needed for HF name change
        trust_remote_code=True,
        local_files_only=True,
    )
    print("PAD Token:", tokenizer.pad_token, tokenizer.pad_token_id)
    print("BOS Token", tokenizer.bos_token, tokenizer.bos_token_id)
    print("EOS Token", tokenizer.eos_token, tokenizer.eos_token_id)

    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    print("model_config", model.config)
    if 'llama' in args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
        # LLaMA tokenizer may not have correct special tokens set.
        # Check and add them if missing to prevent them from being parsed into different tokens.
        # Note that these are present in the vocabulary.
        # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
        print('Adding special tokens.')
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
            "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
            "unk_token": tokenizer.convert_ids_to_tokens(
                model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
            ),
        })

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    print(f'adding LoRA modules...')
    modules = find_all_linear_names(args, model)
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=modules,
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

def build_instruction_prompt(instruction: str):
    return '''
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
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
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    # print("XXX Preprocess:")
    # for i in range(3):
    #     print(i)
    #     print("input_ids", tokenizer.convert_ids_to_tokens(input_ids[i]))
    #     print("labels", tokenizer.convert_ids_to_tokens(labels[i]))
    return dict(input_ids=input_ids, labels=labels)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def train_tokenize_function(examples, tokenizer):
    sources = [
        build_instruction_prompt(instruction)
        for instruction in examples['instruction']
    ]
    targets = [f"{output}\n{EOT_TOKEN}" for output in examples['output']]
    data_dict = preprocess(sources, targets, tokenizer)
    return data_dict

# def get_last_checkpoint(checkpoint_dir):
#     if isdir(checkpoint_dir):
#         is_completed = exists(join(checkpoint_dir, 'completed'))
#         if is_completed: return None, True  # already finished
#         max_step = 0
#         for filename in os.listdir(checkpoint_dir):
#             if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
#                 max_step = max(max_step, int(filename.replace('checkpoint-', '')))
#         if max_step == 0: return None, is_completed  # training started, but no checkpoint
#         checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
#         print(f"Found a previous checkpoint at: {checkpoint_dir}")
#         return checkpoint_dir, is_completed  # checkpoint found!
#     return None, False  # first training


def train(cli_args: Namespace):
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    training_args.run_name = cli_args.run_name
    delattr(cli_args, 'run_name')
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args), **vars(cli_args)
    )
    print(args)

    # checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    # if completed_training:
    #     print('Detected that training was already completed!')

    # if 'checkpoint-' in args.continue_on:
    #     checkpoint_dir = args.continue_on
    # print(checkpoint_dir)

    model, tokenizer = get_accelerate_model(args)#, checkpoint_dir)

    model.config.use_cache = False
    print('loaded model')
    set_seed(0)

    dataset_path = os.path.join(DATA_DIR, args.data_path)
    dataset = load_dataset(
        'json',
        data_files=dataset_path,
        split="train",
    )
    if args.max_train_samples:
        dataset = dataset.select(range(args.max_train_samples))

    if training_args.local_rank > 0: 
        torch.distributed.barrier()
        
    train_dataset = dataset.map(
        train_tokenize_function,
        batched=True,
        batch_size=3000,
        num_proc=32,
        remove_columns=dataset.column_names,
        load_from_cache_file=True,
        desc="Running Encoding",
        fn_kwargs={ "tokenizer": tokenizer }
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator
    )

    # Callbacks
    trainer.add_callback(SavePeftModelCallback)

    # Verifying the datatypes and parameter counts before training.
    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    with open(os.path.join(OUTPUT_DIR, args.run_name, "metrics.json"), "w") as f:
        f.write(json.dumps(all_metrics))


if __name__ == "__main__":
    cli_args = get_arguments()
    print(f"CLI arguments were: {str(cli_args)}.")
    train(cli_args=cli_args)

"""
--time=0-8 \
sbatch \
    --gpus=rtx_3090:1 \
    --mem-per-cpu=8G \
    --wrap="cd repos/sft; \
        python3 run_sft.py \
            --run_name deepseek-7b-base-baseline-2 \
            --data_path baseline/train.json \
            --max_train_samples 20"; \
sbatch \
    --gpus=rtx_4090:1 \
    --mem-per-cpu=8G \
    --wrap="cd repos/sft; \
        python3 run_sft.py \
            --run_name deepseek-7b-base-baseline-2 \
            --data_path baseline/train.json \
            --max_train_samples 20";
"""