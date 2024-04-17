import sys
import os
import logging
import copy

import torch
import bitsandbytes as bnb
import transformers

from os.path import exists, join, isdir
from typing import Tuple, List, Optional, Dict, Sequence
from dataclasses import dataclass

from trl import DataCollatorForCompletionOnlyLM
from accelerate import PartialState
from datasets import Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence

from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaTokenizer
)
from peft import (
    prepare_model_for_kbit_training,
    get_peft_model,
    LoraConfig,
    PeftModel
)

from src.arguments import (
    ModelArguments,
    SFTTrainingArguments,
    DataArguments,
    TrainingArguments
)


DEFAULT_PAD_TOKEN = "[PAD]"
IGNORE_INDEX = -100

class SavePeftModelCallback(transformers.TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            checkpoint_folder = os.path.join(args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

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

        touch(join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True,
                              padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


def set_dtypes(model: AutoModelForCausalLM, model_args: ModelArguments) -> None:
    bf16 = getattr(torch, model_args.torch_dtype) == getattr(torch, "bfloat16")
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if bf16:
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)


def verify_datatypes_parameter_counts(model):
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        print(k, v, v / total)


def find_all_linear_names(bits, model):
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


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


def get_accelerate_model(
    training_args: TrainingArguments,
    model_args: ModelArguments
):
    print("Setting devices and other settings for training")

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    max_memory = f'{training_args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"
    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    print(f'Using model cache directory {os.environ["HF_HOME"]}.')
    
    # TODO: maybe this is the reason for masking error:
    compute_dtype = getattr(torch, model_args.torch_dtype)

    print(f'Using quantization config.')

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=model_args.bits == 4,
        load_in_8bit=model_args.bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type
    )

    print(f'Loading base model {model_args.model_name_or_path}.')

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        device_map=device_map,
        max_memory=max_memory,
        quantization_config=quantization_config,
        torch_dtype = compute_dtype,
        trust_remote_code = model_args.trust_remote_code,
        local_files_only=True,
    )
    model.config.torch_dtype = compute_dtype
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    print(f'Prepare model for kbit training.')

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    print(f'Adding LoRA modules.')

    modules = find_all_linear_names(model_args.bits, model)
    config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        target_modules=modules,
        lora_dropout=model_args.lora_dropout,
        bias=model_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    set_dtypes(model, model_args)

    print(f'Loading tokenizer.')

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        padding_side=model_args.padding_side,
        use_fast=False,  # Fast tokenizer giving issues.
        trust_remote_code=model_args.trust_remote_code,
        local_files_only=True,
    )
    if tokenizer._pad_token is None:
        smart_tokenizer_and_embedding_resize(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )

    return model, tokenizer

def load_data(args):
    full_dataset_path = os.path.join(args.data_dir, args.dataset_name)
    dataset = Dataset.from_json(path_or_paths=full_dataset_path)
    if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    return dataset.remove_columns(
        [col for col in dataset.column_names if col not in ['input', 'output']]
    )

# def get_model_and_tokenizer(
#         model_args: ModelArguments,
#         training_args: SFTTrainingArguments,
#         checkpoint_dir: str
#     ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

#     print("Load GPU settings.")
#     n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
#     device_map = "auto"
#     # For distributed training, set device map and max memory per device
#     if n_gpus > 1:
#         device_string = PartialState().process_index
#         device_map = {'': device_string}

#     print("Set torch bit settings")
#     if model_args.torch_dtype not in {"auto", None}:
#         compute_dtype = getattr(torch, model_args.torch_dtype)
#     if compute_dtype == torch.float16 and model_args.bits == 4:
#         if torch.cuda.is_bf16_supported() and model_args.torch_dtype != 'bfloat16':
#             print('=' * 80)
#             print('Your GPU supports bfloat16, you can accelerate training by setting torch_dtype to "bfloat16"')
#             print('=' * 80)

#     print("Load quantization config.")
#     # quantization_config = BitsAndBytesConfig(
#     #     load_in_4bit=model_args.bits == 4,
#     #     load_in_8bit=model_args.bits == 8,
#     #     llm_int8_threshold=6.0,
#     #     llm_int8_has_fp16_weight=False,
#     #     bnb_4bit_compute_dtype=compute_dtype,
#     #     bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
#     #     bnb_4bit_quant_type=model_args.bnb_4bit_quant_type
#     # )
#     quantization_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         llm_int8_threshold=6.0,
#         llm_int8_has_fp16_weight=False,
#         bnb_4bit_compute_dtype=getattr(torch, "bfloat16"),
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4"
#     )

#     # print(f'Load model {model_args.model_name_or_path}.')
#     # print(f'Using cache {os.environ["HF_HOME"]}')
#     # model = AutoModelForCausalLM.from_pretrained(
#     #     model_args.model_name_or_path,
#     #     cache_dir=os.environ["HF_HOME"],
#     #     device_map=device_map,
#     #     quantization_config=quantization_config,
#     #     torch_dtype=compute_dtype,
#     #     local_files_only=False
#     # )
#     # model.config.use_cache = False
#     model = AutoModelForCausalLM.from_pretrained(
#         "deepseek-ai/deepseek-llm-7b-base",
#         quantization_config=quantization_config,
#         device_map="auto",
#         cache_dir=os.environ["HF_HOME"],
#     )

#     print("Load model for kbit training.")
#     model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

#     # print(f'Load LoRA model.')
#     # if checkpoint_dir is None:
#     #     modules = _find_all_linear_names(model_args, model)
#     #     config = LoraConfig(
#     #         r=model_args.lora_r,
#     #         lora_alpha=model_args.lora_alpha,
#     #         target_modules=modules,
#     #         lora_dropout=model_args.lora_dropout,
#     #         bias=model_args.lora_bias,
#     #         task_type="CAUSAL_LM",
#     #     )
#     #     model = get_peft_model(model, config)
#     #     _set_datatypes(model_args, model)
#     # else:
#     #     print("Load adapters from checkpoint.")
#     #     model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True, local_files_only=True,)

#     # setattr(model, 'model_parallel', True)
#     # setattr(model, 'is_parallelizable', True)

#     peft_config = LoraConfig(
#             r=model_args.lora_r,
#             lora_alpha=model_args.lora_alpha,
#             lora_dropout=model_args.lora_dropout,
#             bias=model_args.lora_bias,
#             task_type="CAUSAL_LM",
#     )


#     print("Load tokenizer")
#     tokenizer = AutoTokenizer.from_pretrained(
#         model_args.model_name_or_path,
#         cache_dir=os.environ["HF_HOME"],
#         padding_side=model_args.padding_side,
#         use_fast=False,  # Fast tokenizer giving issues
#         trust_remote_code=model_args.trust_remote_code,
#         local_files_only=False,
#     )
#     if tokenizer._pad_token is None:
#         model, tokenizer = _resize_tokenizer_and_model(
#             model=model,
#             tokenizer=tokenizer,
#             special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
#         )
#     if 'llama' in model_args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
#         # LLaMA tokenizer may not have correct special tokens set.
#         # Check and add them if missing to prevent them from being parsed into different tokens.
#         # Note that these are present in the vocabulary.
#         # Note also that `model.config.pad_token_id` is 0 which corresponds to `<unk>` token.
#         print('Adding special tokens.')
#         tokenizer.add_special_tokens({
#             "eos_token": tokenizer.convert_ids_to_tokens(model.config.eos_token_id),
#             "bos_token": tokenizer.convert_ids_to_tokens(model.config.bos_token_id),
#             "unk_token": tokenizer.convert_ids_to_tokens(
#                 model.config.pad_token_id if model.config.pad_token_id != -1 else tokenizer.pad_token_id
#             ),
#         })

#     return model, tokenizer

# def get_dataset(data_args: DataArguments):
#     full_dataset_path = os.path.join(data_args.data_dir, data_args.dataset_name)
#     return load_dataset("json", data_files=full_dataset_path, split="train")

#     # dataset = Dataset.from_json(path_or_paths=full_dataset_path)
#     # # dataset = dataset.map(lambda x: {'input_output': x['input'] + x['output']})
#     # return dataset.remove_columns([col for col in dataset.column_names if col not in ['input', 'output']])

# def get_formatting_func() -> callable:

#     def _get_formatting_func(example):
#         return [
#             f"### Question: {example['prompt'][i]}\n ### Answer: {example['completion'][i]}"
#             for i in range(len(example['prompt']))
#         ]

#     return _get_formatting_func

# def get_data_collator(tokenizer) -> str:
#     response_template = " ### Answer:"
#     return DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

# def get_last_checkpoint(checkpoint_dir: str, continue_on_ckpt: bool, ) -> Tuple[Optional[str], bool]:
    
#     if not isdir(checkpoint_dir) or not continue_on_ckpt:
#         return None
    
#     if exists(join(checkpoint_dir, 'completed')):
#         print("Model training is already done! Terminating.")
#         sys.exit(0)
        
#     max_step = 0
#     for filename in os.listdir(checkpoint_dir):
#         if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
#             max_step = max(max_step, int(filename.replace('checkpoint-', '')))
    
#     if max_step > 0:
#         checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
#         print(f"Continuing on checkpoint {checkpoint_dir}.")
#         return checkpoint_dir
            
#     print(f"Directory passed {checkpoint_dir} only has checkpoint 0. Starting training from scratcht.")
#     return None

# def print_trainable_parameters(model_args: ModelArguments, model: AutoModelForCausalLM):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     if model_args.bits == 4: trainable_params /= 2
#     print(
#         f"trainable params: {trainable_params} || "
#         f"all params: {all_param} || "
#         f"trainable: {100 * trainable_params / all_param}"
#     )

# def _resize_tokenizer_and_model(
#         model: AutoModelForCausalLM,
#         tokenizer: AutoTokenizer,
#         special_tokens_dict: dict,
# ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
#     """Resize tokenizer and embedding.

#     Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
#     """
#     num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
#     model.resize_token_embeddings(len(tokenizer))

#     if num_new_tokens > 0:
#         input_embeddings_data = model.get_input_embeddings().weight.data
#         output_embeddings_data = model.get_output_embeddings().weight.data

#         input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
#         output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

#         input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
#         output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

#     return model, tokenizer


# def _set_datatypes(model_arguments: ModelArguments, model: AutoModelForCausalLM) -> None:
#     for name, module in model.named_modules():
#         if isinstance(module, LoraLayer):
#             if model_arguments.torch_dtype == "bfloat16":
#                 module = module.to(torch.bfloat16)
#         if 'norm' in name:
#             module = module.to(torch.float32)
#         if 'lm_head' in name or 'embed_tokens' in name:
#             if hasattr(module, 'weight'):
#                 if (
#                     model_arguments.torch_dtype == "bfloat16"
#                     and module.weight.dtype == torch.float32
#                 ):
#                     module = module.to(torch.bfloat16)


# def _find_all_linear_names(model_args: ModelArguments, model: AutoModelForCausalLM) -> List[str]:
    # if model_args.bits == 4:
    #     cls = bnb.nn.Linear4bit
    # elif model_args.bits == 4:
    #     cls = bnb.nn.Linear8bitLt
    # else:
    #     cls = torch.nn.Linear
    # lora_module_names = set()
    # for name, module in model.named_modules():
    #     if isinstance(module, cls):
    #         names = name.split('.')
    #         lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    # if 'lm_head' in lora_module_names:  # needed for 16-bit
    #     lora_module_names.remove('lm_head')
    # return list(lora_module_names)