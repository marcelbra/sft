import sys
import os
import logging

import torch
import bitsandbytes as bnb
import transformers

from os.path import exists, join, isdir
from typing import Tuple, List, Optional

from trl import DataCollatorForCompletionOnlyLM
from accelerate import PartialState
from datasets import Dataset
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
    DataArguments
)


DEFAULT_PAD_TOKEN = "[PAD]"


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


def get_model_and_tokenizer(
        model_args: ModelArguments,
        training_args: SFTTrainingArguments,
        checkpoint_dir: str
    ) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:

    print("Load GPU settings.")
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    device_map = "auto"
    # For distributed training, set device map and max memory per device
    if n_gpus > 1:
        device_string = PartialState().process_index
        device_map = {'': device_string}

    print("Set torch bit settings")
    if model_args.torch_dtype not in {"auto", None}:
        compute_dtype = getattr(torch, model_args.torch_dtype)
    if compute_dtype == torch.float16 and model_args.bits == 4:
        if torch.cuda.is_bf16_supported() and model_args.torch_dtype != 'bfloat16':
            print('=' * 80)
            print('Your GPU supports bfloat16, you can accelerate training by setting torch_dtype to "bfloat16"')
            print('=' * 80)

    print("Load quantization config.")
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=model_args.bits == 4,
        load_in_8bit=model_args.bits == 8,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=model_args.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=model_args.bnb_4bit_quant_type
    )

    print(f'Load model {model_args.model_name_or_path}.')
    print(f'Using cache {os.environ["HF_HOME"]}')
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        local_files_only=False
    )
    model.config.use_cache = False

    print("Load model for kbit training.")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    print(f'Load LoRA model.')
    if checkpoint_dir is None:
        modules = _find_all_linear_names(model_args, model)
        config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=modules,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        _set_datatypes(model_args, model)
    else:
        print("Load adapters from checkpoint.")
        model = PeftModel.from_pretrained(model, join(checkpoint_dir, 'adapter_model'), is_trainable=True, local_files_only=True,)

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    print("Load tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=os.environ["HF_HOME"],
        padding_side=model_args.padding_side,
        use_fast=False,  # Fast tokenizer giving issues
        trust_remote_code=model_args.trust_remote_code,
        local_files_only=False,
    )
    if tokenizer._pad_token is None:
        model, tokenizer = _resize_tokenizer_and_model(
            model=model,
            tokenizer=tokenizer,
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
        )
    if 'llama' in model_args.model_name_or_path or isinstance(tokenizer, LlamaTokenizer):
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

    return model, tokenizer


def get_data_module(
    tokenizer: AutoTokenizer, 
    data_args: DataArguments,
    model_args: ModelArguments,
    training_args: SFTTrainingArguments
) -> dict:

    full_dataset_path = os.path.join(data_args.dataset_dir, data_args.dataset_name)
    full_dataset = Dataset.from_json(path_or_paths=full_dataset_path)
    dataset = full_dataset.map(lambda x: {'input_output': x['input'] + x['output']})
    dataset = dataset.remove_columns([col for col in dataset.column_names if col not in ['input_output']])

    data_module = {
        "packing": data_args.packing,
        "train_dataset": dataset,
    }

    if not data_args.packing and not training_args.train_on_source:

        def get_response_template(model_name, tokenizer):
            if "gemma" in model_name:
                return "<start_of_turn>model\n"
            elif "llama" in model_name:
                temp = "\n [/INST] "
                template_ids = tokenizer.encode(temp, add_special_tokens=False)[2:]
                return template_ids
            elif "zephyr" in model_name:
                return "\n<|assistant|>\n"
            else:
                return "<start_of_turn>model\n" # ChatLM as default
        
        data_collator = DataCollatorForCompletionOnlyLM(get_response_template(model_args.model_name_or_path, tokenizer), tokenizer=tokenizer)
        
        return dict(**data_module, data_collator=data_collator)
    
    return data_module


def get_last_checkpoint(checkpoint_dir: str, continue_on_ckpt: bool, ) -> Tuple[Optional[str], bool]:
    
    if not isdir(checkpoint_dir) or not continue_on_ckpt:
        return None
    
    if exists(join(checkpoint_dir, 'completed')):
        print("Model training is already done! Terminating.")
        sys.exit(0)
        
    max_step = 0
    for filename in os.listdir(checkpoint_dir):
        if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
            max_step = max(max_step, int(filename.replace('checkpoint-', '')))
    
    if max_step > 0:
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Continuing on checkpoint {checkpoint_dir}.")
        return checkpoint_dir
            
    print(f"Directory passed {checkpoint_dir} only has checkpoint 0. Starting training from scratcht.")
    return None


def print_trainable_parameters(model_args: ModelArguments, model: AutoModelForCausalLM):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if model_args.bits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


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


def _resize_tokenizer_and_model(
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        special_tokens_dict: dict,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
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

    return model, tokenizer


def _set_datatypes(model_arguments: ModelArguments, model: AutoModelForCausalLM) -> None:
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if model_arguments.torch_dtype == "bfloat16":
                module = module.to(torch.bfloat16)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if (
                    model_arguments.torch_dtype == "bfloat16"
                    and module.weight.dtype == torch.float32
                ):
                    module = module.to(torch.bfloat16)


def _find_all_linear_names(model_args: ModelArguments, model: AutoModelForCausalLM) -> List[str]:
    if model_args.bits == 4:
        cls = bnb.nn.Linear4bit
    elif model_args.bits == 4:
        cls = bnb.nn.Linear8bitLt
    else:
        cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)