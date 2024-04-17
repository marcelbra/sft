import os

from typing import Any, Dict, List, Optional
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field, fields

from transformers import (
    HfArgumentParser,
    TrainingArguments
)


def get_arguments() -> Namespace:
    """
    Gets the arguments from the command line or accepts a pre-defined list
    of arguments such that it can be used programatically.

    :param predefined_args: The pre-defined arguments.
    :return: The arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, help="Specifies the path to the config.",)
    return parser.parse_args()


class H4ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, allow_extra_keys: bool = False, other_args: Optional[List[str]] = None) -> List[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg), allow_extra_keys=allow_extra_keys)

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == List[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs


@dataclass
class SFTTrainingArguments(TrainingArguments):
    """
    Arguments for Training.
    """
    run_name: str = field(default="", metadata={"help": "The name of the run"})
    double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    max_memory_MB: int = field(default=80000, metadata={"help": "Free memory per gpu."})
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={"help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={"help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=-1, metadata={"help": 'How many optimizer update steps to take'})
    num_train_epochs: int = field(default=3)
    weight_decay: float = field(default=0.0, metadata={"help": 'The L2 weight decay rate of AdamW'})  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": 'The learnign rate'})
    max_grad_norm: float = field(default=0.3, metadata={"help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True, metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    gradient_checkpointing_kwargs: Optional[dict] = field(default_factory=lambda: {"use_reentrant": False}, metadata={"help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='constant', metadata={"help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10, metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=False, metadata={"help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=40, metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    max_seq_length: int = field(default=4096, metadata={"help": 'maximum sequence length for SFTTrainer'})
    dataset_kwargs: Optional[Dict[str, Any]] = field(default=None, metadata={"help": "Dataset kwargs for the SFTTrainer"})
    logging_first_step: bool = field(default=True, metadata={"help": ("Whether to log and evaluate the first global_step or not.")},)
    optim: Optional[str] = field(default="adamw_torch")
    report_to: Optional[List[str]] = field(default_factory=lambda: ["wandb"], metadata={"help": "The list of integrations to report the results and logs to."})
    continue_on_ckpt: bool = field(default=False,metadata={"help": "Whether to continue training from checkpoint if one was found."})
    train_on_source: Optional[bool] = field(default=False, metadata={"help": "Whether to train on the input in addition to the target text."})
    max_memory_MB: Optional[int] = field(default=24000, metadata={"help": "Available memory per GPU."})
    full_finetune: Optional[bool] = field(default=False, metadata={"help": "Whether to perform full fine-tuning."})
    predict_with_generate: Optional[bool] = field(default=False, metadata={"help": ""})

@dataclass
class GenerationArguments:
    """
    Aguments for generation during inference. More info: https://shorturl.at/amAFJ
    """
    max_new_tokens: Optional[int] = field(default=256, metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops if predict_with_generate is set."})
    min_new_tokens: Optional[int] = field(default=None, metadata={"help": "Minimum number of new tokens to generate."})
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


@dataclass
class ModelArguments:
    """
    Arguments for model, tokenizer, lora.
    """
    model_name_or_path: Optional[str] = field(default=None, metadata={ "help": ("The model checkpoint for weights initialization. Don't set if you want to train a model from scratch.")},)
    torch_dtype: Optional[str] = field(default="bfloat16", metadata={"help": ("Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights."), "choices": ["auto", "bfloat16", "float16", "float32"],},)
    tokenizer_name_or_path: Optional[str] = field(default=None, metadata={"help": ("The path to the tokenizer. Useful if you want to use a different tokenizer to the one stored in `model_name_or_path`.")},)
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust remote code when loading a model."})
    use_flash_attention_2: bool = field(default=False, metadata={"help": ("Whether to use flash attention 2. You must install this manually by running `pip install flash-attn --no-build-isolation`")},)
    use_peft: bool = field(default=True, metadata={"help": ("Whether to use PEFT or not for training.")},)
    lora_r: Optional[int] = field(default=16, metadata={"help": ("LoRA R value.")},)
    lora_alpha: Optional[int] = field(default=32, metadata={"help": ("LoRA alpha.")},)
    lora_dropout: Optional[float] = field(default=0.05, metadata={"help": ("LoRA dropout.")},)
    lora_target_modules: Optional[List[str]] = field(default=None, metadata={"help": ("LoRA target modules.")},)
    lora_modules_to_save: Optional[List[str]] = field(default=None, metadata={"help": ("Model layers to unfreeze & train")},)
    load_in_8bit: bool = field(default=False, metadata={"help": "use 8 bit precision"})
    load_in_4bit: bool = field(default=False, metadata={"help": "use 4 bit precision"})
    lora_bias: float = field(default=None, metadata={"help": ""})
    bnb_4bit_use_double_quant: bool = field(default=True, metadata={"help": "Compress the quantization statistics through double quantization."})
    bnb_4bit_quant_type: Optional[str] = field(default="nf4", metadata={"help": "precise the quantization type (fp4 or nf4)"})
    use_bnb_nested_quant: bool = field(default=False, metadata={"help": "use nested quantization"})
    modules_to_save: Optional[List[str]] = field(default=None, metadata={"help":""})
    padding_side: Optional[str] = field(default='right', metadata={"help":""})
    bits: Optional[int] = field(default=4, metadata={"help":"With how many bits to train the model"})

    def __post_init__(self):
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("You can't use 8 bit and 4 bit precision at the same time")


@dataclass
class DataArguments:
    """
    Arguments for training and eval.
    """
    chat_template: Optional[str] = field(default=None, metadata={"help": "The chat template to use."})
    dataset_mixer: Optional[Dict[str, float]] = field(default=None, metadata={"help": ("Datasets and their proportions to be used for training ift/rl.")},)
    text_column: Optional[str] = field(default="text", metadata={"help": "The column name to use for the text in the dataset (only used for continued pretraining)."})
    dataset_splits: Optional[List[str]] = field(default_factory=lambda: ["train", "test"], metadata={"help": ("List of train test splits to use in the dataset")},)
    dataset_configs: Optional[List[str]] = field(default=None, metadata={"help": "List of dataset config names. If given must be the same length as 'dataset_mixer' keys."},)
    preprocessing_num_workers: Optional[int] = field(default=None, metadata={"help": "The number of processes to use for the preprocessing."},)
    truncation_side: Optional[str] = field(default=None, metadata={"help": "Truncation side to use for the tokenizer."})
    auto_insert_empty_system_msg: bool = field(default=True, metadata={"help": ("Whether to automatically insert an empty system message as the first message if `system` is mentioned in the chat template.")})
    max_train_samples: Optional[int] = field(default=None, metadata={"help": "For debugging purposes or quicker training, truncate the number of training examples to this value if set."})
    source_max_len: int = field(default=1024, metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},)
    target_max_len: int = field(default=256, metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},)
    packing: bool = field(default=False, metadata={"help": "Apply packing when fine-tuning or not"})
    data_dir: Optional[str] = field(default="", metadata={"help": "The directory path to the data."})
    dataset_name: Optional[str] = field(default="data/train_formatted_data.json", metadata={"help": "The file name (and possibly path) to the file.", "required": True})
    load_with_result: Optional[bool] = field(default=True, metadata={"help": "In wich format to load the GSM8K dataset."})
