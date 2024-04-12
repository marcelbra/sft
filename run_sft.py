import json
import os

from argparse import Namespace

from trl import SFTTrainer
from transformers import set_seed

from src.arguments import (
    get_arguments,
    ModelArguments,
    DataArguments,
    SFTTrainingArguments,
    H4ArgumentParser
)

from src.training import (
    get_last_checkpoint,
    get_model_and_tokenizer,
    get_data_module,
    print_trainable_parameters,
    verify_datatypes_parameter_counts,
    SavePeftModelCallback
)

def train(cli_args: Namespace):

    print("-- Get args")

    parser = H4ArgumentParser((
        ModelArguments, DataArguments, SFTTrainingArguments
    ))
    all_args = parser.parse_yaml_and_args(
        yaml_arg=cli_args.config, other_args=[], allow_extra_keys=True
    )
    for args in all_args:
        # TODO: format
        print(str(type(args)), ":")
        print(args, "\n")
    model_args, data_args, training_args = all_args

    print("-- Get model, tokenizer and data")

    output_dir_with_name = os.path.join(training_args.output_dir, training_args.run_name)
    if not os.path.exists(output_dir_with_name):
        os.makedirs(output_dir_with_name)

    set_seed(training_args.seed)

    checkpoint_dir = get_last_checkpoint(output_dir_with_name, training_args.continue_on_ckpt)
    model, tokenizer = get_model_and_tokenizer(model_args, training_args, checkpoint_dir)
    data_module = get_data_module(tokenizer=tokenizer, data_args=data_args, model_args=model_args)

    print("-- Initalize trainer")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        dataset_text_field="input_output",
        max_seq_length=training_args.max_seq_length,
        **data_module,
    )
    trainer.add_callback(SavePeftModelCallback)

    print_trainable_parameters(model_args, model)
    verify_datatypes_parameter_counts(model)

    print("-- Start training")

    # Note: `resume_from_checkpoint` currently not supported for
    # adapter checkpoints by HF. Currently adapter checkpoint is
    # reloaded as expected but optimizer/scheduler states are not.
    all_metrics = {"run_name": training_args.run_name}
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
        with open(os.path.join(output_dir_with_name, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))


if __name__ == "__main__":
    cli_args = get_arguments()
    train(cli_args=cli_args)
