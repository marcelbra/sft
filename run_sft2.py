import os
import json

from argparse import Namespace
from trl import SFTTrainer

from transformers import (
    TrainerCallback,
    trainer_utils
)

from src.arguments import (
    get_arguments,
    ModelArguments,
    DataArguments,
    SFTTrainingArguments,
    H4ArgumentParser
)


from src.training import (
    get_accelerate_model,
    load_data,
    DataCollatorForCausalLM
)


class SavePeftModelCallback(TrainerCallback):
    def save_model(self, args, state, kwargs):
        print('Saving PEFT checkpoint...')
        if state.best_model_checkpoint is not None:
            checkpoint_folder = os.path.join(state.best_model_checkpoint, "adapter_model")
        else:
            output_dir = os.path.join(args.output_dir, args.run_name)
            checkpoint_folder = os.path.join(output_dir, f"{trainer_utils.PREFIX_CHECKPOINT_DIR}-{state.global_step}")

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

        touch(os.path.join(args.output_dir, 'completed'))
        self.save_model(args, state, kwargs)


def train(cli_args: Namespace):

    parser = H4ArgumentParser((
        ModelArguments, DataArguments, SFTTrainingArguments
    ))
    model_args, data_args, training_args = parser.parse_yaml_and_args(
        yaml_arg=cli_args.config, other_args=[], allow_extra_keys=True
    )

    model, tokenizer = get_accelerate_model(training_args, model_args)

    dataset = load_data(data_args)

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=data_args.source_max_len,
        target_max_len=data_args.target_max_len,
        train_on_source=training_args.train_on_source,
        predict_with_generate=training_args.predict_with_generate,
    )

    trainer = SFTTrainer(
        model=model,
        packing=data_args.packing,
        train_dataset=dataset,
        args=training_args,
        data_collator=data_collator
    )

    output_dir_with_name = os.path.join(training_args.output_dir, training_args.run_name)
    if not os.path.exists(output_dir_with_name):
        os.makedirs(output_dir_with_name)

    all_metrics = {"run_name": training_args.run_name}
    if training_args.do_train:
        trainer.add_callback(SavePeftModelCallback)
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

"""
sbatch \
    --gpus=rtx_4090:1 \
    --mem-per-cpu=8G \
    --time=0-10 \
    --wrap=" \
        cd repos/sft; \
        python3 run_sft2.py \
            --config /cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output/deepseek-llm-7b-base-baseline-with-packing/config.yaml"
"""