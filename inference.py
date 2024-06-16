import os
import json

from typing import List
from argparse import ArgumentParser, Namespace

from tqdm import tqdm
from vllm import SamplingParams, LLMEngine, EngineArgs
from vllm.lora.request import LoRARequest

from evaluate import calc_metrics
from prompting import build_source_prompt

def get_inference_arguments() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--data_path", type=str, help="Adds a default data directory to the front, only the json specification is needed.", default=None)
    parser.add_argument("--instruction", type=str, default="Solve the given math problem step by step and put your final answer after 'Final answer: '.")
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-2b-it")
    parser.add_argument("--output_dir", type=str, help="Specifies the path to the directory where everything is happening.", default="/cluster/work/lawecon/Work/mbraasch/output/")
    parser.add_argument("--data_dir", type=str, help="Adds a default data directory to the front, only the json specification is needed.",default="/cluster/work/lawecon/Work/mbraasch/data")
    parser.add_argument("--target_file_name", type=str, default="next_step_predictions.json")
    parser.add_argument("--start_from", type=int, default=None)
    parser.add_argument("--amount_samples", type=int, default=None)
    return parser.parse_args([])


def get_max_step_adapter_name(output_dir):
    ckpt_name = 'checkpoint-'
    max_step = max([
        int(file_name.split(ckpt_name)[1])
        for file_name in os.listdir(output_dir)
        if file_name.startswith(ckpt_name)
    ])
    return os.path.join(output_dir, f"{ckpt_name}{max_step}", "adapter_model")


def format_question(question: str, formatting_template: str):
    return formatting_template.format(question.strip()).lstrip()


def initialize_engine(model_name_or_path) -> LLMEngine:
    print("Initialize the LLMEngine.")
    engine_args = EngineArgs(
        model=model_name_or_path,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        max_num_seqs=256,
        trust_remote_code=True,
        max_num_batched_tokens=64000,
        max_model_len=22200
    )
    return LLMEngine.from_engine_args(engine_args)


def get_sampling_params(eot_token):
    print("Initalizating sampling params.")
    return SamplingParams(
        temperature=0,
        max_tokens=2048,
        stop=[eot_token]
    )


def get_lora_request(lora_path: str):
    print(f"Getting max adapter checkpoint from {lora_path}.")
    adapter_ckpt = get_max_step_adapter_name(lora_path)
    print(f"Loaded ckpt {adapter_ckpt}.")
    return LoRARequest("lora", 1, adapter_ckpt)

def split_steps(next_steps: List[dict]):

    same = 0
    different = 0
    error_count = 0
    strategy_one = [] # Here, we naively split off the first step
    strategy_two = [] # Here, we only split if the first step has a "=", if not we go on until the next and remove the "\n"s in between

    for element in next_steps:

        steps = [x for x in element["prediction"].split("\n") if x]
        if not steps:
            error_count += 1
            print("Error:")
            print(element)
            continue

        # strategy one
        # Add "." if the step does not end with a "." or with a ":"
        if not steps[0].endswith(".") and not steps[0].endswith(":"):
            steps[0] += "."
        new_element_s1 = {
            "instruction": element["instruction"],
            "prediction": steps[0]
        }
        strategy_one.append(new_element_s1)

        # strategy two
        new_element_s2 = {
            "instruction": element["instruction"],
            "prediction": ""
        }
        new_steps = ""
        for step in steps:
            if "=" in step:
                new_steps += step + (". " if not step.endswith(".") or step.endswith(":") else "")
                break
            else:
                if not step.endswith(".") and not step.endswith(":"):
                    new_steps += step + ". "
                elif step.endswith(".") or step.endswith(":"):
                    new_steps += step + " "
        new_element_s2["prediction"] = new_steps.strip()
        new_steps = new_steps.strip()
        if not new_steps.endswith(".") and not new_steps.endswith(":"):
            new_steps += "."
        new_element_s2["prediction"] = new_steps.strip()
        strategy_two.append(new_element_s2)
    
        # Statistics on how much overlap there is
        if new_element_s1["prediction"] == new_element_s2["prediction"]:
            same += 1
        else:
            different += 1

    print(f"Overlap in percentage: {same / (same + different) * 100}%")
    print(f"Error count: {error_count}")

    return {"s1": strategy_one, "s2": strategy_two}


def write_results(run_name, next_steps, final_result, previous_next_steps, postfix):

        strategy = ""
        if previous_next_steps:
            if "s1" in previous_next_steps:
                strategy = "s1" 
                print(f"Strategy specified: {strategy}")
            elif "s2" in previous_next_steps:
                strategy = "s2"
                print(f"Strategy specified: {strategy}")
        else:
            print("No strategy specified.")

        # First, write the next step predictions
        file_name = f"{strategy}_next_step_predictions{postfix}.json" if strategy else f"next_step_predictions{postfix}.json"
        next_step_path = os.path.join(run_name, file_name)
        print(f"Writing {next_step_path}.")
        with open(next_step_path, "w") as f:
            json.dump(next_steps, f, indent=4, ensure_ascii=False)

        # Then, write the final results
        file_name = f"{strategy}_final_results{postfix}.json" if strategy else f"final_results{postfix}.json"
        final_results_path = os.path.join(run_name, file_name)
        print(f"Writing {final_results_path}.")
        with open(final_results_path, "w") as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)

        # Lastly, split the data and write it
        if strategy:
            split = split_steps(next_steps)
            split_next_steps = os.path.join(run_name, f"{strategy}_next_step_predictions_{strategy}{postfix}.json")
            final_split = split["s1"] if "s1" in split_next_steps else split["s2"]
            print(f"Writing {split_next_steps}.")
            with open(split_next_steps, "w") as f:
                json.dump(final_split, f, indent=4, ensure_ascii=False)
            # else:
            # split_next_steps_s1 = os.path.join(run_name, f"s1_next_step_predictions_s1{postfix}.json")
            # print(f"Writing {split_next_steps_s1}.")
            # with open(split_next_steps_s1, "w") as f:
            #     json.dump(split["s1"], f, indent=4, ensure_ascii=False)
            # split_next_steps_s2 = os.path.join(run_name, f"s2_next_step_predictions_s2{postfix}.json")
            # print(f"Writing {split_next_steps_s2}.")
            # with open(split_next_steps_s2, "w") as f:
            #     json.dump(split["s2"], f, indent=4, ensure_ascii=False)


def format_data(data_dir, sampling_params, lora_request, previous_next_steps=None, amount_samples=None):
            
    if previous_next_steps:

        print(f"Load data from {previous_next_steps}")
        with open(previous_next_steps) as f:
            data = json.load(f)
        formatted_data = []
        for element in data:
            if "Final answer" in element["prediction"]:
                continue
            question, steps = element["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:\n")
            maybe_linebreak = "\n" if not steps.endswith("\n") else ""
            formatted_data.append(
                {
                    "instruction": build_source_prompt(
                        question=question,
                        steps=steps + maybe_linebreak + element["prediction"]
                    ),
                    "sampling_params": sampling_params,
                    "lora_request": lora_request
                }
            )

    else:

        print(f"DP Load data from {data_dir}")
        with open(data_dir) as f:
            data = json.load(f)

        print("Format data.")
        formatted_data = [
            {
                "instruction": build_source_prompt(
                    question=element["source_question"],
                    steps=element["source_steps"]
                ),
                "sampling_params": sampling_params,
                "lora_request": lora_request
            }
            for element in data
        ]

    # print("Sample test data:")
    # for datapoint in formatted_data[:5]:
    #     print(datapoint["instruction"])
    if amount_samples:
        formatted_data = formatted_data[:amount_samples]
    
    return formatted_data


def filter_out_results(results, delimiter = "\nFinal answer: "):

    print("Filter out results.")
    next_steps = []
    done = []

    for element in results:

        obj = {
            "instruction": element["instruction"],
            "prediction": element["prediction"],
        }
        
        if delimiter in element["prediction"]:
            obj["result"] = element["prediction"].split(delimiter)[1]
            done.append(obj)
        else:
            next_steps.append(obj)

    return next_steps, done


def process_requests(engine: LLMEngine, data: List):
    
    print("Start processing.")
    request_id = 0
    results = []

    with tqdm(total=len(data)) as pbar:
        while data or engine.has_unfinished_requests():
            if data:
                element = data.pop(0)
                engine.add_request(
                    str(request_id),
                    element["instruction"],
                    element["sampling_params"],
                    lora_request=element["lora_request"]
                )
                request_id += 1

            for request_output in engine.step():
                if request_output.finished:
                    results.append(
                        {
                            "instruction": request_output.prompt,
                            "prediction": request_output.outputs[0].text
                        }
                    )
                    pbar.update(1)
        
    return results


def run_inference(
    run_name: str,
    model_name_or_path: str,
    data_dir = "/cluster/work/lawecon/Work/mbraasch/data/gsm8k_test.json",
    previous_next_steps: str = None,
    last_step: bool = False,
    postfix: str = "",
    eos_token: str = "<eos>"
):  
    if previous_next_steps:
        print(f"Previous next steps given: {previous_next_steps}")
    else:
        print("No previous next steps given.")
        
    sampling_params = get_sampling_params(eot_token=eos_token)
    lora_request = get_lora_request(lora_path=run_name)
    data = format_data(
        data_dir=data_dir,
        sampling_params=sampling_params,
        lora_request=lora_request,
        previous_next_steps=previous_next_steps
    )
    engine = initialize_engine(model_name_or_path)
    results = process_requests(engine, data)
    next_steps, done = filter_out_results(results)
    write_results(run_name, next_steps, done, previous_next_steps, postfix)
    if last_step:
        calc_metrics(
            test_data_path=data_dir,
            output_dir=run_name
        )

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--previous_next_steps", type=str, default=None)
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--eos_token", type=str, default="<eos>")
    parser.add_argument("--data_dir", type=str, default="/cluster/work/lawecon/Work/mbraasch/data")
    args = parser.parse_args()

    run_inference(
        run_name=args.run_name,
        model_name_or_path=args.model_name_or_path,
        previous_next_steps=args.previous_next_steps,
        postfix=args.postfix,
        eos_token="<|endoftext|>",
        data_dir=args.data_dir
    )
"""
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:10:00 --wrap="python3 sft/inference.py \
    --run_name /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/3 \
    --model_name_or_path microsoft/Phi-3-mini-128k-instruct \
    --postfix _test_2_steps \
    --data_dir /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/test_2_steps.json"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:10:00 --wrap="python3 sft/inference.py \
    --run_name /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/3 \
    --model_name_or_path microsoft/Phi-3-mini-128k-instruct \
    --postfix _test_3_steps \
    --data_dir /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/test_3_steps.json"

sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:10:00 --wrap="python3 sft/inference.py \
    --run_name /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/34 \
    --model_name_or_path microsoft/Phi-3-mini-128k-instruct \
    --postfix _test_2_steps \
    --data_dir /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/test_2_steps.json"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:10:00 --wrap="python3 sft/inference.py \
    --run_name /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/34 \
    --model_name_or_path microsoft/Phi-3-mini-128k-instruct \
    --postfix _test_3_steps \
    --data_dir /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/test_3_steps.json"

sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:10:00 --wrap="python3 sft/inference.py \
    --run_name /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/baseline \
    --model_name_or_path microsoft/Phi-3-mini-128k-instruct \
    --postfix _test_2_steps \
    --data_dir /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/test_2_steps.json"
sbatch --gpus=rtx_3090:1 --mem-per-cpu=8G --time=00:10:00 --wrap="python3 sft/inference.py \
    --run_name /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/baseline \
    --model_name_or_path microsoft/Phi-3-mini-128k-instruct \
    --postfix _test_3_steps \
    --data_dir /cluster/work/lawecon/Work/mbraasch/output/phi-3-mini-instruct/gsm8k-gt/upscale_steps/test_3_steps.json"
"""