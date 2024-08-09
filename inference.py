import os
import json

from typing import List, Optional
from argparse import ArgumentParser
from collections import defaultdict, Counter

from tqdm import tqdm
from vllm import SamplingParams, LLMEngine, EngineArgs
from vllm.lora.request import LoRARequest

from prompting import build_source_prompt
from transformers import AutoTokenizer


def get_max_step_adapter_name(output_dir):
    ckpt_name = 'checkpoint-'
    max_step = max([
        int(file_name.split(ckpt_name)[1])
        for file_name in os.listdir(output_dir)
        if file_name.startswith(ckpt_name)
    ])
    return os.path.join(output_dir, f"{ckpt_name}{max_step}", "adapter_model")


def initialize_engine(model_name_or_path, gpus) -> LLMEngine:
    print("Initialize the LLMEngine.")
    kwargs = {}
    if ("phi" in model_name_or_path.lower()):
        kwargs = {
            "trust_remote_code": True,
            "max_num_batched_tokens": 64000,
            "max_model_len": 22200
        }
    if ("mistral" in model_name_or_path.lower()):
        kwargs = {
            "gpu_memory_utilization": 1,
            "max_num_batched_tokens": 64000,
            "max_model_len": 15000
        }
    if ("gemma-1.1-7b-it" in model_name_or_path.lower()):
        kwargs = {
            "gpu_memory_utilization": 1,
            "max_model_len": 5200
        }
    if ("qwen" in model_name_or_path.lower()):
        kwargs = {
            "max_num_batched_tokens": 64000,
            "max_model_len": 32000
        }
    
    print(f"Setting kwargs:")
    print(*list(kwargs.items()), sep="\n")
    engine_args = EngineArgs(
        model=model_name_or_path,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        max_num_seqs=256,
        tensor_parallel_size=gpus,
        **kwargs
    )
    return LLMEngine.from_engine_args(engine_args)


def get_sampling_params(
        eot_token: str,
        temperature: float = 0.0
    ):
    print("Initalizating sampling params.")
    if eot_token == "linebreak":
        stop = ["\n"]
    else:
        stop = [eot_token]
    params = SamplingParams(
        temperature=temperature,
        max_tokens=2048,
        stop=stop
    )
    print(f"Sampling params:\n{params}")
    return params


def get_lora_request(lora_path: str):
    print(f"Getting max adapter checkpoint from {lora_path}.")
    adapter_ckpt = get_max_step_adapter_name(lora_path)
    print(f"Loaded ckpt {adapter_ckpt}.")
    return LoRARequest("lora", 1, adapter_ckpt)


def format(data):
            
    # Group by question
    formatted_data = defaultdict(list)
    for element in data:
        key = element["instruction"]
        value = element["prediction"]
        formatted_data[key].append(value)

    new_formatted_data = {}
    for key, _ in formatted_data.items():
        new_key = key.split("\n\n### Input:\nQuestion: ")[1].split("\n\n### Response:")[0]
        new_formatted_data[new_key] = formatted_data[key]
    formatted_data = new_formatted_data

    new_formatted_data = {}
    for key, value in formatted_data.items():
        counted_values = Counter(value)
        # Sort by count
        sorted_values = sorted(counted_values.items(), key=lambda x: x[1], reverse=True)
        new_formatted_data[key] = sorted_values
    formatted_data = new_formatted_data

    # Move the count to the front
    new_formatted_data = {}
    for key, answer_counts in formatted_data.items():
        for i, answer_count in enumerate(answer_counts):
            # Change the order of the tuple
            answer_count = (answer_count[1], answer_count[0])
            answer_counts[i] = answer_count
        new_formatted_data[key] = answer_counts
    formatted_data = new_formatted_data

    return formatted_data


def write_results(run_name, next_steps, final_result, postfix, experiment):

        file_name = f"next_step_predictions{postfix}.json"
        next_step_path = os.path.join(run_name, file_name)
        print(f"Writing {next_step_path}.")

        if experiment=="sample":
            next_steps = format(next_steps)

        with open(next_step_path, "w") as f:
            json.dump(next_steps, f, indent=4)#, ensure_ascii=False)

        file_name = f"final_results{postfix}.json"
        final_results_path = os.path.join(run_name, file_name)
        print(f"Writing {final_results_path}.")
        with open(final_results_path, "w") as f:
            json.dump(final_result, f, indent=4)#, ensure_ascii=False)


def inference_TEC(data, sampling_params, lora_request):
    formatted_data = []
    for i, element in enumerate(data):
        if "<final answer>" in element["prediction"]:
            continue
        question, steps = element["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:")
        question.replace("Question: ", "")
        maybe_linebreak = "\n" if not steps.endswith("\n") and steps else ""
        instruction, replaced, new_target_steps = build_source_prompt(
            question=question,
            gt_steps=element["prediction"]
            # gt_steps=steps + maybe_linebreak + element["prediction"]
        )

        instruction += "\n" # !!!!! ACHTUNG KANN BUGS MACHEN NUR FÜR JETZT !!!!!!

        if i <= 32:
            print(instruction)
            print("---")

        formatted_data.append(
            {
                "instruction": instruction,
                "sampling_params": sampling_params,
                "lora_request": lora_request
            }
        )
    
    return formatted_data

    
def inference_sample(data, sampling_params, lora_request, previous_step_n, n):

    formatted_data = []
    for element in data:
        
        steps = element["prediction"].count("\n")

        if steps <= previous_step_n:
            continue
        
        question = element["instruction"].split("\n\n### Input:\nQuestion: ")[1].split("\n\n### Response:")[0].split("\n<step ")[0]
        step = element["prediction"].split("\n")[previous_step_n-1].replace(f"<step {previous_step_n}>: ", "")
        instruction, replaced, new_target_steps = build_source_prompt(
            question=question,
            steps=step,
            previous_step_n=previous_step_n
        )

        print("Source prompt:")
        print(instruction)
        print("-----------------")

        formatted_data += [
            {
                "instruction": instruction,
                "sampling_params": sampling_params,
                "lora_request": lora_request
            }
            for _ in range(n)   
        ]

    return formatted_data


def inference_normal(data_dir, sampling_params, lora_request, n):

    print(f"DP Load data from {data_dir}")
    with open(data_dir) as f:
        data = json.load(f)

    print("Format data.")
    formatted_data = []
    print("Sample data:")
    for i in range(n):
        for element in data:
            instruction, replaced, new_target_steps = build_source_prompt(
                element["source_question"],
                element["source_steps"]
            )
            
            # if not i:
            #     print(instruction)
            #     print("---")
            
            formatted_data.append(
                {
                    "instruction": instruction,
                    "sampling_params": sampling_params,
                    "lora_request": lora_request
                }
            )
            
    return formatted_data


def inference_encoder(
        data: dict,
        sampling_params: SamplingParams,
        lora_request: LoRARequest,
        n: int = 1,
        type_: Optional[str] = None
    ):

    formatted_data = []

    for question, predictions in data.items():

        instruction, replaced, new_target_steps = build_source_prompt(
            question=question,
            gt_steps="",
            first_step_data=predictions,
            type_=type_
        )

        print("Source prompt:")
        print(instruction)
        print("-----------------")

        formatted_data += [
            {
                "instruction": instruction,
                "sampling_params": sampling_params,
                "lora_request": lora_request
            }
            for _ in range(n)   
        ]

    return formatted_data


def format_data(
        data_dir,
        sampling_params,
        lora_request,
        n: int = 1,
        previous_next_steps=None,
        amount_samples=None,
        previous_step_n = 0,
        experiment: Optional[str] = None,
        type_: Optional[str] = None
    ):
    print(f"Experiment: {experiment}")
    if previous_next_steps:
        print(f"Load data from {previous_next_steps}")
        with open(previous_next_steps) as f:
            data = json.load(f)
        if experiment=="TEC":
            formatted_data = inference_TEC(data, sampling_params, lora_request)
        elif experiment=="sampling":
            formatted_data = inference_sample(data, sampling_params, lora_request, previous_step_n, n)
        elif experiment=="encoder":
            formatted_data = inference_encoder(data, sampling_params, lora_request, n, type_)
        elif experiment=="oracle":
            pass
            # formatted_data = inference_oracle(data, sampling_params, lora_request, n, type_)
    else:
        print("Conducting normal inference.")
        formatted_data = inference_normal(data_dir, sampling_params, lora_request, n)

    if amount_samples:
        formatted_data = formatted_data[:amount_samples]
    
    return formatted_data


def filter_out_results(results, delimiter = "\n<final answer>: "):

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
    data_dir: str,
    n: int = 1,
    previous_next_steps: Optional[str] = None,
    postfix: Optional[str] = "",
    eos_token: str = "<eos>",
    previous_step_n: int = 0,
    experiment: str = "normal",
    type_: Optional[str] = None,
    temperature: float = 0.0,
    gpus: int = 1
):          
    sampling_params = get_sampling_params(
        eot_token=eos_token,
        temperature=temperature
    )
    lora_request = get_lora_request(lora_path=run_name)
    data = format_data(
        data_dir=data_dir,
        sampling_params=sampling_params,
        lora_request=lora_request,
        n=n,
        previous_next_steps=previous_next_steps,
        previous_step_n=previous_step_n,
        experiment=experiment,
        type_=type_
    )
    engine = initialize_engine(model_name_or_path, gpus)
    results = process_requests(engine, data)
    next_steps, done = filter_out_results(results)
    write_results(run_name, next_steps, done, postfix, experiment)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--previous_next_steps", type=str, default=None)
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--previous_step_n", type=int)
    parser.add_argument("--data_dir", type=str, default="/cluster/work/lawecon/Work/mbraasch/data/gsm8k_test.json")
    parser.add_argument("--experiment", type=str, default="encoder")
    parser.add_argument("--type", type=str, default="training", choices=["training", "inference_tf", "inference", "oracle", "new_dawn"])
    parser.add_argument("--eos_token", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()

    kwargs = {}
    if ("phi" in args.model_name_or_path.lower()):
        kwargs["trust_remote_code"] = True
    if not args.eos_token:
        args.eos_token = AutoTokenizer.from_pretrained(args.model_name_or_path, **kwargs).eos_token
    print(f"EOS token: '{args.eos_token}'")
    print(f"Temperature: {args.temperature}")
    print(*list(vars(args).items()), sep="\n")
    

    run_inference(
        run_name=args.run_name,
        model_name_or_path=args.model_name_or_path,
        previous_next_steps=args.previous_next_steps,
        data_dir=args.data_dir,
        n=args.n,
        postfix=args.postfix,
        eos_token=args.eos_token,
        previous_step_n=args.previous_step_n,
        experiment=args.experiment,
        type_=args.type,
        temperature=args.temperature,
        gpus=args.gpus
    )
