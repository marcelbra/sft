import os
import json

from typing import List

from tqdm import tqdm
from vllm import SamplingParams, LLMEngine, EngineArgs
from vllm.lora.request import LoRARequest

from prompting import build_source_prompt, EOT_TOKEN, M1, MI, MA
from arguments import get_inference_arguments
from settings import OUTPUT_DIR


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


def initialize_engine(args) -> LLMEngine:
    print("Initialize the LLMEngine.")
    engine_args = EngineArgs(
        model=args.model_name_or_path,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=64,
        max_num_seqs=256
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


def write_results(args, results, final_result, experiment_name):

        # First, write the results
        target_path = os.path.join(args.run_name, args.target_file_name)
        print(f"Writing results to {target_path}.")
        with open(target_path, "w") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

        # Now write back the final results but first try to load it (if exists) and append those datapoins that are not already in there
        target_path = os.path.join(args.run_name, "final_results.json")
        try:
            print(f"Writing final results to {target_path}.")
            with open(target_path, "r") as f:
                final_data = json.load(f)
        except FileNotFoundError:
            final_data = []

        for element in final_result:
            if element not in final_data:
                final_data.append(element)

        with open(target_path, "w") as f:
            json.dump(final_data, f, indent=4, ensure_ascii=False)


def format_data(data_dir, sampling_params, instruction, lora_request, target_path=None, amount_samples=None):
    
    if target_path:

        print(f"TP Load data from {target_path}")
        with open(target_path) as f:
            data = json.load(f)

        formatted_data = []
        for element in data:
            question, steps = element["instruction"].split("\n\n### Input:\n")[1].split("\n\n### Response:\n")
            formatted_data.append(
                {
                    "instruction": build_source_prompt(
                        question=question,
                        steps=steps + element["prediction"],
                        instruction=instruction
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
                    steps=element["source_steps"],
                    instruction=instruction
                ),
                "sampling_params": sampling_params,
                "lora_request": lora_request
            }
            for element in data
        ]

    # print("Sample test data:")
    # print(*([f'Instruction:\n---\n{x["instruction"]}---' for x in formatted_data][:10]), sep="\n")
    # if amount_samples:
    #     formatted_data = formatted_data[:amount_samples]
    
    return formatted_data


def filter_out_results(results, delimiter = "Final answer: "):

    print("Filter out results.")
    new_data = []
    final_result = []

    for element in results:

        obj = {
            "instruction": element["instruction"],
            "prediction": element["prediction"],
        }
        
        if delimiter in element["prediction"]:
            obj["result"] = element["prediction"].split(delimiter)[1]
            final_result.append(obj)
        else:
            new_data.append(obj)

    return new_data, final_result


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


if __name__ == "__main__":
    
    print("Getting arguments")
    args = get_inference_arguments()
    print(f"Using args:")
    print(*list(vars(args).items()), sep="\n")
    sampling_params = get_sampling_params()
    decomposition = [1, 2, 3, 4]#, [5, 6, 7, 8]]
    experiment_name = os.path.join(OUTPUT_DIR, "deepseek-7b-base-m1m2m3m4m58")
    target_path = None
    final_result = []
    data_dir = "/cluster/work/lawecon/Work/mbraasch/output/deepseek-7b-base-m1m2m3m4m58/data/test/normal/1.json"
    print(f"Using data dir {data_dir}.")
    
    for m in tqdm(decomposition):

        print(f"\n*** Start inference for model {m} ***")

        args.run_name = os.path.join(experiment_name, f'm{m[0]}-{m[-1]}' if isinstance(m, list) else f"m{m}")
        lora_request = get_lora_request(lora_path=args.run_name)
        print(f"Doing inference for model (at args.run_name) {args.run_name}.")

        instruction = M1 if m == 1 else (MA if isinstance(m, list) else MI)
        print(f"Using instruction: {instruction}")

        data = format_data(
            data_dir=data_dir,
            sampling_params=sampling_params,
            instruction=instruction,
            lora_request=lora_request,
            target_path=target_path
        )
        engine = initialize_engine(args)
        results = process_requests(engine, data)
        results, finished = filter_out_results(results)
        write_results(args, results, finished, experiment_name)
