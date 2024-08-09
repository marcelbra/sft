from typing import Optional, Union
import random

EOT_TOKEN = "<|EOT|>"

SOURCE_TEMPLATE = \
"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
<instruction>

### Input:
<question>

### Response:
<source_steps>"""

INSTRUCTION = "Solve the following math word problem step-by-step."
INSTRUCTION_NEW = "Copy, create from or get inspired from the given first step suggestions given in the \"### Input\"-part. If none are good create a new first step. Pay close attention to the semantic nuances in differences between the suggestions and be specific in your wording."
ADDITIONAL_INSTRUCTION_PREVIOUS = "Use the step that is given in the \"### Input\" part when generating the solution."
FIRST_STEP_CHOSEN_TOKEN = "<first step chosen from selection>"
FIRST_STEP_GENERATE_TOKEN = "<first step generated independently>"
ADDITIONAL_INSTRUCTION_1ST_STEP = f"There are up to 10 possible first steps for the problem given. Choose and begin with the first step that is the most appropriate for solving the problem. If one of the selection is chosen, add the {FIRST_STEP_CHOSEN_TOKEN}-token to front of the first step. If no suggested first step is sufficient, generate an own one and add the {FIRST_STEP_GENERATE_TOKEN}-token to the front of the first step."


def check_and_replace_gt(
        gt_first_step,
        pred_first_step
    ) -> tuple[str, bool]:
    
    def get_solution(first_step) -> Optional[int]:
        try:
            return int(first_step.split(">>")[1].split()[0].replace(".", ""))
        except:
            return None

    gt_first_step_solution = get_solution(gt_first_step)
    pred_first_step_solution = get_solution(pred_first_step)
    if gt_first_step_solution is None or pred_first_step_solution is None:
        return gt_first_step, False
    if gt_first_step_solution == pred_first_step_solution:
        return pred_first_step, True
    return gt_first_step, False


def build_source_prompt(
        question: str,
        gt_steps: Union[list, str],
        previous_step_n: Optional[int] = None,
        first_step_data: Optional[dict] = None,
        type_: Optional[str] = None
    ) -> tuple[str, bool]:
    
    # If steps begins with a linebreak, remove it
    if gt_steps.startswith("\n"):
        print(gt_steps)
        raise ValueError("Steps began with a linebreak. There is a bug!")

    instruction = INSTRUCTION
    split_gt_steps = None
    new_gt_steps = None
    replaced = 0
    
    if previous_step_n:
        question = f"Question: {question}"
        instruction += "\n" + ADDITIONAL_INSTRUCTION_PREVIOUS
        if isinstance(gt_steps, list):
            split_step = gt_steps.split('\n')[previous_step_n-1]  # Training
        elif isinstance(gt_steps, str):
            split_step = gt_steps  # Inference
        question += "\n<step " + f"{previous_step_n}>: {split_step}"
        gt_steps = ""
    elif first_step_data:
        keep = 10
        sampled_first_steps = []
        if type_ == "new_dawn":
            if isinstance(first_step_data, dict):
                sampled_first_steps = [x[1] for x in first_step_data[question][:keep]] # Training
            if isinstance(first_step_data, list):
                sampled_first_steps = [x[1] for x in first_step_data[:keep]] # Inference
            instruction += " " + INSTRUCTION_NEW
        elif "training" in type_:
            instruction += " " + ADDITIONAL_INSTRUCTION_1ST_STEP
            sampled_first_steps = [x[1] for x in first_step_data[question][:keep]]
            split_gt_steps = gt_steps.split("\n")
            gt_first_step = f'<step 1>: {split_gt_steps[0]}'
            if type_=="training_tf":
                for pred_first_step in sampled_first_steps:
                    gt_first_step, replaced = check_and_replace_gt(gt_first_step, pred_first_step)
                    if replaced:
                        split_gt_steps[0] = gt_first_step
                        break

            if gt_first_step not in sampled_first_steps and replaced:
                sampled_first_steps.append(gt_first_step)
        elif type_=="inference":
            sampled_first_steps = [x[1] for x in first_step_data[:keep]]
        else:
            raise ValueError("CLI arg `type_` is empty.")
        question = f"Question: {question}\n\nPossible first steps:"
        sampled_first_steps = list(set(sampled_first_steps))
        random.shuffle(sampled_first_steps)
        for i, first_step in enumerate(sampled_first_steps):
            question += "\n" + first_step
        gt_steps = ""
    else:
        question = f"Question: {question}"

    if type_=="training":
        gt_steps = ""

    prompt = SOURCE_TEMPLATE \
        .replace("<instruction>", instruction.strip()).lstrip() \
        .replace("<question>", question.strip()).lstrip() \
        .replace("<source_steps>", gt_steps.strip()).lstrip()
    
    # if prompt.endswith("\n"): # !!!! ACHTUNG !!!!
    if prompt.endswith("\n\n"): # !!!! ACHTUNG !!!!
        # Only have one line break at the end
        prompt = prompt[:-1]

    if split_gt_steps is not None:
        new_gt_steps = "\n".join(split_gt_steps).replace("<step 1>: ", "")
    
    return prompt, replaced, new_gt_steps

def build_target_prompt(
        steps: str,
        result: str,
        eos_token: str,
        replace: bool,
        type_: str
    ) -> str:
    
    formatted_source_steps = (
        FIRST_STEP_CHOSEN_TOKEN
        if replace else
        FIRST_STEP_GENERATE_TOKEN
    ) if type_=="training_tf" else ""

    step_splits = steps.split("\n")
    for i, step in enumerate(step_splits):
        formatted_source_steps += f"<step {i+1}>: {step}\n"
    steps = formatted_source_steps.strip()
    return steps + (f"\n<final answer>: {result}" if result else result) + eos_token