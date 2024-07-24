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

ADDITIONAL_INSTRUCTION_PREVIOUS = "Use the step that is given in the \"### Input\" part when generating the solution."

ADDITIONAL_INSTRUCTION_1ST_STEP = \
"""There are up to 10 possible first steps for the problem given.
Choose and begin with the first step that is the most appropriate for solving the problem.
"""

def check_and_replace_gt(
        gt_first_step,
        pred_first_step
    ) -> str:
    
    def get_solution(first_step) -> Optional[int]:
        try:
            return int(first_step.split(">>")[1].split()[0].replace(".", ""))
        except:
            return None

    gt_first_step_solution = get_solution(gt_first_step)
    pred_first_step_solution = get_solution(pred_first_step)
    if gt_first_step_solution is None or pred_first_step_solution is None:
        return gt_first_step
    if gt_first_step_solution == pred_first_step_solution:
        print("CountMe123")
        return pred_first_step
    return gt_first_step


def build_source_prompt(
        question: str,
        gt_steps: Union[list, str],
        previous_step_n: Optional[int] = None,
        first_step_data: Optional[dict] = None,
        type_: Optional[str] = None
    ):
    
    # If steps begins with a linebreak, remove it
    if gt_steps.startswith("\n"):
        gt_steps = gt_steps[1:]

    instruction = INSTRUCTION
        
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
        keep = 9
        instruction += "\n" + ADDITIONAL_INSTRUCTION_1ST_STEP
        first_steps = []
        if "training" in type_:
            first_steps = [x[1] for x in first_step_data[question][:keep]]
            linebreak = "\n"
            gt_first_step = f'<step 1>: {gt_steps.split(linebreak)[0]}'
            if type_=="training_tf":
                gt_first_step = check_and_replace_gt(gt_first_step, first_steps[0])
            first_steps.append(gt_first_step)
        elif type_=="inference":
            first_steps = [x[1] for x in first_step_data[:keep]]
        else:
            raise ValueError("Variable `first_steps` is empty.")
        question = f"Question: {question}\n\nPossible first steps:"
        random.shuffle(first_steps)
        for first_step in first_steps:
            question += "\n" + first_step
        gt_steps = ""
    else:
        question = f"Question: {question}"

    prompt = SOURCE_TEMPLATE \
        .replace("<instruction>", instruction.strip()).lstrip() \
        .replace("<question>", question.strip()).lstrip() \
        .replace("<source_steps>", gt_steps.strip()).lstrip()
    
    if prompt.endswith("\n"):
        # Only have one line break at the end
        prompt = prompt[:-1]

    return prompt

def build_target_prompt(steps: str, result: str, eos_token: str):
    
    formatted_source_steps = ""
    split_splits = steps.split("\n")
    for i, split in enumerate(split_splits):
        formatted_source_steps += f"<step {i+1}>: {split}\n"
    steps = formatted_source_steps.strip()
    return steps + (f"\nFinal answer: {result}" if result else result) + eos_token