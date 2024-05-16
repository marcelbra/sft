EOT_TOKEN = "<|EOT|>"
SOURCE_TEMPLATE = \
"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
<instruction>

### Input:
<input>

### Response:

"""

def build_source_prompt(formatting_template: str, question: str, model_type: str):
    if model_type == "bl":
        instruction = "Solve the given math problem step by step and put your final answer after 'Final answer: '."
    elif model_type == "m1":
        instruction = "Solve the given math problem step by step but only conduct the first step."
    elif model_type == "mi":
        instruction = "Solve the given math problem step by step but only solve the next step. If you have reached the final answer, put your final answer after 'Final answer: '."
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    return SOURCE_TEMPLATE \
        .replace("<instruction>", instruction).lstrip() \
        .replace("<input>", question.strip()).lstrip()


def build_target_prompt(steps: str, result: str):
    return steps + f"\nFinal answer: {result}" if result else result + EOT_TOKEN