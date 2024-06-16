EOT_TOKEN = "<|EOT|>"

SOURCE_TEMPLATE = \
"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
Solve the following math word problem step-by-step.

### Input:
<question>

### Response:
<source_steps>"""

def build_source_prompt(question: str, steps: str):
    prompt = SOURCE_TEMPLATE \
        .replace("<question>", question.strip()).lstrip() \
        .replace("<source_steps>", steps.strip()).lstrip()
    if prompt.endswith("\n\n"):
        # Only have one line break at the end
        prompt = prompt[:-1]
    return prompt


def build_target_prompt(steps: str, result: str, eos_token: str):
    return steps + (f"\nFinal answer: {result}" if result else result) + eos_token