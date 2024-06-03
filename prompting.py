EOT_TOKEN = "<|EOT|>"

SOURCE_TEMPLATE = \
"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
<instruction>

### Input:
<question>

### Response:
<source_steps>"""

M1 = "Solve the given math problem step by step but only conduct the first step."
MI = "Solve the given math problem step by step but only solve the next step. If you have reached the final answer, put your final answer after 'Final answer: '."
MA = "Solve the given math problem step by step."

def build_source_prompt(question: str, steps: str, instruction: str):
    prompt = SOURCE_TEMPLATE \
        .replace("<instruction>", instruction).lstrip() \
        .replace("<question>", question.strip()).lstrip() \
        .replace("<source_steps>", steps.strip()).lstrip()
    if prompt.endswith("\n\n"):
        # Only have one line break at the end
        prompt = prompt[:-1]
    return prompt


def build_target_prompt(steps: str, result: str, eos_token: str):
    return steps + (f"\n#### {result}" if result else result) + eos_token