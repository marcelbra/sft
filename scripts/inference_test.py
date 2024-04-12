import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

device = "cuda"
cache_dir = "/cluster/work/lawecon/Work/mbraasch/models"
model_name_or_path = "deepseek-ai/deepseek-llm-7b-chat"
adapter_model_name = "/cluster/work/lawecon/Work/mbraasch/projects/moe_decomposition/output/deepseek-llm-7b-chat-baseline/checkpoint-2330"

messages = [
    {
        "role": "user",
        "content": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    }
]

model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    torch_dtype=torch.bfloat16,
    cache_dir=cache_dir, 
    device_map="auto"
)
model = PeftModel.from_pretrained(model, adapter_model_name)

model.generation_config = GenerationConfig.from_pretrained(model_name_or_path)
model.generation_config.do_sample = False
model.generation_config.num_beams = 1
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.pad_token_id = model.generation_config.eos_token_id
print(model.generation_config)

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

input_tensor = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
outputs = model.generate(input_tensor.to(model.device), max_new_tokens=1000)

result = tokenizer.decode(outputs[0][input_tensor.shape[1]:], skip_special_tokens=True)
print(result)