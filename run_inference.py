
import os
import warnings
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import torch

MODEL_MAP = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama":     "meta-llama/Llama-3.2-1B-Instruct",
}

model_key   = os.environ["MODEL"]
prompt      = os.environ["PROMPT"]
system      = os.environ.get("SYSTEM", "You are a helpful assistant.")
max_tokens  = int(os.environ.get("MAX_TOKENS", "512"))
temperature = float(os.environ.get("TEMPERATURE", "0.7"))
model_id    = MODEL_MAP[model_key]

print(f"Loading {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="model_cache")
model     = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir="model_cache",
    dtype=torch.float32,
    low_cpu_mem_usage=True,
)
model.eval()

messages = [
    {"role": "system", "content": system},
    {"role": "user",   "content": prompt},
]

encoded   = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
input_ids      = encoded["input_ids"]
attention_mask = encoded["attention_mask"]
input_len      = input_ids.shape[-1]

gen_config = GenerationConfig(
    max_new_tokens=max_tokens,
    temperature=temperature if temperature > 0 else None,
    do_sample=temperature > 0,
    pad_token_id=tokenizer.eos_token_id,
)

print("Running inference...")
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        generation_config=gen_config,
    )

new_tokens = output_ids[0][input_len:]
output     = tokenizer.decode(new_tokens, skip_special_tokens=True)

print("\n=== OUTPUT ===")
print(output)

with open("output.txt", "w") as f:
    f.write(output)
