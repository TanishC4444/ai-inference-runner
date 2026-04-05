
import os, urllib.request
from transformers import pipeline

MODEL_MAP = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama":     "bartowski/Llama-3.2-1B-Instruct",
}

model_key   = os.environ["MODEL"]
prompt      = os.environ["PROMPT"]
system      = os.environ.get("SYSTEM", "You are a helpful assistant.")
max_tokens  = int(os.environ.get("MAX_TOKENS", "512"))
temperature = float(os.environ.get("TEMPERATURE", "0.7"))

model_id = MODEL_MAP[model_key]
print(f"Loading {model_id}...")

pipe = pipeline(
    "text-generation",
    model=model_id,
    cache_dir="model_cache",
    device_map="cpu",
)

output = pipe(
    [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
    max_new_tokens=max_tokens,
    temperature=temperature,
    do_sample=temperature > 0,
)[0]["generated_text"][-1]["content"]

print("\n=== OUTPUT ===")
print(output)

with open("output.txt", "w") as f:
    f.write(output)
