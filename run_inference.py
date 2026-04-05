
import os, urllib.request
from llama_cpp import Llama

MODEL_MAP = {
    "tinyllama": {
        "url":      "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
        "n_ctx":    2048,
    },
    "llama": {
        "url":      "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "filename": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "n_ctx":    4096,
    },
}

model_key   = os.environ["MODEL"]
prompt      = os.environ["PROMPT"]
system      = os.environ.get("SYSTEM", "You are a helpful assistant.")
max_tokens  = int(os.environ.get("MAX_TOKENS", "512"))
temperature = float(os.environ.get("TEMPERATURE", "0.7"))
n_ctx_env = os.environ.get("N_CTX", "").strip()
n_ctx     = int(n_ctx_env) if n_ctx_env else MODEL_MAP[model_key]["n_ctx"]

cfg        = MODEL_MAP[model_key]
model_path = f"model_cache/{cfg['filename']}"

if not os.path.exists(model_path):
    os.makedirs("model_cache", exist_ok=True)
    print(f"Downloading {cfg['filename']}...")
    urllib.request.urlretrieve(cfg["url"], model_path)
    print("Download complete.")
else:
    print(f"Using cached {cfg['filename']}")

print("Loading model...")
llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=2, verbose=False)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": system},
        {"role": "user",   "content": prompt},
    ],
    max_tokens=max_tokens,
    temperature=temperature,
)

output = response["choices"][0]["message"]["content"]
print("\n=== OUTPUT ===")
print(output)

with open("output.txt", "w") as f:
    f.write(output)
