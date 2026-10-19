
import hashlib
import json
import os
import platform
import resource
import shutil
import time
import urllib.error
import urllib.request
import warnings

warnings.filterwarnings("ignore")

MODEL_MAP = {'qwen0.5b': {'name': 'Qwen 2.5 0.5B Instruct', 'parameters_b': 0.5, 'quantization': 'Q4_K_M', 'size_bytes': 491400032, 'size_gb': 0.491400032, 'sha256': '74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db', 'n_ctx': 8192, 'max_n_ctx': 32768, 'context_source': 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct', 'max_output_tokens': 8192, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.125, 'url': 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf', 'filename': 'qwen2.5-0.5b-instruct-q4_k_m.gguf'}, 'tinyllama': {'name': 'TinyLlama 1.1B Chat', 'parameters_b': 1.1, 'quantization': 'Q4_K_M', 'size_bytes': 668788096, 'size_gb': 0.668788096, 'sha256': '9fecc3b3cd76bba89d504f29b616eedf7da85b96540e490ca5824d3f7d2776a0', 'n_ctx': 2048, 'max_n_ctx': 2048, 'context_source': 'https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/blob/main/config.json', 'max_output_tokens': None, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf', 'filename': 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf'}, 'llama': {'name': 'Llama 3.2 1B Instruct', 'parameters_b': 1, 'quantization': 'Q4_K_M', 'size_bytes': 807694464, 'size_gb': 0.807694464, 'sha256': '6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83', 'n_ctx': 8192, 'max_n_ctx': 131072, 'context_source': 'https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct', 'max_output_tokens': None, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf', 'filename': 'Llama-3.2-1B-Instruct-Q4_K_M.gguf'}, 'phi3': {'name': 'Phi-3.5 Mini Instruct', 'parameters_b': 3.8, 'quantization': 'Q4_K_M', 'size_bytes': 2393232672, 'size_gb': 2.393232672, 'sha256': 'e4165e3a71af97f1b4820da61079826d8752a2088e313af0c7d346796c38eff5', 'n_ctx': 8192, 'max_n_ctx': 131072, 'context_source': 'https://huggingface.co/microsoft/Phi-3.5-mini-instruct', 'max_output_tokens': None, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.4, 'url': 'https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf', 'filename': 'Phi-3.5-mini-instruct-Q4_K_M.gguf'}, 'qwen': {'name': 'Qwen 2.5 1.5B Instruct', 'parameters_b': 1.5, 'quantization': 'Q4_K_M', 'size_bytes': 1117320736, 'size_gb': 1.117320736, 'sha256': '6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e', 'n_ctx': 8192, 'max_n_ctx': 32768, 'context_source': 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct', 'max_output_tokens': 8192, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf', 'filename': 'qwen2.5-1.5b-instruct-q4_k_m.gguf'}, 'gemma2': {'name': 'Gemma 2 2B Instruct', 'parameters_b': 2, 'quantization': 'Q4_K_M', 'size_bytes': 1708582752, 'size_gb': 1.708582752, 'sha256': 'e0aee85060f168f0f2d8473d7ea41ce2f3230c1bc1374847505ea599288a7787', 'n_ctx': 8192, 'max_n_ctx': 8192, 'context_source': 'https://ai.google.dev/gemma/docs/core/model_card_2', 'max_output_tokens': None, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf', 'filename': 'gemma-2-2b-it-Q4_K_M.gguf'}, 'deepseek': {'name': 'DeepSeek R1 Distill Qwen 1.5B', 'parameters_b': 1.5, 'quantization': 'Q4_K_M', 'size_bytes': 1117320800, 'size_gb': 1.1173208, 'sha256': '1741e5b2d062b07acf048bf0d2c514dadf2a48f94e2b4aa0cfe069af3838ee2f', 'n_ctx': 8192, 'max_n_ctx': 131072, 'context_source': 'https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B/blob/main/config.json', 'max_output_tokens': None, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf', 'filename': 'DeepSeek-R1-Distill-Qwen-1.5B-Q4_K_M.gguf'}, 'qwen3b': {'name': 'Qwen 2.5 3B Instruct', 'parameters_b': 3, 'quantization': 'Q4_K_M', 'size_bytes': 1929903264, 'size_gb': 1.929903264, 'sha256': '9c9f56a391a3abbd5b89d0245bf6106081bcc3173119d4229235dd9d23253f94', 'n_ctx': 8192, 'max_n_ctx': 32768, 'context_source': 'https://huggingface.co/Qwen/Qwen2.5-3B-Instruct', 'max_output_tokens': 8192, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf', 'filename': 'Qwen2.5-3B-Instruct-Q4_K_M.gguf'}, 'qwen7b': {'name': 'Qwen 2.5 7B Instruct', 'parameters_b': 7, 'quantization': 'Q4_K_M', 'size_bytes': 4683074240, 'size_gb': 4.68307424, 'sha256': '65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423', 'n_ctx': 8192, 'max_n_ctx': 131072, 'context_source': 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct', 'max_output_tokens': 8192, 'yarn_original_context': 32768, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf', 'filename': 'Qwen2.5-7B-Instruct-Q4_K_M.gguf'}, 'qwen-coder7b': {'name': 'Qwen 2.5 Coder 7B Instruct', 'parameters_b': 7, 'quantization': 'Q4_K_M', 'size_bytes': 4683074336, 'size_gb': 4.683074336, 'sha256': '1664fccab734674a50763490a8c6931b70e3f2f8ec10031b54806d30e5f956b6', 'n_ctx': 8192, 'max_n_ctx': 131072, 'context_source': 'https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct', 'max_output_tokens': None, 'yarn_original_context': 32768, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf', 'filename': 'Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf'}, 'llama8b': {'name': 'Llama 3.1 8B Instruct', 'parameters_b': 8, 'quantization': 'Q4_K_M', 'size_bytes': 4920739232, 'size_gb': 4.920739232, 'sha256': '7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c', 'n_ctx': 8192, 'max_n_ctx': 131072, 'context_source': 'https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct', 'max_output_tokens': None, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf', 'filename': 'Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf'}, 'gemma9b': {'name': 'Gemma 2 9B Instruct', 'parameters_b': 9, 'quantization': 'Q4_K_M', 'size_bytes': 5761057728, 'size_gb': 5.761057728, 'sha256': '13b2a7b4115bbd0900162edcebe476da1ba1fc24e718e8b40d32f6e300f56dfe', 'n_ctx': 8192, 'max_n_ctx': 8192, 'context_source': 'https://ai.google.dev/gemma/docs/core/model_card_2', 'max_output_tokens': None, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf', 'filename': 'gemma-2-9b-it-Q4_K_M.gguf'}, 'mistral12b': {'name': 'Mistral Nemo 12B Instruct', 'parameters_b': 12, 'quantization': 'Q4_K_M', 'size_bytes': 7477208192, 'size_gb': 7.477208192, 'sha256': '7c1a10d202d8788dbe5628dc962254d10654c853cae6aaeca0618f05490d4a46', 'n_ctx': 8192, 'max_n_ctx': 131072, 'context_source': 'https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407', 'max_output_tokens': None, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF/resolve/main/Mistral-Nemo-Instruct-2407-Q4_K_M.gguf', 'filename': 'Mistral-Nemo-Instruct-2407-Q4_K_M.gguf'}, 'qwen14b': {'name': 'Qwen 2.5 14B Instruct', 'parameters_b': 14, 'quantization': 'Q4_K_M', 'size_bytes': 8988110976, 'size_gb': 8.988110976, 'sha256': 'e47ad95dad6ff848b431053b375adb5d39321290ea2c638682577dafca87c008', 'n_ctx': 8192, 'max_n_ctx': 131072, 'context_source': 'https://huggingface.co/Qwen/Qwen2.5-14B-Instruct', 'max_output_tokens': 8192, 'yarn_original_context': 32768, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf', 'filename': 'Qwen2.5-14B-Instruct-Q4_K_M.gguf'}, 'qwen-coder14b': {'name': 'Qwen 2.5 Coder 14B Instruct', 'parameters_b': 14, 'quantization': 'Q4_K_M', 'size_bytes': 8988111072, 'size_gb': 8.988111072, 'sha256': '2946d28c9e1bb2bcae6d42e8678863a31775df6f740315c7d7e6d6b6411f5937', 'n_ctx': 8192, 'max_n_ctx': 131072, 'context_source': 'https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct', 'max_output_tokens': None, 'yarn_original_context': 32768, 'ram_per_1k_ctx_gb': 0.25, 'url': 'https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf', 'filename': 'Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf'}, 'deepseek-coder16b': {'name': 'DeepSeek Coder V2 Lite Instruct', 'parameters_b': 16, 'quantization': 'Q4_K_M', 'size_bytes': 10364416768, 'size_gb': 10.364416768, 'sha256': '603bd3f8a0281d16571da7c08bd661ee17ff0d1be6fcbd1b42242da257ef0bb8', 'n_ctx': 8192, 'max_n_ctx': 131072, 'context_source': 'https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct', 'max_output_tokens': None, 'yarn_original_context': None, 'ram_per_1k_ctx_gb': 0.5, 'url': 'https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf', 'filename': 'DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf'}}
_LOADED_MODELS = {}
_MODEL_STATS = {}
RUNTIME_OVERHEAD_GB = 1.5
DISK_RESERVE_GB = 2.0
DEFAULT_LIMITS = {'enforce_model_limits': True, 'enforce_output_budget': True, 'check_resources': True, 'max_prompt_chars': 60000, 'max_workflow_input_chars': 65000, 'max_prompts': 32, 'min_temperature': 0.0, 'max_temperature': 2.0, 'context_reserve_tokens': 128, 'message_overhead_tokens': 32, 'max_cached_model_gb': 6.0, 'runtime_overhead_gb': 1.5, 'disk_reserve_gb': 2.0, 'ram_per_1k_ctx_gb': None, 'job_timeout_minutes': 30, 'artifact_retention_days': 3, 'download_timeout_seconds': 30, 'provider_timeout_seconds': 300, 'refiner_min_tokens': 128, 'refiner_max_tokens': 512, 'refiner_context_fraction': 0.25}


def memory_info():
    values = {}
    try:
        with open("/proc/meminfo", encoding="utf-8") as stream:
            for line in stream:
                key, value = line.split(":", 1)
                values[key] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return values


def resource_snapshot(started):
    memory = memory_info()
    disk = shutil.disk_usage(".")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() != "Darwin":
        peak *= 1024
    return {
        "cpu_count": os.cpu_count() or 1,
        "memory_total_bytes": memory.get("MemTotal", 0),
        "memory_available_bytes": memory.get("MemAvailable", 0),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "peak_memory_bytes": peak,
        "duration_seconds": round(time.monotonic() - started, 3),
        "runner_os": platform.platform(),
        "cpu_seconds": time.process_time() - cpu_started,
    }


def provider_inference(messages, max_tokens, temperature, model, base_url):
    base_url = base_url.rstrip("/")
    api_key = os.environ.get("PROVIDER_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GitHub Actions secret 'GH_AI_PROVIDER_API_KEY' is missing"
        )
    payload = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": max_tokens, "temperature": temperature,
        **config.get("generation_options", {}),
    }).encode()
    request = urllib.request.Request(
        base_url + "/chat/completions", data=payload,
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=config.get("limits", {}).get("provider_timeout_seconds", 300)) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"Provider returned HTTP {error.code}: {detail}") from error


def model_config(local_model):
    return config.get("model_spec") or MODEL_MAP[local_model]


def get_local_model(local_model, requested_n_ctx):
    from llama_cpp import Llama

    cfg = model_config(local_model)
    policy = dict(DEFAULT_LIMITS)
    policy.update(config.get("limits", {}))
    n_ctx = int(requested_n_ctx) if requested_n_ctx else cfg["n_ctx"]
    if n_ctx < 1 or (policy["enforce_model_limits"] and cfg["max_n_ctx"] and n_ctx > cfg["max_n_ctx"]):
        raise ValueError(f"{local_model} supports context up to {cfg['max_n_ctx']:,} tokens")
    key = (local_model, n_ctx)
    if key in _LOADED_MODELS:
        return _LOADED_MODELS[key]
    cache_dir = os.path.join("model_cache", local_model) if "files" in cfg else "model_cache"
    files = cfg.get("files", [cfg])
    paths = [os.path.join(cache_dir, item["filename"]) for item in files]
    model_path = paths[0]
    required_ram = cfg["size_gb"] + policy["runtime_overhead_gb"] + n_ctx / 1024 * (
        cfg["ram_per_1k_ctx_gb"] if policy["ram_per_1k_ctx_gb"] is None else policy["ram_per_1k_ctx_gb"])
    available_ram = memory_info().get("MemAvailable")
    if policy["check_resources"] and available_ram and required_ram * 1e9 > available_ram:
        raise RuntimeError(
            f"Model/context needs approximately {required_ram:.1f} GB RAM, "
            f"but only {available_ram / 1e9:.1f} GB is available. Use a smaller model/context or a larger runner."
        )
    # Cached files already occupy disk; cache upload can need another full copy.
    required_disk = (
        sum(item["size_bytes"] for item, path in zip(files, paths) if not os.path.exists(path)) / 1e9
        + cfg["size_gb"] * int(config.get("cache_weights", False))
        + policy["disk_reserve_gb"]
    )
    if policy["check_resources"] and required_disk * 1e9 > shutil.disk_usage(".").free:
        raise RuntimeError(f"Model/cache needs approximately {required_disk:.1f} GB additional free disk")
    cached_weights = all(os.path.exists(path) for path in paths)
    download_started = time.monotonic()
    for file_cfg, file_path in zip(files, paths):
        if os.path.exists(file_path):
            continue
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        partial_path = file_path + ".part"
        try:
            with urllib.request.urlopen(file_cfg["url"], timeout=policy["download_timeout_seconds"]) as source:
                with open(partial_path, "wb") as target:
                    digest = hashlib.sha256()
                    size = 0
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        target.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
            if size != file_cfg["size_bytes"] or digest.hexdigest() != file_cfg["sha256"]:
                raise RuntimeError("Downloaded model size/checksum does not match the catalog")
            os.replace(partial_path, file_path)
        finally:
            if os.path.exists(partial_path):
                os.remove(partial_path)
    cpu_count = os.cpu_count() or 1
    threads = max(1, cpu_count - 1) if cpu_count > 2 else cpu_count
    download_seconds = time.monotonic() - download_started
    load_started = time.monotonic()
    context_options = {}
    original_context = cfg.get("yarn_original_context")
    if original_context and n_ctx > original_context:
        from llama_cpp import LLAMA_ROPE_SCALING_TYPE_YARN
        # Qwen's published long-context recipe uses static factor-4 YaRN.
        # Leave short contexts and other models' GGUF scaling metadata alone.
        context_options = {
            "rope_scaling_type": LLAMA_ROPE_SCALING_TYPE_YARN,
            "rope_freq_scale": original_context / cfg["max_n_ctx"],
            "yarn_orig_ctx": original_context,
            "yarn_ext_factor": 1.0,
        }
    options = {"n_threads": threads, "verbose": False, **context_options,
               **config.get("model_options", {})}
    if isinstance(options.get("rope_scaling_type"), str):
        import llama_cpp
        options["rope_scaling_type"] = getattr(
            llama_cpp, "LLAMA_ROPE_SCALING_TYPE_" + options["rope_scaling_type"].upper())
    llm = Llama(model_path=model_path, n_ctx=n_ctx,
                n_gpu_layers=0, use_mmap=False, use_mlock=True, **options)
    _LOADED_MODELS[key] = llm
    _MODEL_STATS[key] = {
        "weights_cached": cached_weights,
        "download_seconds": download_seconds,
        "load_seconds": time.monotonic() - load_started,
    }
    return llm


def local_inference(messages, max_tokens, temperature, local_model, requested_n_ctx):
    llm = get_local_model(local_model, requested_n_ctx)
    if max_tokens is None and config.get("limits", {}).get("enforce_model_limits", True):
        max_tokens = model_config(local_model).get("max_output_tokens")
    return llm.create_chat_completion(
        messages=messages, max_tokens=max_tokens, temperature=temperature,
        **config.get("generation_options", {})
    )


import re
from functools import lru_cache


def run_agent(agent, initial_messages, inference_config):
    started = time.monotonic()
    trim_history = agent.get("trim_history", True)
    if type(trim_history) is not bool:
        raise ValueError("trim_history must be a bool")
    policy = dict(DEFAULT_LIMITS)
    policy.update(inference_config.get("limits", {}))
    local_model = os.environ.get("LOCAL_MODEL", "qwen0.5b")
    context = inference_config.get("n_ctx") or model_config(local_model)["n_ctx"]
    maximum = inference_config.get("max_tokens", 512)
    temperature = float(inference_config.get("temperature", .7))
    key = (local_model, context)
    was_loaded = key in _LOADED_MODELS
    llm = get_local_model(local_model, context)
    stats = _MODEL_STATS[key]
    tasks = [initial_messages[-1]["content"]] + agent.get("tasks", [])
    if (not tasks or (policy["max_prompts"] is not None and len(tasks) > policy["max_prompts"])
            or any(not isinstance(task, str) or not task.strip() for task in tasks)):
        raise ValueError("Agent prompts are empty, invalid, or exceed configured max_prompts")
    max_turns = agent.get("max_history_turns", 6)
    if max_turns is not None and (type(max_turns) is not int or max_turns < 1):
        raise ValueError("max_history_turns must be a positive integer or None")
    history = list(agent.get("history", [])) if agent.get("remember", True) else []
    if len(history) % 2 or any(
        not isinstance(item, dict) or item.get("role") != ("user" if i % 2 == 0 else "assistant")
        or not isinstance(item.get("content"), str) for i, item in enumerate(history)
    ):
        raise ValueError("Invalid conversation history")
    if trim_history and max_turns is not None:
        history = history[-2 * max_turns:]
    system = initial_messages[0]
    @lru_cache(maxsize=256)
    def count(text):
        return len(llm.tokenize(text.encode(), add_bos=False))
    def fits(messages, output):
        return (policy["context_reserve_tokens"] + sum(count(item["content"]) +
                policy["message_overhead_tokens"] for item in messages) +
                ((output or 0) if policy["enforce_output_budget"] else 0) <= context)
    def generate(messages, output=maximum, temp=temperature, check_context=True, use_options=True):
        if check_context and not fits(messages, output):
            raise ValueError("Prompt/instructions exceed context; shorten input or increase n_ctx")
        # Let llama.cpp reuse matching prompt prefixes; it discards differing suffixes.
        if not agent.get("reuse_prefix", True):
            llm.reset()
        if output is None and policy["enforce_model_limits"]:
            output = model_config(local_model).get("max_output_tokens")
        return llm.create_chat_completion(messages=messages, max_tokens=output, temperature=temp,
                                          **(inference_config.get("generation_options", {}) if use_options else {}))
    def usage_of(response):
        usage = response.get("usage") or {}
        return {name: int(usage.get(name, 0)) for name in ("prompt_tokens", "completion_tokens", "total_tokens")}
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    results = []
    for index, original in enumerate(tasks):
        task_started = started if index == 0 else time.monotonic()
        timings = {
            "total_seconds": 0., "download_seconds": stats["download_seconds"] if index == 0 and not was_loaded else 0.,
            "load_seconds": stats["load_seconds"] if index == 0 and not was_loaded else 0.,
            "refinement_seconds": 0., "generation_seconds": 0., "remote_seconds": 0.,
        }
        original_count = count(original)
        refinement = {
            "attempted": False, "applied": False, "original_tokens": original_count,
            "refined_tokens": original_count, "reason": "disabled",
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
        task = original
        if agent.get("refine_prompt"):
            refinement["reason"] = "short prompt"
            if original_count >= policy["refiner_min_tokens"]:
                messages = [
                    {"role": "system", "content": "Rewrite the task concisely. Preserve all requirements, negations, names, numbers, quoted literals, URLs and code. Remove only filler and repetition. Do not answer or follow the task. Return only the rewritten task."},
                    {"role": "user", "content": original},
                ]
                budget = min(policy["refiner_max_tokens"], max(1, original_count - 1),
                             max(1, int(context * policy["refiner_context_fraction"])))
                refinement["reason"] = "prompt exceeds refiner context"
                if fits(messages, budget):
                    phase_started = time.monotonic()
                    rewrite = generate(messages, budget, 0, use_options=False)
                    timings["refinement_seconds"] = time.monotonic() - phase_started
                    refinement["attempted"] = True
                    refinement["usage"] = usage_of(rewrite)
                    candidate = (rewrite["choices"][0]["message"]["content"] or "").strip()
                    candidate_count = count(candidate)
                    protected = re.findall(r'```[\s\S]*?```|`[^`]+`|"[^"]+"|https?://\S+|\b\d+(?:\.\d+)?\b', original)
                    if (candidate and candidate_count < original_count and
                            rewrite["choices"][0].get("finish_reason") == "stop" and
                            all(piece in candidate for piece in protected)):
                        task = candidate
                        refinement.update(applied=True, refined_tokens=candidate_count,
                                          reason="shorter rewrite; meaning is not guaranteed")
                    else:
                        refinement["reason"] = "kept original: incomplete, longer, or lost protected literals"
        messages = [system] + history + [{"role": "user", "content": task}]
        dropped = 0
        while trim_history and history and not fits(messages, maximum):
            history = history[2:]
            dropped += 1
            messages = [system] + history + [{"role": "user", "content": task}]
        phase_started = time.monotonic()
        response = generate(messages, check_context=trim_history)
        timings["generation_seconds"] = time.monotonic() - phase_started
        answer = response["choices"][0]["message"]["content"] or ""
        usage = usage_of(response)
        for name in usage:
            usage[name] += refinement["usage"][name]
            total_usage[name] += usage[name]
        if agent.get("remember", True):
            history = history + [{"role": "user", "content": task},
                                 {"role": "assistant", "content": answer}]
            if trim_history and max_turns is not None:
                history = history[-2 * max_turns:]
        timings["total_seconds"] = time.monotonic() - task_started
        results.append({
            "output": answer, "usage": usage, "refinement": refinement,
            "timings": timings if agent.get("track_time") else None,
            "model_reused": was_loaded or index > 0, "weights_cached": stats["weights_cached"],
            "dropped_history_turns": dropped,
            "finish_reason": response["choices"][0].get("finish_reason", "stop"),
        })
    return {"results": results, "usage": total_usage, "history": history}


def run_discussion(agent, initial_messages, inference_config):
    options = agent["discussion"]
    count, rounds, peer_chars = (options.get(name) for name in ("agents", "rounds", "peer_chars"))
    if any(type(value) is not int or value < 1 for value in (count, rounds, peer_chars)):
        raise ValueError("Discussion counts must be positive integers")
    topology = options.get("topology", "all")
    if topology not in ("all", "ring"):
        raise ValueError("topology must be all or ring")
    policy = dict(DEFAULT_LIMITS)
    policy.update(inference_config.get("limits", {}))
    if policy["max_prompts"] is not None and count * rounds > policy["max_prompts"]:
        raise ValueError("Discussion exceeds configured max_prompts")
    task = initial_messages[-1]["content"]
    turn_config = dict(agent, tasks=[], remember=False, history=[])
    turn_config.pop("discussion")
    results, previous = [], []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for round_index in range(rounds):
        current = []
        for index in range(count):
            peers = ([previous[(index - 1) % count]] if topology == "ring" and previous else previous)
            shared = "\n".join("Agent %d: %s" % (peer, reply[:peer_chars]) for peer, reply in peers)
            prompt = ("Task: %s\nPrevious round replies:\n%s\n"
                      "You are Agent %d, round %d. Propose a distinct improvement and test. "
                      "Address another agent by number when replies are provided."
                      % (task, shared, index, round_index + 1))
            if policy["max_prompt_chars"] is not None and len(prompt) + len(initial_messages[0]["content"]) > policy["max_prompt_chars"]:
                raise ValueError("Discussion prompt exceeds configured max_prompt_chars")
            messages = [initial_messages[0], {"role": "user", "content": prompt}]
            response = run_agent(turn_config, messages, inference_config)
            item = response["results"][0]
            results.append(item)
            current.append((index, item["output"]))
            for name in usage:
                usage[name] += response["usage"][name]
        previous = current
    return {"results": results, "usage": usage, "history": []}

started = time.monotonic()
cpu_started = time.process_time()
job_id = os.environ["JOB_ID"]
attempt = int(os.environ.get("ATTEMPT", "1"))
backend = os.environ.get("BACKEND", "local")
config = json.loads(os.environ.get("CONFIG_JSON", "{}"))
provider_name = config.get("provider_name", "local")
model = config["model"]
messages = [
    {"role": "system", "content": os.environ.get("SYSTEM", "You are a helpful assistant.")},
    {"role": "user", "content": os.environ["PROMPT"]},
]
max_tokens = config.get("max_tokens", 512)
temperature = float(config.get("temperature", 0.7))
request_document = {
    "prompt": os.environ["PROMPT"],
    "system": os.environ.get("SYSTEM", "You are a helpful assistant."),
    "backend": backend,
    "model": model,
    "local_model": os.environ.get("LOCAL_MODEL", "tinyllama"),
    "cache": os.environ.get("CACHE", "true") == "true",
    "max_tokens": max_tokens,
    "temperature": temperature,
    "n_ctx": config.get("n_ctx"),
    "provider_name": provider_name,
    "provider_url": config.get("provider_url", ""),
}
if "agent" in config:
    request_document["agent"] = config["agent"]
for key in ("limits", "model_options", "generation_options", "model_spec"):
    if key in config:
        request_document[key] = config[key]
request_sha256 = hashlib.sha256(
    json.dumps(request_document, sort_keys=True, separators=(",", ":")).encode()
).hexdigest()

result = {
    "schema_version": 1, "job_id": job_id, "attempt": attempt,
    "correlation_id": f"{job_id}-{attempt}", "model": model,
    "provider": provider_name if backend == "provider" else "local",
    "output": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    "resources": {},
    "integrity": {
        "algorithm": "sha256", "request_sha256": request_sha256, "output_sha256": "",
    },
}

try:
    if "agent" in config:
        result["agent"] = (run_discussion if "discussion" in config["agent"] else run_agent)(
            config["agent"], messages, config)
        result["output"] = result["agent"]["results"][-1]["output"]
        result["usage"] = result["agent"]["usage"]
    else:
        response = (
        provider_inference(
            messages, max_tokens, temperature, model, config.get("provider_url", "")
        )
        if backend == "provider"
        else local_inference(
            messages, max_tokens, temperature,
            os.environ.get("LOCAL_MODEL", "tinyllama"), config.get("n_ctx"),
        )
    )
        result["output"] = response["choices"][0]["message"]["content"] or ""
        usage = response.get("usage") or {}
        result["usage"] = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "total_tokens": int(usage.get("total_tokens", 0)),
        }
except Exception as error:
    result["error"] = {"type": type(error).__name__, "message": str(error)}
    raise
finally:
    if "agent" in result:
        result["integrity"]["agent_sha256"] = hashlib.sha256(
            json.dumps(result["agent"], sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    result["integrity"]["output_sha256"] = hashlib.sha256(result["output"].encode()).hexdigest()
    result["resources"] = resource_snapshot(started)
    with open("result.json", "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
