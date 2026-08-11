# ai-inference-runner

A GitHub Actions worker repository for running local LLM inference with `llama-cpp-python`.

## Overview

The repository provides the execution side of a serverless-style inference workflow. A GitHub Actions runner receives model and prompt configuration through environment variables, downloads or reuses a GGUF model, runs inference, and writes the response to `output.txt`.

## Features

- GitHub Actions-based inference
- GGUF model support
- TinyLlama and Llama model configurations
- Configurable prompt and system message
- Configurable token limit and temperature
- Model caching between runs

## Prerequisites

- GitHub Actions enabled for the repository
- Python environment compatible with `llama-cpp-python`
- A workflow that supplies the required model and prompt environment variables

## Quick Start

The repository is primarily intended to be invoked by its GitHub Actions workflow rather than as a standalone application.

For local execution, install the project's inference dependency and run:

```bash
python run_inference.py
```

The script expects `MODEL` and `PROMPT` environment variables.

## Configuration

| Variable | Required | Description |
|---|---|---|
| `MODEL` | Yes | Model key, such as `tinyllama` or `llama` |
| `PROMPT` | Yes | User prompt |
| `SYSTEM` | No | System prompt |
| `MAX_TOKENS` | No | Maximum generated tokens |
| `TEMPERATURE` | No | Sampling temperature |
| `N_CTX` | No | Context-window override |

## Project Structure

```text
ai-inference-runner/
├── .github/workflows/
├── run_inference.py
└── README.md
```

## Output

Inference text is printed to the workflow log and written to `output.txt` for downstream artifact handling.

## Status

Worker repository for GitHub Actions-based inference experiments.

## License

No separate license is currently specified in the repository.
