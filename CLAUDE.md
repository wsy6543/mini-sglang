# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Mini-SGLang is a compact (~5k lines of Python) LLM inference serving framework, a minimal reimplementation of [SGLang](https://github.com/sgl-project/sglang). It is fully type-annotated and stiffly typed (`mypy --strict`-style settings in `pyproject.toml`).

**Platform**: Linux only (x86_64/aarch64). The CUDA kernels (`sgl-kernel`, `flashinfer`, `quack-kernels`) are JIT-compiled and require an NVIDIA CUDA Toolkit matching the driver. macOS/Windows are not supported.

## Commands

The source lives under `python/minisgl` (SPDX-style layout: `pyproject.toml` sets `package-dir = {"" = "python"}`).

```bash
# Install (editable)
uv venv --python=3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

# Run the server (openai-compatible API on port 1919 by default)
python -m minisgl --model "Qwen/Qwen3-0.6B"
python -m minisgl --model "meta-llama/Llama-3.1-70B-Instruct" --tp 4 --port 30000

# Interactive shell mode
python -m minisgl --model "Qwen/Qwen3-0.6B" --shell

# Run tests (pytest configured via [tool.pytest.ini_options]; single test)
pytest
pytest tests/core/test_scheduler.py
pytest tests/core/test_scheduler.py::TestClassName::test_name

# Lint / format / types (also enforced by pre-commit)
ruff check .          # ruff select W,F,I,C4
black .               # line-length 100
mypy python/minisgl   # strict-ish; see [tool.mypy]

# pre-commit
pre-commit run --all-files
```

Note: tests are CPU-only and largely cover the kernel/serialization/scheduler logic, not the full CUDA inference path.

## Architecture

The system is a **multi-process distributed design**. Subprocesses communicate via ZMQ (control messages) and NCCL/`torch.distributed` (tensor data between GPUs). The process topology, started by `minisgl.server.launch_server`, is:

- **API Server** (`server/api_server.py`, FastAPI) — OpenAI-compatible frontend (`/v1/chat/completions`).
- **Tokenizer Worker** and **Detokenizer Worker** (`tokenizer/`) — text↔token conversion; shared in one process when `num_tokenizer == 0` (the default).
- **Scheduler Workers** — one per GPU ("TP rank"). Each owns an `Engine`. Rank 0 is the coordinator.

**Request lifecycle**: API server → tokenizer → scheduler(rank 0) → broadcast to all schedulers → each local `Engine` computes the next token → rank 0 collects output token → detokenizer → API server streams to user.

### Key modules (`python/minisgl`)

- `core.py` — dataclasses `Req`, `Batch`, `SamplingParams`, and `Context` (global inference state, backed by a module-level `_GLOBAL_CTX` accessed via `get_global_ctx()`).
- `engine/` — `Engine` (a TP worker on one process): owns the model, `Context`, KV cache, attention/MoE backend, sampler, and CUDA-graph replay. `graph.py` handles CUDA-graph capture/replay; `sample.py` is the sampler.
- `scheduler/` — `Scheduler` running on each TP-worker process, managing its `Engine`. `config.py` defines `SchedulerConfig`; `scheduler.py` contains scheduling logic (`decode.py`/`prefill.py`/`cache.py` handle the respective phases). Overlap scheduling hides CPU scheduling work behind GPU compute.
- `models/` — model implementations (`llama.py`, `qwen3.py`, `qwen3_moe.py`, ...) plus HF weight loading/sharding (`weight.py`). Models register via `register.py` keyed by HF `model_type`.
- `layers/` — TP-aware building blocks (linear, norm, embedding, rotary, attention, moe) sharing `layers/base.py`.
- `attention/` — attention backend interface (`base.py`) with `fa.py` (FlashAttention), `fi.py` (FlashInfer), `trtllm.py`.
- `kvcache/` — KV cache pool interface + `mha_pool.py`, and cache managers `naive_cache.py` / `radix_cache.py` (radix is default).
- `kernel/` — custom CUDA kernels bound via `tvm-ffi` (JIT). `pynccl.py` wraps NCCL for TP.
- `distributed/` — all-reduce/all-gather interface and `DistributedInfo` TP metadata.
- `message/` — ZMQ messages with automatic (de)serialization.
- `moe/` — MoE backend interface (`fused.py` Triton kernel, etc.).

### Config hierarchy

Dataclass-inheritance chain: `EngineConfig` (engine/) ← `SchedulerConfig` (scheduler/) ← `ServerArgs` (server/args.py). CLI args are parsed in `server/args.py:parse_args` (`--model`/`--tp`/`--port`/`--attn`/`--cache-type`/`--page-size`/`--cuda-graph-max-bs`/`--max-prefill-length`, etc.). `python -m minisgl --help` lists all flags.

### Important conventions

- Everything is type-annotated; `disallow_untyped_defs = true` means new code must have signatures. Config dataclasses are `frozen=True`.
- The codebase uses `from __future__ import annotations` throughout, so type hints are strings (no runtime cost).
- The `Context` global is a deliberate pattern — attention/layers read metadata (batch, page table) from `get_global_ctx()` rather than passing it through the call stack.
- `page_table` in `Context` always treats `page_size = 1` internally; the actual KV cache pool handles the real page size.
- Formatting: black (100 cols), isort via ruff, and clang-format for `kernel/csrc/**` CUDA/C++ code.