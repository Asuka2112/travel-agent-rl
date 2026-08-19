# Compatibility record

Checked on 2026-08-19. Sources are linked so the assessment can be audited.

## Host constraint

The server has NVIDIA driver 550.144.03, CUDA runtime/toolkit 12.4, and RTX
4090 GPUs. The system driver must not be upgraded without explicit approval.

## Upstream facts

- Qwen3-4B-Instruct-2507 is Apache-2.0 and the model card recommends a recent
  Transformers release: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- Qwen's official training guide documents ms-swift, DeepSpeed ZeRO-3, GRPO,
  and RTX support: https://github.com/QwenLM/Qwen3/blob/main/docs/source/training/ms_swift.md
- ms-swift documents server-mode asynchronous multi-turn schedulers and custom
  environment steps: https://github.com/modelscope/ms-swift/blob/main/docs/source_en/Instruction/GRPO/DeveloperGuide/multi_turn.md
- Current ms-swift full-install guidance targets Python 3.10/3.11 and CUDA 12,
  but its newest example image uses a newer CUDA stack than this host:
  https://github.com/modelscope/ms-swift/blob/main/requirements/install_all.sh

## Validated CUDA 12.4 import stack

The pinned set in `requirements/train-cu124.txt` is PyTorch
2.6.0 cu124, Transformers 4.53.3, ms-swift 3.8.3, vLLM 0.8.5.post1,
DeepSpeed 0.16.9, and FlashAttention 2.7.4.post1.

On 2026-08-19, `pip check`, all core imports, RTX 4090 bf16 matmul,
FlashAttention import, and pinned Qwen config/tokenizer loading passed. The
model revision is `cdbee75f17c01a7cc42f958dc650907174af0554`.

Two installation details are required:

1. Pin `gradio==5.32.1`; an unconstrained current 6.x dependency caused pip
   backtracking, matching upstream guidance to keep Gradio below 5.33.
2. Build/install FlashAttention with `TMPDIR` and `PIP_CACHE_DIR` on the `/data`
   filesystem; otherwise its downloaded wheel crosses filesystems during an
   atomic rename and fails with `Invalid cross-device link`.
3. Pin FastAPI 0.115.12 and Starlette 0.46.2. The unconstrained 0.141.1/0.52.1
   pair installs successfully but is incompatible with vLLM 0.8.5's
   Prometheus route instrumentation, causing every HTTP endpoint to return 500
   after an otherwise successful model load.

The server cannot reach `huggingface.co` directly. `HF_ENDPOINT=https://hf-mirror.com`
is used only as a transport mirror while the official full commit revision is
pinned.

Full model/vLLM inference passed on an otherwise empty RTX 4090: vLLM loaded
7.6065 GiB of weights, enabled its FlashAttention backend, allocated a 68,208
token KV cache at a 32,768-token limit, and generated the requested exact text
`OK`. `scripts/vllm_smoke.py` writes repeatable machine-readable evidence.

Verified shard SHA-256 values:

- `model-00001-of-00003.safetensors`: `75311d91bb08cf0b882913da464a1e722a31fb44db35208663487efb7a3d8ed6`
- `model-00002-of-00003.safetensors`: `0b48adbb1f60e901153d91907ba11ce63bd4b8b584482e730f48808d055dfba1`
- `model-00003-of-00003.safetensors`: `7dd39ccca5e4de123c74c14af44c9bf2eb75df33b4614382af0134528e060d5d`

A minimal multi-turn GRPO parameter update remains a later M5 gate; import and
inference success do not claim that training has already run.

## External teacher provider gate

On 2026-08-19 a new Google Cloud project with billing disabled could list the
Gemini API model catalog, but every native and OpenAI-compatible generation
request returned project-level HTTP 403 `Your project has been denied access`.
The result was identical for Gemini and Gemma models, so this is an account or
project access denial rather than a model-name, quota, or client defect. No
paid tier was enabled and the blocked provider must not be represented as a
working teacher.

The distillation and judge code therefore remains provider-neutral. Mock
end-to-end validation is complete; real generation stays gated until an
authorized endpoint or an evidence-backed local teacher configuration passes a
small smoke run. Any local teacher choice must be recorded separately and does
not change the Qwen3-4B student/base-model target.

## Local teacher validation

The Apache-2.0 `Qwen/Qwen3-30B-A3B-Instruct-2507` snapshot at revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` was downloaded and served on four
otherwise idle RTX 4090 GPUs with tensor parallelism 4. vLLM loaded 14.3011 GiB
of weights per GPU, initialized a 32,768-token context with a 215,376-token KV
cache per worker, and reported 6.57-way maximum concurrency at the full context
length. A deterministic chat request returned exactly `OK`.

A subsequent real distillation smoke completed one 13-tool-turn trajectory with
49,986 cumulative input tokens and 602 output tokens across its model calls. The
trajectory passed rule filtering, canonicalization, and the structured local
judge. This validates the execution path only; it does not claim that the full
teacher dataset has been generated.

## Credentialed live-tool gate

On 2026-08-19 the production registry passed a credentialed live smoke with all
eight tools returning `ok` on their first attempt. The live calls covered AMap
POI search, nearby search, weather, and walking-route planning plus Firecrawl
search and scrape. The deterministic flight and train simulators completed the
same run. Provider credentials are stored outside the repository in a
user-owned mode-`600` environment file; no credential value is logged or
committed. No paid plan, billing account, or automatic recharge was enabled.

`ds_report` found the fused Adam and CPU Adam build paths compatible with this
toolchain. Optional `async_io` offload is currently unavailable because the
host lacks `libaio` development headers; no system package was installed or
driver changed. Standard ZeRO-3 remains testable, while an offload experiment
must either provision `libaio` explicitly or record that async I/O is disabled.
