# Travel Agentic RL

Reproducible post-training pipeline for long-horizon travel tool use with
Qwen3-4B-Instruct-2507, teacher trajectory distillation, LoRA SFT, online
multi-turn GRPO, and frozen evaluation.

## Project status

| Area | Status | Evidence |
| --- | --- | --- |
| Machine audit | completed | `docs/machine_profile.md` |
| Repository skeleton | completed | source tree and CI |
| Eight-tool mock environment | completed | environment tests and smoke script |
| CUDA 12.4 stack + vLLM inference | completed | compatibility record and smoke evidence |
| Public-data acquisition and frozen split | completed | `docs/data_report.md` and manifests |
| Teacher distillation | planned | no API-backed run yet |
| SFT | planned | no checkpoint or metric claimed |
| Agentic GRPO | planned | blocked until GPUs are available |
| Base/SFT/RL evaluation | planned | frozen test set not created yet |

No training metrics or checkpoints are claimed until corresponding artifacts
exist under `outputs/` and are referenced from an experiment manifest.

## Quick start

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/base.txt -r requirements/dev.txt
python scripts/smoke_env.py
pytest
```

See `PROJECT_SPEC.md`, `TASKS.md`, and `docs/reproduction.md` for scope and
reproducible commands.
