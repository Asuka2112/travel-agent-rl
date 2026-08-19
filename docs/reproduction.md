# Reproduction

## Host prerequisites

- Ubuntu 22.04 x86_64
- Python 3.10
- NVIDIA driver compatible with the selected CUDA wheel set
- Git, at least 200 GB free project storage, and explicitly allocated GPUs

Do not alter the system CUDA toolkit or driver to reproduce this project.
Python and CUDA user-space dependencies live in a project-local environment.

## Foundation environment

```bash
git clone <repository-url> travel-agent-rl
cd travel-agent-rl
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip==26.2.1
python -m pip install -r requirements/base.txt -r requirements/dev.txt
python scripts/smoke_env.py
pytest
```

## Training dependencies

The CUDA 12.4 stack is created in a separate environment:

```bash
bash scripts/setup_train_env.sh
```

`requirements/train-cu124-lock.txt` records all observed transitive versions.
Read `docs/compatibility.md` before using the stack on a different driver.

## Required secrets

Live tools and judges read provider keys from environment variables. Copy the
future `.env.example` names into a shell secret manager; never commit `.env`.
Mock mode requires no keys and is the default for tests.

## Reproducibility controls

- Global seed defaults to 20260819.
- Source datasets are recorded by repository/dataset revision and SHA-256.
- The model is recorded by Hugging Face commit revision.
- Prompts and configs are committed and referenced by content hash.
- The frozen test manifest is read-only after its creation commit.
- Training/evaluation commands produce an experiment manifest before work.
