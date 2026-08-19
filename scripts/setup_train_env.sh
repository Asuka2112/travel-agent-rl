#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV_DIR="${ROOT_DIR}/.venv-train"
TMP_DIR="${ROOT_DIR}/.tmp"
CACHE_DIR="${ROOT_DIR}/.pip-cache"

mkdir -p "${TMP_DIR}" "${CACHE_DIR}"
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip==26.2.1
"${VENV_DIR}/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0
"${VENV_DIR}/bin/python" -m pip install \
  wheel==0.45.1 \
  gradio==5.32.1 \
  -r "${ROOT_DIR}/requirements/base.txt" \
  -r "${ROOT_DIR}/requirements/dev.txt" \
  transformers==4.53.3 \
  ms-swift==3.8.3 \
  vllm==0.8.5.post1 \
  deepspeed==0.16.9
TMPDIR="${TMP_DIR}" PIP_CACHE_DIR="${CACHE_DIR}" MAX_JOBS=16 \
  "${VENV_DIR}/bin/python" -m pip install \
  flash-attn==2.7.4.post1 --no-build-isolation
"${VENV_DIR}/bin/python" -m pip check
