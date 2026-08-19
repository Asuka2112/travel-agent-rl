#!/usr/bin/env bash
set -euo pipefail

hostnamectl
uname -a
nvidia-smi
nvcc --version
lscpu
free -h
df -hT .
python3 --version
git --version

