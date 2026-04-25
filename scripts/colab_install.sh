#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${1:-/content/GEN_AI/project}"

cd "$PROJECT_DIR"

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Install a CUDA-enabled PyTorch build when Colab exposes a GPU.
python - <<'PY'
import subprocess
import sys

try:
    import torch
    has_cuda = torch.cuda.is_available()
except Exception:
    has_cuda = False

if not has_cuda:
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "torch",
        "torchvision",
        "--index-url",
        "https://download.pytorch.org/whl/cu121",
    ])
PY

python - <<'PY'
import torch
print("cuda_available=", torch.cuda.is_available())
print("device_count=", torch.cuda.device_count())
print("device_name=", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY
