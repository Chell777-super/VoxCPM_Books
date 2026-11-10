#!/usr/bin/env bash
# VoxCPM Books — установка на Linux (uv + PyTorch CUDA)
set -euo pipefail
cd "$(dirname "$0")"
PY_VER="${PY_VER:-3.11}"
TORCH_VER="${TORCH_VER:-2.10.0}"
TORCH_CUDA="${TORCH_CUDA:-cu126}"
export UV_LINK_MODE=copy HF_HOME="$PWD/.cache/huggingface"

command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
[ -x .venv/bin/python ] || uv venv .venv --python "$PY_VER"
PY="$PWD/.venv/bin/python"
uv pip install --python "$PY" "torch==$TORCH_VER" "torchaudio==$TORCH_VER" --index-url "https://download.pytorch.org/whl/$TORCH_CUDA" \
  || uv pip install --python "$PY" torch torchaudio --index-url "https://download.pytorch.org/whl/$TORCH_CUDA"
mkdir -p .cache
"$PY" installer/setup_helper.py constraints .cache/constraints.txt
uv pip install --python "$PY" -r installer/requirements-books.txt -c .cache/constraints.txt
"$PY" installer/setup_helper.py download models/VoxCPM2
"$PY" installer/setup_helper.py check || true
echo "Готово. Запуск: ./run.sh"
