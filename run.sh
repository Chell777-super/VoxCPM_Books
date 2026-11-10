#!/usr/bin/env bash
cd "$(dirname "$0")"
export PYTHONUTF8=1 HF_HOME="$PWD/.cache/huggingface" TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor" \
  TRITON_CACHE_DIR="$PWD/.cache/triton" TOKENIZERS_PARALLELISM=false GRADIO_ANALYTICS_ENABLED=False \
  VOXBOOKS_MODEL="$PWD/models/VoxCPM2" VOXBOOKS_PROJECTS="$PWD/projects"
exec .venv/bin/python -m audiobook "${@:-ui}"
