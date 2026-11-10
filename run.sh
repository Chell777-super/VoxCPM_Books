#!/usr/bin/env bash
# VoxCPM Books — запуск на Linux.
#   ./run.sh                          веб-интерфейс (Nano-vLLM, если установлен)
#   ./run.sh --backend torch ui       веб-интерфейс на обычном PyTorch-движке
#   ./run.sh --concurrency 24 ui      больше фрагментов одновременно (если хватает видеопамяти)
#   ./run.sh synth книга --voice Диктор
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Сначала запустите ./install.sh"; exit 1; }
export PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false GRADIO_ANALYTICS_ENABLED=False HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME="$PWD/.cache/huggingface" TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor" TRITON_CACHE_DIR="$PWD/.cache/triton"
export NANOVLLM_CACHE_DIR="$PWD/.cache/nanovllm"
export VOXBOOKS_MODEL="$PWD/models/VoxCPM2" VOXBOOKS_PROJECTS="$PWD/projects" VOXBOOKS_VOICES="$PWD/voices"
export VOXBOOKS_BACKEND="${VOXBOOKS_BACKEND:-auto}"
# если команда не указана (только опции вроде --concurrency 24) — запускаем веб-интерфейс
has_cmd=0
for a in "$@"; do
  case "$a" in ui|new|text|synth|export|make|status|pdf2text|voice-add|voices|-h|--help) has_cmd=1 ;; esac
done
[ "$has_cmd" = 1 ] || set -- "$@" ui
exec .venv/bin/python -m audiobook "$@"
