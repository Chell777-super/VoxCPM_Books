#!/usr/bin/env bash
# VoxCPM Books — установщик для Ubuntu (22.04 / 24.04, в т.ч. WSL2) с Nano-vLLM.
#
#   ./install.sh            — Nano-vLLM (параллельный синтез, в разы быстрее) + обычный движок
#   ./install.sh --no-nano  — только обычный PyTorch-движок (последний torch)
#
# Что ставится: uv, Python 3.11 (.venv), PyTorch 2.8 + CUDA 12.8, готовый flash-attn,
# nano-vllm-voxcpm, зависимости приложения, модель VoxCPM2 (~5 ГБ) в models/VoxCPM2.
set -euo pipefail
cd "$(dirname "$0")"

NANO=1
for arg in "$@"; do
  case "$arg" in
    --no-nano) NANO=0 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
  esac
done

PY_VER="${PY_VER:-3.11}"           # nano-vllm-voxcpm: Python 3.10–3.12
NANO_VER="${NANO_VER:-2.0.5}"
export UV_LINK_MODE=copy
export HF_HOME="$PWD/.cache/huggingface"
step() { printf '\n\033[1;34m[%s]\033[0m %s\n' "$1" "$2"; }
warn() { printf '\033[1;33m⚠ %s\033[0m\n' "$*"; }
die()  { printf '\033[1;31m✖ %s\033[0m\n' "$*"; exit 1; }

# ---------------------------------------------------------------- 1. система
step 1/7 "Проверка системы"
IS_WSL=0; grep -qi microsoft /proc/version 2>/dev/null && IS_WSL=1
[ "$(uname -m)" = "x86_64" ] || warn "Архитектура $(uname -m): готовые сборки flash-attn есть только для x86_64"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
  if [ "$IS_WSL" = 1 ]; then
    die "nvidia-smi не найден. В WSL2 драйвер ставится в Windows (NVIDIA Game Ready/Studio), в Ubuntu — НЕ ставить."
  fi
  die "Нет драйвера NVIDIA. Установите: sudo ubuntu-drivers install && sudo reboot   (нужен драйвер >= 570)"
fi

SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null && SUDO="sudo"
NEED_APT=""
for pkg in ffmpeg libsndfile1 build-essential git curl; do
  dpkg -s "$pkg" >/dev/null 2>&1 || NEED_APT="$NEED_APT $pkg"
done
if [ -n "$NEED_APT" ]; then
  step 1/7 "Системные пакеты:$NEED_APT (gcc нужен Triton для компиляции ядер)"
  if [ -n "$SUDO" ] || [ "$(id -u)" -eq 0 ]; then
    $SUDO apt-get update -qq && $SUDO apt-get install -y -qq $NEED_APT
  else
    warn "Нет прав root — установите вручную: apt-get install -y$NEED_APT"
  fi
fi

# ---------------------------------------------------------------- 2. uv
step 2/7 "uv"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null || die "uv не установился"
export UV_EXE="$(command -v uv)"
uv --version

# ---------------------------------------------------------------- 3. venv
step 3/7 "Окружение Python $PY_VER"
[ -x .venv/bin/python ] || uv venv .venv --python "$PY_VER"
PY="$PWD/.venv/bin/python"
"$PY" -c 'import sys; v=sys.version_info; assert (3,10)<=v[:2]<=(3,12), v' 2>/dev/null \
  || { [ "$NANO" = 1 ] && die "Nano-vLLM требует Python 3.10–3.12, а в .venv другой. Удалите .venv и запустите снова."; true; }

# ---------------------------------------------------------------- 4. PyTorch (+ flash-attn)
mkdir -p .cache
if [ "$NANO" = 1 ]; then
  step 4/7 "PyTorch 2.8 + CUDA 12.8 (версия с готовым flash-attn) — около 3 ГБ"
  "$PY" installer/setup_helper.py torch-nano
  step 4/7 "flash-attn (готовая сборка, без компиляции)"
  "$PY" installer/setup_helper.py flash-attn
else
  step 4/7 "PyTorch (последний, под ваш драйвер)"
  "$PY" installer/setup_helper.py torch "${TORCH_CHANNEL:-latest}"
fi
"$PY" installer/setup_helper.py constraints .cache/constraints.txt

# ---------------------------------------------------------------- 5. зависимости
step 5/7 "Зависимости приложения"
uv pip install --python "$PY" -r installer/requirements-books.txt -c .cache/constraints.txt
if [ "$NANO" = 1 ]; then
  step 5/7 "Nano-vLLM-VoxCPM $NANO_VER"
  # torchcodec в nano объявлен, но не используется; ограничиваем версией, совместимой с torch 2.8
  uv pip install --python "$PY" "nano-vllm-voxcpm==$NANO_VER" "torchcodec<0.8" xxhash -c .cache/constraints.txt
fi

# ---------------------------------------------------------------- 6. модель
step 6/7 "Модель VoxCPM2 (~5 ГБ, один раз)"
"$PY" installer/setup_helper.py download models/VoxCPM2

# ---------------------------------------------------------------- 7. проверка
step 7/7 "Проверка"
"$PY" installer/setup_helper.py check || true
chmod +x run.sh pdf2text.sh 2>/dev/null || true
cat <<'EOF'

  ==========================================================
    Готово!
    Веб-интерфейс:       ./run.sh            (http://127.0.0.1:7861)
    Командная строка:    ./run.sh --help
    PDF → текст:         ./pdf2text.sh книга.pdf
    Обычный движок:      ./run.sh --backend torch ui
  ==========================================================
EOF
