#!/usr/bin/env bash
# PDF → «<имя> — разметка.txt» и «<имя> — озвучка.txt» рядом с файлом (без модели и GPU).
#   ./pdf2text.sh книга.pdf [ещё.pdf …]
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || { echo "Сначала запустите ./install.sh"; exit 1; }
[ $# -gt 0 ] || { echo "Использование: ./pdf2text.sh книга.pdf"; exit 1; }
export PYTHONUTF8=1
files=(); for f in "$@"; do files+=("$(realpath "$f")"); done
exec .venv/bin/python -m audiobook pdf2text "${files[@]}"
