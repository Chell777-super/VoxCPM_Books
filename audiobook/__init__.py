"""VoxCPM Books — генератор аудиокниг из PDF на базе VoxCPM2.

Модули:
    book        — структура книги (главы / абзацы), сохранение и загрузка
    pdf_extract — извлечение текста из PDF (колонтитулы, переносы, сноски, главы)
    normalize   — очистка и нормализация текста для TTS
    segmenter   — разбиение на сегменты по границам предложений
    engine      — обёртка над VoxCPM2 (оптимизации под CUDA, контроль качества)
    audio       — пост-обработка, склейка, экспорт через ffmpeg (MP3 / M4B)
    project     — проект книги на диске (манифест, возобновление)
    pipeline    — конвейер синтеза (GPU-поток + поток записи)
    webui       — веб-интерфейс на Gradio
    cli         — командная строка
"""

import sys as _sys
from pathlib import Path as _Path

__version__ = "0.1.0"

# Пакет voxcpm берём прямо из src/ этого репозитория (без pip install -e .),
# чтобы установка не зависела от git-метаданных и тяжёлых зависимостей.
_SRC = _Path(__file__).resolve().parent.parent / "src"
if (_SRC / "voxcpm").is_dir() and str(_SRC) not in _sys.path:
    _sys.path.insert(0, str(_SRC))
